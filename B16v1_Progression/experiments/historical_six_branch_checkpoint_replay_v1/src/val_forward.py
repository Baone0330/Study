from __future__ import annotations

import time
import traceback
from pathlib import Path

from .common import (SourceGuard, configure_inference_environment, load_config, metadata,
                     read_json, utc_now, write_frame, write_json)
from .checkpoint_loader import safe_checkpoint
from .historical_dataset_adapter import create_dataset, original_entries
from .historical_model_factory import create_original_model, install_inference_guards


def replay_one(config_path, out, branch_id, seed):
    configure_inference_environment()
    import torch
    import torchvision
    import sklearn
    import pandas as pd
    from torch.utils.data import DataLoader
    out = Path(out)
    config = load_config(config_path)
    index = read_json(out / "configs_private/HISTORICAL_RUN_INDEX_V1.json")
    row = next(x for x in index if x["branch"] == branch_id and int(x["seed"]) == int(seed))
    branch = next(x for x in config["branches"] if x["branch_id"] == branch_id)
    run = out / "predictions_private" / branch_id / f"seed_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    guard = SourceGuard()
    for item in read_json(out / "audit/PROTECTED_SOURCE_METADATA_V1.json"):
        guard.records[item["path"]] = item
    started = time.perf_counter()
    counters = None
    write_json(run / "status.json", dict(status="REPLAY_RUNNING", branch=branch_id, seed=seed, utc=utc_now()))
    try:
        if guard.check()["status"] != "PASS":
            raise RuntimeError("Source metadata no longer matches preflight")
        if row["eligibility"] != "SELECTED_BEST_PROVEN":
            raise RuntimeError("Historical selected-best provenance unresolved")
        if config["device"] != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Validated historical CUDA route is unavailable; no silent CPU fallback")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = bool(branch["matmul_allow_tf32"])
        device = torch.device("cuda")
        entries = original_entries(config, guard)
        counters = install_inference_guards()
        model, base, cfg = create_original_model(config, out / "configs_private/native_scratch", guard, device)
        # Reassert the evidenced route after all original import-time code.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = bool(branch["matmul_allow_tf32"])
        checkpoint = safe_checkpoint(guard.track(row["checkpoint_path"]), row)
        loaded = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise RuntimeError("Strict state loading was incomplete")
        model.requires_grad_(False)
        model.eval()
        if any(module.training for module in model.modules()) or any(p.requires_grad for p in model.parameters()):
            raise RuntimeError("Model not frozen/eval")
        if not all(p.dtype == torch.float32 for p in model.parameters()):
            raise RuntimeError("Historical FP32 parameter contract violated")
        dataset = create_dataset(base, entries, guard)
        collator = base.fixed.BranchCollator(dict(k=6, image=True, report=False, time=False, variable_length=False))
        loader = DataLoader(dataset, batch_size=4, shuffle=False, drop_last=False, num_workers=0,
                            collate_fn=collator)
        records, shapes = [], []
        with torch.inference_mode():
            for batch in loader:
                inputs = batch["model_inputs"]
                if (inputs["visit_valid_mask"].shape[1] != 6 or not inputs["visit_valid_mask"].all()
                        or inputs["report_features"].any() or inputs["time_features"].any()):
                    raise RuntimeError("Original image-only K6 inputs changed")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                logits = model(**inputs, stage="val")["logits"].float()
                if logits.shape != (len(batch["patient_ids"]), 2) or not torch.isfinite(logits).all():
                    raise RuntimeError("Invalid binary logits")
                probabilities = torch.softmax(logits, dim=-1)[:, 1]
                if not torch.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
                    raise RuntimeError("Invalid positive probabilities")
                shapes.append(list(inputs["images"].shape))
                for i, patient in enumerate(batch["patient_ids"]):
                    records.append(dict(patient_id=str(patient), label=int(batch["labels"][i]),
                                        prob_ht1=float(probabilities[i]), logit_negative=float(logits[i, 0]),
                                        logit_positive=float(logits[i, 1]), branch_id=branch_id, seed=int(seed),
                                        best_epoch=row["best_epoch"], prediction_source="CHECKPOINT_FORWARD_THIS_EXECUTION"))
        frame = pd.DataFrame(records)
        if len(frame) != 189 or frame.patient_id.nunique() != 189 or sum(frame.label) != 58:
            raise RuntimeError("Full historical Val output completeness failed")
        if any(counters.values()):
            raise RuntimeError("Replay guard violation")
        result = guard.check()
        if result["status"] != "PASS":
            raise RuntimeError("Protected source changed during replay")
        write_frame(run / "val_patient_probabilities.parquet", frame)
        write_json(run / "checkpoint_source.json", row)
        write_json(run / "protected_source_metadata.json", list(guard.records.values()))
        qa = dict(status="PASS", branch=branch_id, seed=int(seed), best_epoch=row["best_epoch"],
                  actual_model_class=type(model).__name__, state_dict_tensor_count=len(checkpoint["model_state_dict"]),
                  strict_load=True, missing_keys=[], unexpected_keys=[], all_modules_eval=True,
                  all_parameters_frozen=True, inference_mode=True, precision="fp32", batch=4, workers=0,
                  shuffle=False, forward_N=189, unique_patients=189, HT0=131, HT1=58, finite=True,
                  probability_range_valid=True, batch_shapes=shapes, image_reads=dataset.image_read_count,
                  original_model_config=cfg, matmul_allow_tf32=bool(branch["matmul_allow_tf32"]),
                  cudnn_allow_tf32=True, elapsed_seconds=time.perf_counter()-started,
                  counters=counters, source_metadata_guard=result, checkpoint_metadata=metadata(row["checkpoint_path"]),
                  environment=dict(python=__import__("sys").version, torch=torch.__version__,
                                   torchvision=torchvision.__version__, sklearn=sklearn.__version__,
                                   cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(0)), utc=utc_now())
        write_json(run / "reload_qa.json", qa)
        write_json(run / "status.json", dict(status="REPLAY_COMPLETED", branch=branch_id, seed=int(seed),
                                             prediction_source="CHECKPOINT_FORWARD_THIS_EXECUTION", utc=utc_now()))
        print(f"REPLAY_COMPLETED branch={branch_id} seed={seed} N=189", flush=True)
        return 0
    except Exception as error:
        category = "CUDA_DEVICE_FAILURE" if "CUDA" in str(error) and "out of memory" not in str(error) else "REPLAY_INCOMPLETE"
        write_json(run / "reload_qa.json", dict(status="FAIL", error=repr(error), traceback=traceback.format_exc(),
                                               category=category, counters=counters, utc=utc_now()))
        write_json(run / "status.json", dict(status="REPLAY_INCOMPLETE", branch=branch_id, seed=int(seed),
                                             error=repr(error), category=category, utc=utc_now()))
        traceback.print_exc()
        return 2
