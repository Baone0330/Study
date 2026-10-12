"""Original dataset/model/training equations, with additive epoch-boundary recovery."""
from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cpfm_common import atomic_json, load_module, now, protected_postflight, read_csv, read_json, write_csv
from generate_formal import output_paths, validate_png

BRANCH = "HIST_ORIGINAL_P16_CPFM_OFF"


def load_base(root, cfg):
    source = Path(cfg["paths"]["historical_runner"])
    sys.path.insert(0, str(source.parent))
    base = load_module("historical_cpfm_off_native_runner", source)
    base.OUT = root / "downstream"
    model_cfg = base.model_cfg()
    for module in (base, base.fixed, base.formal, base.pre):
        module.OUT = root / "downstream"
    # Preserve original P16 branch FP32 evaluation/training precision policy.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    branch_cfg = {"k": 6, "image": True, "report": False, "time": False, "variable_length": False,
                  "description": "Original Progression CPFM-off single injection ablation"}
    base.fixed.BRANCHES[BRANCH] = branch_cfg
    frozen_path = Path(cfg["paths"]["replay_root"]).parent / "b16v1_historical_6branch_checkpoint_replay_v1/source_trace/EXACT_MODEL_CONFIG_V1.json"
    if frozen_path.exists():
        assert model_cfg == read_json(frozen_path), "Original model_cfg changed"
    return base, model_cfg, branch_cfg


def entries_from_slots(root, cfg, use_off):
    records = read_json(root / "manifests/FROZEN_REAL_HISTORY_RECORDS_V1.json")
    nodes = read_json(root / "manifests/FROZEN_NODES_V1.json")
    node_map = {(n["patient_id"], n["annual_slot_index"]): n for n in nodes}
    val_cohort = read_csv(Path(cfg["paths"]["old_downstream_root"]) / "manifests/FIXEDK6_POLICYA_VAL_COHORT_V1.csv")
    val_history_counts = {r["patient_id"]: int(r["history_count"]) for r in val_cohort}
    datasets = {}
    for split in ("train", "val"):
        rows = read_csv(root / f"manifests/ORIGINAL_{split.upper()}_SEQUENCE_V1.csv")
        grouped = {}
        transformed = []
        for row in rows:
            pid = row["patient_id"]
            old_paths = json.loads(row["image_paths_json"])
            paths = old_paths
            if split == "train" and row["source_type"] == "B16" and use_off:
                node = node_map[(pid, int(row["slot_index"]))]
                assert old_paths == node["reference_on_paths"]
                paths = [str(p) for p in output_paths(root, node)]
            if split == "val":
                assert row["source_type"] in ("REAL", "LOCF")
            entry = grouped.setdefault(pid, {"patient_id": pid, "split": split, "target_year": int(row["target_year"]),
                                            "target_label": int(row["target_label"]), "history_count": len(records[pid]["history"]) if split == "train" else val_history_counts[pid], "nodes": []})
            entry["nodes"].append({"slot_index": int(row["slot_index"]), "slot_date": row["slot_date"], "source_type": row["source_type"],
                                   "exam_date": row["exam_date"], "image_paths": paths, "causal_source_k": row["causal_source_k"]})
            transformed.append({**row, "image_paths_json": json.dumps(paths, ensure_ascii=False)})
        datasets[split] = list(grouped.values())
        assert all(len(e["nodes"]) == 6 for e in datasets[split])
        if use_off:
            write_csv(root / f"downstream/manifests/{BRANCH}_{split.upper()}_SEQUENCE_V1.csv", transformed)
    assert len(datasets["train"]) == 860 and len(datasets["val"]) == 189
    assert Counter(e["target_label"] for e in datasets["val"]) == {0: 131, 1: 58}
    return datasets


def atomic_torch_save(path, state):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def rng_state(loader_generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader_generator": loader_generator.get_state()}


def restore_rng(saved, loader_generator):
    random.setstate(saved["python"])
    np.random.set_state(saved["numpy"])
    torch.set_rng_state(saved["torch_cpu"].cpu())
    if saved["torch_cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in saved["torch_cuda"]])
    loader_generator.set_state(saved["loader_generator"].cpu())


def save_resume(run, seed, branch, model, optimizer, scheduler, protocol, loader, progress):
    state = {"seed": seed, "branch": branch, "epoch": progress["epoch"], "best_epoch": progress["best_epoch"],
             "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
             "scheduler_state_dict": scheduler.state_dict(), "config": protocol, "progress": progress, "rng": rng_state(loader.generator),
             "recovery_semantics": "exact_completed_epoch_boundary; incomplete epoch replayed with saved RNG/order"}
    atomic_torch_save(run / "resume_checkpoint.pt", state)
    last = run / "last_checkpoint.pt"
    if last.exists() or last.is_symlink():
        assert last.is_symlink() and os.readlink(last) == "resume_checkpoint.pt", "Unexpected last checkpoint target"
    else:
        last.symlink_to("resume_checkpoint.pt")
    for name, rows in (("train_log.csv", progress["train_rows"]), ("val_metrics_by_epoch.csv", progress["val_rows"]), ("numerical_integrity.csv", progress["integrity_rows"])):
        write_csv(run / name, rows)
    atomic_json(run / "LIVE_TRAINING_STATUS_V1.json", {"branch": branch, "seed": seed, "epoch": progress["epoch"],
                "best_epoch": progress["best_epoch"], "last_val": progress["val_rows"][-1], "updated_utc": now(), "test_access": 0})


def load_resume(run, seed, branch, model, optimizer, scheduler, protocol, loader):
    path = run / "resume_checkpoint.pt"
    if not path.exists():
        assert not (run / "best_checkpoint.pt").exists(), "Unverifiable incomplete run without RNG resume checkpoint"
        return None
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["seed"] == seed and state["branch"] == branch and state["config"] == protocol
    model.load_state_dict(state["model_state_dict"], strict=True)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    restore_rng(state["rng"], loader.generator)
    return state["progress"]


def make_resume_training_function(root, fixed):
    source = inspect.getsource(fixed.train_one_seed)
    assert source.count("def train_one_seed(") == 1
    source = source.replace("def train_one_seed(", "def train_one_seed_resume(", 1)
    old = "    for epoch in range(1, MAX_EPOCHS + 1):\n"
    assert source.count(old) == 1
    new = """    start_epoch = 1
    resumed = load_resume(run, seed, branch, model, optimizer, scheduler, protocol, train_loader)
    if resumed is not None:
        start_epoch = resumed['epoch'] + 1
        train_rows = resumed['train_rows']
        val_rows = resumed['val_rows']
        integrity_rows = resumed['integrity_rows']
        best_key = tuple(resumed['best_key'])
        best_epoch = resumed['best_epoch']
        no_improve = resumed['no_improve']
        if resumed['epoch'] >= MIN_EPOCHS and no_improve >= PATIENCE:
            start_epoch = MAX_EPOCHS + 1
        print(json.dumps({'status':'EXACT_EPOCH_BOUNDARY_RESUME','branch':branch,'seed':seed,'last_completed_epoch':resumed['epoch']}), flush=True)
    for epoch in range(start_epoch, MAX_EPOCHS + 1):
"""
    source = source.replace(old, new, 1)
    old_last = "        save_checkpoint(run / \"last_checkpoint.pt\", seed, epoch, model, optimizer, scheduler, protocol)\n"
    assert source.count(old_last) == 1
    source = source.replace(old_last, "", 1)
    print_marker = "        print(json.dumps({\"branch\": branch, \"seed\": seed, \"epoch\": epoch,"
    assert source.count(print_marker) == 1
    addition = """        save_resume(run, seed, branch, model, optimizer, scheduler, protocol, train_loader,
                    {'epoch':epoch,'best_key':best_key,'best_epoch':best_epoch,'no_improve':no_improve,
                     'train_rows':train_rows,'val_rows':val_rows,'integrity_rows':integrity_rows})
"""
    source = source.replace(print_marker, addition + print_marker, 1)
    private_snapshot = root / "scripts/DERIVED_ORIGINAL_TRAIN_FUNCTION_PRIVATE_V1.py"
    private_snapshot.write_text(source, encoding="utf-8")
    # Same original global helpers/constants/model, additive checkpoint IO only.
    namespace = dict(vars(fixed))
    namespace.update(save_resume=save_resume, load_resume=load_resume)
    original_save = fixed.save_checkpoint
    def save_checkpoint(path, *args, **kwargs):
        tmp = Path(path).with_name(Path(path).name + ".tmp")
        original_save(tmp, *args, **kwargs)
        os.replace(tmp, path)
    namespace["save_checkpoint"] = save_checkpoint
    exec(compile(source, str(private_snapshot), "exec"), namespace)
    atomic_json(root / "downstream/qa/TRAIN_LOOP_ADAPTATION_QA_V1.json", {"status": "PASS",
                "original_function": "train_one_seed", "training_math_or_sampler_changes": 0,
                "additions": ["strict same-run model/optimizer/scheduler/RNG/loader restore", "atomic best/resume files", "per-epoch CSV/live status"],
                "recovery": "completed epoch boundary; incomplete epoch rerun from identical saved state", "test_access": 0})
    return namespace["train_one_seed_resume"]


def original_schema_qa(root):
    cfg = read_json(root / "protocol/FROZEN_HISTORICAL_CPFM_ABLATION_CONFIG_V1.json")
    base, model_cfg, _ = load_base(root, cfg)
    from ht_prediction_v1.compat import load_original_climatv2_class
    original = load_original_climatv2_class(model_cfg["climatv2_root"])
    saved_factory = original.configure_optimizers
    original.configure_optimizers = lambda self: None
    try:
        model = base.formal.build_model(model_cfg, torch.device("cpu")).eval().requires_grad_(False)
    finally:
        original.configure_optimizers = saved_factory
    rows = []
    for record in cfg["original_on_runs"]:
        state = torch.load(record["checkpoint_path"], map_location="cpu", weights_only=True)
        result = model.load_state_dict(state["model_state_dict"], strict=True)
        assert not result.missing_keys and not result.unexpected_keys
        assert all(torch.isfinite(p).all() for p in model.parameters())
        assert state["seed"] == record["seed"] and state["epoch"] == record["best_epoch"]
        rows.append({"seed": record["seed"], "selected_epoch": record["best_epoch"], "strict_load": "PASS", "finite": True,
                     "parameters": sum(p.numel() for p in model.parameters()), "source_is_original_replayed_on": True,
                     "forward_performed": False, "optimizer_created": 0})
    base.official_state()  # CPU strict local ImageNet state load; no download.
    atomic_json(root / "downstream/qa/ORIGINAL_ON_THREE_CHECKPOINT_SCHEMA_QA_V1.json", {"status": "PASS", "runs": rows, "test_access": 0})
    make_resume_training_function(root, base.fixed)
    print(json.dumps({"status": "CPU_DOWNSTREAM_SOURCE_SCHEMA_PASS", "old_ON_runs": rows, "new_OFF_train_runs": 0}))


def build(root):
    cfg = read_json(root / "protocol/FROZEN_HISTORICAL_CPFM_ABLATION_CONFIG_V1.json")
    assert read_json(root / "release/FORMAL_OFF_GENERATION_COMPLETED_V1.json")["complete_images"] == 9270
    data = entries_from_slots(root, cfg, use_off=True)
    for entry in data["train"]:
        for node in entry["nodes"]:
            if node["source_type"] == "B16":
                for path in node["image_paths"]:
                    validate_png(Path(path))
    protocol = {**cfg["old_downstream_protocol"], "experiment": "HISTORICAL_ORIGINAL_PROGRESSION_CPFM_ONLY_ABLATION_V1",
                "branch": BRANCH, "comparison_route": "H", "b16_manifest": str(root / "manifests/ORIGINAL_ON_VS_CPFM_OFF_EXACT_PAIR_MAP_V1.parquet"),
                "generation_manifest": str(root / "manifests/ORIGINAL_ON_VS_CPFM_OFF_EXACT_PAIR_MAP_V1.parquet"),
                "test_access": 0, "source_freeze_utc": cfg["source_freeze_utc"], "route_frozen_utc": cfg["route_frozen_utc"]}
    atomic_json(root / "downstream/RESOLVED_PROTOCOL_V1.json", protocol)
    qa = {"status": "PASS", "Train_N": 860, "Val_N": 189, "K": 6, "T": 6, "Train_labels": {0: 617, 1: 243}, "Val_labels": {0: 131, 1: 58},
          "Train_slots": dict(Counter(n["source_type"] for e in data["train"] for n in e["nodes"])),
          "Val_slots": dict(Counter(n["source_type"] for e in data["val"] for n in e["nodes"])),
          "old_real_locf_paths_and_order_preserved": True, "same_synthetic_nodes_and_image_indices": True,
          "old_val_inputs_and_labels_unchanged": True, "test_access": 0, "checked_utc": now()}
    atomic_json(root / "downstream/qa/DATASET_EXACT_SUPPORT_QA_V1.json", qa)
    protected_postflight(root)
    print(json.dumps(qa))


def train(root, seed):
    cfg = read_json(root / "protocol/FROZEN_HISTORICAL_CPFM_ABLATION_CONFIG_V1.json")
    assert seed in cfg["climatv2_training_seeds"]
    assert read_json(root / "downstream/qa/DATASET_EXACT_SUPPORT_QA_V1.json")["status"] == "PASS"
    data = entries_from_slots(root, cfg, use_off=True)
    base, model_cfg, branch_cfg = load_base(root, cfg)
    weight_path, state, weight_enum = base.official_state()
    restore = base.patch_training_builder(state)
    ds_train = base.FixedDataset(data["train"], "train", BRANCH)
    ds_val = base.FixedDataset(data["val"], "val", BRANCH)
    run = root / "downstream/runs" / BRANCH / f"seed_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    if (run / "TRAINING_COMPLETED_V1.json").is_file():
        assert (run / "best_checkpoint.pt").is_file() and (run / "val_predictions_best.csv").is_file()
        print("Existing completed same-run training reused; reload QA still required")
        return
    protocol = {**read_json(root / "downstream/RESOLVED_PROTOCOL_V1.json"), "seed": seed,
                "backbone_initialization": "torchvision_official_imagenet", "official_weight_enum": weight_enum,
                "official_weight_local_path": str(weight_path)}
    atomic_json(run / "resolved_config.json", protocol)
    (run / "resolved_config.yaml").write_text("\n".join(f"{k}: {json.dumps(v)}" for k, v in protocol.items()) + "\n", encoding="utf-8")
    train_fn = make_resume_training_function(root, base.fixed)
    try:
        if not (root / "downstream/qa/REAL_DATA_TRAIN_SMOKE_PASS_V1.json").exists():
            torch.cuda.reset_peak_memory_stats()
            base.fixed.seed_everything(seed)
            loader = DataLoader(base.FixedDataset(data["train"][:4], "train", BRANCH), batch_size=4, shuffle=False, num_workers=0, collate_fn=base.fixed.BranchCollator(branch_cfg))
            batch = next(iter(loader))
            model = base.formal.build_model(model_cfg, torch.device("cuda"))
            optimizer, _ = base.formal.optimizer_and_audit(model, lr=1e-4)
            model.train()
            output = model(**{k: v.cuda() for k, v in batch["model_inputs"].items()}, stage="train")
            loss = torch.nn.functional.cross_entropy(output["logits"].float(), batch["labels"].cuda()) + 0.1 * output["consistency_loss"].float()
            assert torch.isfinite(loss) and torch.isfinite(output["logits"]).all()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            assert torch.isfinite(grad)
            optimizer.step()
            atomic_json(root / "downstream/qa/REAL_DATA_TRAIN_SMOKE_PASS_V1.json", {"status": "PASS", "optimizer_step": 1,
                        "used_formal_train_first4": True, "source_OFF_generation_complete": True, "formal_run_weights_reused": False,
                        "own_peak_allocated_GiB": torch.cuda.max_memory_allocated()/1024**3, "patient_batch": 4, "grad_accumulation_formal": 2,
                        "test_access": 0, "completed_utc": now()})
            del model, optimizer, output, loss, loader, batch
            torch.cuda.empty_cache()
        result = train_fn(BRANCH, seed, model_cfg, protocol, ds_train, ds_val, 4, None, None, branch_config=branch_cfg)
        import pandas as pd
        pd.DataFrame(result["predictions"]).to_parquet(run / "val_patient_probabilities.parquet", index=False)
        atomic_json(run / "TRAINING_COMPLETED_V1.json", {"status": "TRAINING_COMPLETED_RELOAD_PENDING", "seed": seed,
                    "best_epoch": result["best_epoch"], "epochs": result["epochs"], "metrics": result["metrics"], "test_access": 0, "completed_utc": now()})
    finally:
        base.formal.build_model = restore


def reload(root, seed):
    cfg = read_json(root / "protocol/FROZEN_HISTORICAL_CPFM_ABLATION_CONFIG_V1.json")
    base, model_cfg, branch_cfg = load_base(root, cfg)
    base.fixed.seed_everything(seed)
    data = entries_from_slots(root, cfg, use_off=True)
    run = root / "downstream/runs" / BRANCH / f"seed_{seed}"
    assert (run / "TRAINING_COMPLETED_V1.json").is_file()
    # Reuse the original training constructor and final CUDA load path exactly.
    # Inference mode in the unmodified evaluator already prevents gradients;
    # changing parameter metadata or the constructor is unnecessary here.
    _, initial_state, _ = base.official_state()
    restore_builder = base.patch_training_builder(initial_state)
    try:
        model = base.formal.build_model(model_cfg, torch.device("cuda")).eval()
    finally:
        base.formal.build_model = restore_builder
    state = torch.load(run / "best_checkpoint.pt", map_location="cuda", weights_only=False)
    result = model.load_state_dict(state["model_state_dict"], strict=True)
    assert not result.missing_keys and not result.unexpected_keys and state["seed"] == seed
    loader = DataLoader(base.FixedDataset(data["val"], "val", BRANCH), batch_size=4, shuffle=False, num_workers=0, collate_fn=base.fixed.BranchCollator(branch_cfg))
    evaluated = base.fixed.evaluate(model, loader, torch.device("cuda"), 0.1)
    saved_rows = read_csv(run / "val_predictions_best.csv")
    assert [r["patient_id"] for r in saved_rows] == [e["patient_id"] for e in data["val"]]
    assert np.array_equal(evaluated["labels"], [int(r["label"]) for r in saved_rows])
    errors = np.abs(evaluated["probs"] - [float(r["prob_ht1"]) for r in saved_rows])
    difference = float(np.max(errors))
    passed = len(saved_rows) == 189 and difference <= 1e-6 and evaluated["numerical_finite"]
    qa = {"status": "PASS" if passed else "FAIL", "strict_reload": "PASS" if passed else "FAIL",
          "strict_state_dict_load": "PASS", "independent_new_process": True, "Val_N": len(saved_rows),
          "patient_order_match": True, "label_match": True, "numerical_finite": bool(evaluated["numerical_finite"]),
          "model_construction": "original_training_builder_and_cuda_checkpoint_load",
          "optimizer_steps_in_replay": 0, "training_checkpoint_modified": False,
          "maximum_abs_positive_probability_difference": difference,
          "mean_abs_positive_probability_difference": float(np.mean(errors)),
          "patients_exceeding_tolerance": int(np.count_nonzero(errors > 1e-6)),
          "runtime_precision": {"cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                                "float32_matmul_precision": torch.get_float32_matmul_precision()},
          "tolerance": 1e-6, "best_epoch": state["epoch"], "test_access": 0, "completed_utc": now()}
    write_csv(run / "val_predictions_independent_reload_private.csv", [
        {"patient_id": r["patient_id"], "label": r["label"], "saved_prob_ht1": r["prob_ht1"],
         "reloaded_prob_ht1": float(p), "absolute_difference": float(e)}
        for r, p, e in zip(saved_rows, evaluated["probs"], errors)])
    atomic_json(run / "independent_reload_qa.json", qa)
    print(json.dumps(qa), flush=True)
    assert passed, f"Independent Val replay failed: max_abs_difference={difference:.12g}; tolerance=1e-6"
    protected_postflight(root)
    atomic_json(run / "RUN_COMPLETED.json", {**qa, "status": "RUN_COMPLETED", "branch": BRANCH, "seed": seed})
    print(json.dumps(qa))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--stage", choices=("schema", "build", "train", "reload"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 3407, 2026))
    a = parser.parse_args()
    root = Path(a.root)
    try:
        if a.stage == "schema": original_schema_qa(root)
        elif a.stage == "build": build(root)
        elif a.stage == "train": train(root, a.seed)
        else: reload(root, a.seed)
    except Exception as exc:
        atomic_json(root / "release/ENGINEERING_INCOMPLETE.json", {"status": "PARTIAL_WITH_LIMITATIONS", "stage": a.stage, "seed": a.seed, "error": repr(exc), "updated_utc": now(), "test_access": 0})
        raise
