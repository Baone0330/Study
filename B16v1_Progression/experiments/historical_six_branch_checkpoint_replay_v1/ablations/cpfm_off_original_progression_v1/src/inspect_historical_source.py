"""Read-only lineage recovery plus isolated audit/config writes; no Test reader."""
from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import os
import shutil
import socket
import sys
from collections import Counter, defaultdict
from pathlib import Path

from cpfm_common import atomic_json, load_module, now, read_csv, read_json, safe_root, stat_record, write_csv


def inspect(root, data_root):
    root = safe_root(root)
    data_root = Path(data_root).resolve()
    ab = data_root / "B16_final/b16v1_ablation_progression_v1"
    old_gen = ab / "generation/progression"
    ht = data_root / "CLIMATv2-main/ht_prediction_v1"
    old_down = ab / "downstream/progression"
    replay = data_root / "B16_final/b16v1_historical_six_branch_replay_and_github_v1"
    paper = data_root / "B16_final/b16v1_historical_six_branch_paper_package_v1"
    cond = data_root / "Stable_diffusion/longitudinal_ultrasound_generation/aatug_real_longitudinal_condition_integration_v15_4"
    cpfm = data_root / "Stable_diffusion/longitudinal_ultrasound_generation/aatug_patient_specific_rollout_supervision_v17_22"
    g1 = data_root / "data/HT_2025.12_25_g1_real_longitudinal_history_v2"
    identity = old_gen / "manifests/B16_LONGITUDINAL_ONLY_CAP2_SYNTHETIC_SLOT_IDENTITY_V1.csv"
    formal = old_gen / "manifests/B16_LONGITUDINAL_ONLY_CAP2_FORMAL_MANIFEST_V1.csv"
    mc = ht / "b16_mixture_control_trainval_v1/manifests/MC_CAP2_RANK_TRAIN_SEQUENCE_V1.csv"
    expanded = data_root / "Stable_diffusion/HT_2025.12_25_k6_policyA_b16_current_train_expansion_v1/manifests/B16_CURRENT_K6_POLICYA_EXPANDED_FORMAL_MANIFEST_V1.csv"
    train_path = old_down / "manifests/FIXEDK6_G4_B16_TRAIN_SEQUENCE_V1.csv"
    val_path = old_down / "manifests/FIXEDK6_SHARED_VAL_SEQUENCE_V1.csv"
    canonical_val = ht / "b16v1_experiment1_real_vs_synthetic_cap2_rerun_v1/manifests/FIXEDK6_SHARED_VAL_SEQUENCE_V1.csv"
    sources = [ab / "scripts/b16v1_progression_generator_v1.py", ab / "scripts/b16v1_ablation_generator_v1.py",
               ab / "scripts/b16v1_progression_train_v1.py", ab / "scripts/b16v1_ablation_downstream_v1.py",
               cond / "longitudinal_condition.py", cpfm / "run_v17_19.py",
               ht / "fixedk6_g1a_vs_g4_trainval_runner_v1.py",
               ht / "g4_b16_annual_completion_trainval_v1/scripts/g1_shortcut_mitigation_v1.py",
               ht / "g4_b16_annual_completion_trainval_v1/scripts/g1_formal_trainval_v1.py",
               ht / "models/climatv2_ht_wrapper.py", ht / "compat.py",
               data_root / "CLIMATv2-main/prog_oai/models/climatv2.py"]
    paths = dict(original_generator=str(sources[0]), base_generator=str(sources[1]), original_trainer=str(sources[2]),
                 old_downstream_entry=str(sources[3]), condition_source=str(sources[4]), cpfm_source=str(sources[5]),
                 historical_runner=str(sources[6]), historical_fixed_trainer=str(sources[7]),
                 condition_checkpoint=str(cond / "training/best_val.pt"),
                 progression_checkpoint=str(ab / "progression_training/checkpoints/best_progression_condition_v1.pt"),
                 cpfm_checkpoint=str(data_root / "Stable_diffusion/HT_2025.5_24届_patient_split_v1/modeling_v1/aatug_patient_specific_rollout_supervision_v17_22_rerun1_20260809/checkpoints/selected_for_full_replay/cpfm.pt"),
                 base_model=str(data_root / "Stable_diffusion/models/stable-diffusion-v1-5"),
                 lora=str(data_root / "Stable_diffusion/sd_lora_output_1epoch"), old_generation_root=str(old_gen),
                 old_downstream_root=str(old_down), replay_root=str(replay), paper_root=str(paper))
    for value in paths.values():
        assert Path(value).exists(), value
    config_path = root / "protocol/FROZEN_HISTORICAL_CPFM_ABLATION_CONFIG_V1.json"
    if config_path.exists():
        raise RuntimeError("Frozen source contract already exists; use postflight, do not rebuild")
    for rel in ("audit", "protocol", "manifests", "source_snapshot", "scripts", "pilot/traces", "pilot/metrics", "pilot/qa",
                "generation/CPFM_OFF", "generation/ON_REGEN_IF_REQUIRED", "downstream/runs", "downstream/qa", "metrics", "tables", "logs", "release"):
        (root / rel).mkdir(parents=True, exist_ok=True)
    rows = read_csv(formal)
    identities = read_csv(identity)
    mc_rows = read_csv(mc)
    old_slots = read_csv(train_path)
    val_rows = read_csv(val_path)
    val_ref = read_csv(canonical_val)
    assert len(rows) == 9270 and len(identities) == 1545 and len(mc_rows) == len(old_slots) == 5160
    assert len(val_rows) == len(val_ref) == 1134
    for a, b in zip(val_rows, val_ref):
        for field in ("patient_id", "target_year", "target_label", "slot_index", "slot_date", "source_type", "exam_date", "causal_source_k"):
            assert a[field] == b[field], "Original Val source contract differs"
        assert json.loads(a["image_paths_json"]) == json.loads(b["image_paths_json"])
    train_ids = {r["patient_id"] for r in old_slots}
    val_ids = {r["patient_id"] for r in val_rows}
    assert len(train_ids) == 860 and len(val_ids) == 189 and not (train_ids & val_ids)
    assert set(train_ids) == {r["patient_id"] for r in mc_rows}
    label_train = {r["patient_id"]: int(r["target_label"]) for r in old_slots}
    label_val = {r["patient_id"]: int(r["target_label"]) for r in val_rows}
    assert Counter(label_train.values()) == {0: 617, 1: 243} and Counter(label_val.values()) == {0: 131, 1: 58}
    grouped = defaultdict(list)
    for row in rows:
        assert row["split"] == "train" and row["node_status"] == "COMPLETE"
        assert row["ablation_variant"] == "progression" and int(row["progression_active"]) == 1
        grouped[(row["patient_id"], int(row["annual_slot_index"]))].append(row)
    full_rows = read_csv(expanded)
    full_seeds = {(r["patient_id"], int(r["k"]), int(r["image_index"])): int(r["seed"]) for r in full_rows}
    raw_records = {}
    train_jsonl = g1 / "manifests/train/G1_V2_TRAIN_SEQUENCE.jsonl"
    with train_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            pid = str(raw["patient_id"])
            if pid not in train_ids:
                continue
            history = [{"exam_date": str(e["exam_date"])[:10], "exam_year": int(e.get("exam_year", str(e["exam_date"])[:4])),
                        "image_paths": list(e["image_paths"])} for e in raw["history"]]
            assert all(e["source_type"] == "REAL" for e in raw["history"])
            assert all(e["exam_year"] < int(raw["target_year"]) for e in history)
            raw_records[pid] = {"patient_id": pid, "target_year": int(raw["target_year"]),
                                "target_reference_date": str(raw["target_reference_date"])[:10], "history": history}
    assert set(raw_records) == train_ids
    nodes = []
    lineage = []
    pair_map = []
    image_paths = set()
    from PIL import Image
    for item in identities:
        key = (item["patient_id"], int(item["annual_slot_index"]))
        found = sorted(grouped[key], key=lambda r: int(r["image_index"]))
        assert len(found) == 6 and [int(r["image_index"]) for r in found] == list(range(6))
        seeds = [int(item[f"seed_{i}"]) for i in range(6)]
        assert seeds == [int(r["seed"]) for r in found]
        assert seeds == [full_seeds[(key[0], key[1], i)] for i in range(6)]
        assert all(r["query_date"][:10] == item["query_date"][:10] for r in found)
        hist = [e for e in raw_records[key[0]]["history"] if e["exam_date"] < item["query_date"][:10]]
        assert hist
        node = {"patient_id": key[0], "annual_slot_index": key[1], "query_date": item["query_date"][:10],
                "allocation_rank": int(item["allocation_rank"]), "predictor_position": int(item["predictor_position"]),
                "target_year": int(item["target_year"]), "node_id": item["node_id"], "seeds": seeds,
                "reference_on_paths": [r["image_path"] for r in found]}
        nodes.append(node)
        for i, row in enumerate(found):
            path = Path(row["image_path"])
            assert path.is_file() and path.stat().st_size > 0
            with Image.open(path) as im:
                assert im.size == (512, 512) and im.mode == "RGB"
                im.verify()
            image_paths.add(path)
            new_path = root / "generation/CPFM_OFF" / f"patient_{key[0]}" / f"year_{key[1]:03d}_GENERATED/images" / f"gen_{i:02d}.png"
            pair_map.append({"historical_node_key": item["node_id"], "patient_key_private": key[0], "nominal_slot_id": key[1],
                             "image_index": i, "master_seed": seeds[i], "reference_ON_image_path": str(path),
                             "new_OFF_image_path": str(new_path), "source_type": "SYNTH", "original_G1_train_patient_membership": True})
            lineage.append({**pair_map[-1], "query_date_private": node["query_date"], "history_exam_dates_private": json.dumps([e["exam_date"] for e in hist]),
                            "source_identity": "Original Progression 20260921", "allocation_rank": node["allocation_rank"]})
    assert len(nodes) == len(grouped) == 1545
    # Preserve actual original generation order. Pilot selection uses explicit stable node-key order.
    assert nodes == sorted(nodes, key=lambda n: (n["allocation_rank"], n["patient_id"], n["annual_slot_index"]))
    slot_map = []
    synthetic_keys = set()
    for row in old_slots:
        key = (row["patient_id"], int(row["slot_index"]))
        slot_paths = json.loads(row["image_paths_json"])
        if row["source_type"] == "B16":
            assert key in grouped
            assert slot_paths == [r["image_path"] for r in sorted(grouped[key], key=lambda r: int(r["image_index"]))]
            synthetic_keys.add(key)
        else:
            assert row["source_type"] in ("REAL", "LOCF")
        image_paths.update(Path(p) for p in slot_paths)
        slot_map.append({**row, "source_priority": "REAL>SYNTH>LOCF", "image_count": len(slot_paths),
                         "assigned_synthetic_node": next((n["node_id"] for n in nodes if (n["patient_id"], n["annual_slot_index"]) == key), "") if row["source_type"] == "B16" else "",
                         "original_seed_set": json.dumps([int(r["seed"]) for r in sorted(grouped[key], key=lambda r: int(r["image_index"]))]) if row["source_type"] == "B16" else "[]"})
    assert synthetic_keys == set(grouped), "Generation/downstream synthetic support differs"
    for row in val_rows:
        assert row["source_type"] in ("REAL", "LOCF")
        image_paths.update(Path(p) for p in json.loads(row["image_paths_json"]))
    for r in raw_records.values():
        for exam in r["history"]:
            image_paths.update(Path(p) for p in exam["image_paths"])
    run_index = read_json(replay / "configs_private/HISTORICAL_RUN_INDEX_V1.json")
    original_runs = [r for r in run_index if r["branch"] == "B16V1_PLUS_ORIGINAL_PROGRESSION"]
    assert sorted(r["seed"] for r in original_runs) == [42, 2026, 3407]
    assert {r["seed"]: r["best_epoch"] for r in original_runs} == {42: 48, 3407: 10, 2026: 14}
    snapshots = []
    anchors = []
    for index, path in enumerate(sources):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        snap = root / "source_snapshot" / f"{index:02d}_{path.name}"
        shutil.copy2(path, snap)
        snapshots.append({"source_path": str(path), "private_snapshot_path": str(snap)})
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                anchors.append({"role": path.stem, "absolute_source_path": str(path), "symbol": node.name,
                                "line_range": f"{node.lineno}-{node.end_lineno}", "old_checkpoint_reference": "see frozen config",
                                "old_config_reference": str(old_gen / "configs/B16V1_PROGRESSION_GENERATION_CONFIG_V1.json"),
                                "actual_runtime_activity": "source inspection; no clinical forward yet", "notes": "original source unchanged"})
    write_csv(root / "audit/ORIGINAL_SOURCE_ANCHORS_V1.csv", anchors)
    write_csv(root / "audit/HISTORICAL_SLOT_AND_SEED_LINEAGE_V1.csv", lineage)
    write_csv(root / "source_snapshot/INDEX_V1.csv", snapshots)
    atomic_json(root / "manifests/FROZEN_NODES_V1.json", nodes)
    atomic_json(root / "manifests/FROZEN_REAL_HISTORY_RECORDS_V1.json", raw_records)
    write_csv(root / "manifests/ORIGINAL_TRAIN_SEQUENCE_V1.csv", old_slots)
    write_csv(root / "manifests/ORIGINAL_VAL_SEQUENCE_V1.csv", val_rows)
    import pandas as pd
    pd.DataFrame(nodes).to_parquet(root / "manifests/MASTER_HISTORICAL_SYNTHETIC_NODES_V1.parquet", index=False)
    pd.DataFrame(pair_map).to_parquet(root / "manifests/ORIGINAL_ON_VS_CPFM_OFF_EXACT_PAIR_MAP_V1.parquet", index=False)
    pd.DataFrame(slot_map).to_parquet(root / "manifests/TRAIN_K6_SOURCE_SLOT_MAP_V1.parquet", index=False)
    configs = [old_gen / "configs/B16V1_PROGRESSION_GENERATION_CONFIG_V1.json", old_gen / "configs/B16_LONGITUDINAL_ONLY_GENERATION_CONFIG_V1.json",
               ab / "progression_training/configs/PROGRESSION_ENCODER_CONFIG_V1.json", old_down / "configs/RESOLVED_PROTOCOL_V1.json"]
    protected = set(sources + configs + [identity, formal, mc, expanded, train_path, val_path, canonical_val, train_jsonl,
                                       g1 / "manifests/val/G1_V2_VAL_SEQUENCE.jsonl", replay / "configs_private/HISTORICAL_RUN_INDEX_V1.json"])
    protected.update(Path(paths[k]) for k in ("condition_checkpoint", "progression_checkpoint", "cpfm_checkpoint"))
    for model_dir in (Path(paths["base_model"]), Path(paths["lora"])):
        protected.update(p for p in model_dir.rglob("*") if p.is_file())
    for item in original_runs:
        protected.add(Path(item["checkpoint_path"]))
        protected.add(Path(item["resolved_config_path"]))
    protected.update(image_paths)
    metadata = [stat_record(p) for p in sorted(protected)]
    atomic_json(root / "audit/PROTECTED_SOURCE_METADATA_V1.json", metadata)
    write_csv(root / "audit/PROTECTED_SOURCE_METADATA_V1.csv", metadata)
    # CPU strict loads only. Source models never create an optimizer.
    import torch
    mod = load_module("cpfm_off_original_p16", sources[2])
    sys.path.insert(0, str(cond))
    from longitudinal_condition import LongitudinalConditionModel
    legacy = load_module("cpfm_off_original_b14", sources[5])
    models = [("CurrentF16", LongitudinalConditionModel(), paths["condition_checkpoint"]),
              ("OriginalProgression", mod.ProgressionCondition(), paths["progression_checkpoint"]),
              ("B14CPFM", legacy.B14CPFM(), paths["cpfm_checkpoint"])]
    model_qa = []
    for name, model, ckpt in models:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        result = model.load_state_dict(state.get("model", state), strict=True)
        model.eval().requires_grad_(False)
        assert all(torch.isfinite(p).all() for p in model.parameters())
        model_qa.append({"name": name, "tensors": len(model.state_dict()), "parameters": sum(p.numel() for p in model.parameters()),
                         "strict_load": "PASS", "finite": True, "eval": not model.training,
                         "requires_grad_count": sum(p.requires_grad for p in model.parameters()), "device": "cpu", "saved_epoch": state.get("epoch")})
    assert model_qa[1]["saved_epoch"] == 120
    progression_cfg = read_json(configs[2])
    assert progression_cfg["training"]["seed"] == 20260921
    versions = {}
    for package in ("torch", "torchvision", "diffusers", "transformers", "peft", "numpy", "Pillow", "scikit-learn", "pandas", "pyarrow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "UNKNOWN"
    old_config = read_json(configs[0])
    assert old_config["progression_active"] and old_config["cpfm_active"]
    config = {"protocol": "HISTORICAL_ORIGINAL_PROGRESSION_CPFM_ONLY_ABLATION_V1", "task_identity": "HISTORICAL_ORIGINAL_PROGRESSION_CPFM_OFF_MATCHED_ABLATION_V1",
              "paths": paths, "reference_model": "Original Progression 20260921 epoch120, not S2/Clean",
              "comparison_route": "UNRESOLVED_BEFORE_PILOT", "generator_nodes": len(nodes), "generator_images": len(rows),
              "generator_unique_patients": len({n["patient_id"] for n in nodes}), "images_per_generated_node": 6,
              "sampling_scheduler": "ORIGINAL_PNDM_EPSILON", "requested_inference_steps": 30, "guidance_scale": 7.0,
              "resolution": [512, 512], "dtype": "fp16", "original_generation_batch": 6,
              "lora_override": "NONE", "cpfm_off_operation": "ORIGINAL_HOLDER_ENABLED_GATE_BYPASS",
              "old_val_support": 189, "train_support": 860, "climatv2_training_seeds": [42, 3407, 2026],
              "bootstrap_seed": 20261012, "n_resamples": 2000, "clinical_test_status": "NOT_RUN",
              "old_generation_config": old_config, "old_downstream_protocol": read_json(configs[3]),
              "original_on_runs": original_runs, "runtime_versions": versions, "historical_runtime_exact_revision": "UNKNOWN",
              "python": sys.executable, "hostname": socket.gethostname(), "source_freeze_utc": now(),
              "pilot_selection": "first 6 by stable original node_id, all 6 seeds; no label or quality selection",
              "pilot_node_ids": [n["node_id"] for n in sorted(nodes, key=lambda n: n["node_id"])[:6]],
              "train_tf32": False, "cudnn_tf32": True, "historical_generation_tf32_flag": "UNKNOWN_DEFAULT_SOURCE_NO_OVERRIDE"}
    atomic_json(config_path, config)
    qa = {"status": "SOURCE_CONTRACT_RECOVERED", "generator_nodes": len(nodes), "images": len(rows),
          "generator_unique_patients": config["generator_unique_patients"], "train_patients": len(train_ids), "val_patients": len(val_ids),
          "train_source_slots": dict(Counter(r["source_type"] for r in old_slots)),
          "val_source_slots": dict(Counter(r["source_type"] for r in val_rows)),
          "exact_old_synthetic_slot_paths": "PASS", "old_formal_png_decode_qa": "9270/9270 PASS",
          "original_identity_vs_formal_vs_full_seed_lineage": "PASS", "canonical_val_paths_order_labels": "PASS",
          "protected_files": len(metadata), "model_schema_qa": model_qa, "test_content_access": 0,
          "upstream_optimizer_created": 0, "new_images_generated": 0, "downstream_not_run": True, "checked_utc": now()}
    atomic_json(root / "audit/PHASE1_SOURCE_CONTRACT_QA_V1.json", qa)
    write_csv(root / "manifests/RUN_SUPPORT_MATCHING_AUDIT_V1.csv", [{"item": k, "observed": v, "status": "PASS"} for k, v in qa.items() if isinstance(v, (str, int))])
    atomic_json(root / "release/EXECUTION_STATE_V1.json", {"status": "PHASE1_COMPLETED_PILOT_PENDING", "phase": 1, "updated_utc": now(), "test_access": 0})
    print(json.dumps(qa, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--data-root", required=True)
    args = parser.parse_args()
    try:
        inspect(args.root, args.data_root)
    except Exception as exc:
        atomic_json(Path(args.root) / "release/ENGINEERING_INCOMPLETE.json", {"status": "SOURCE_CONTRACT_UNRESOLVED_STOPPED", "error": repr(exc), "updated_utc": now(), "test_access": 0})
        raise
