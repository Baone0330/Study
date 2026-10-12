"""Freeze identity-based cohorts before any quality score or visual selection."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from b16_quality_common import atomic_json, atomic_parquet, now, postflight, protect, read_json, setup, state, write_csv


def main(output_root, source_root, full_manifest, plain_manifest=None):
    root = setup(output_root)
    source = Path(source_root).resolve()
    cfg_path = root / "protocol/FROZEN_QUALITY_EVALUATION_CONTRACT_V1.json"
    if cfg_path.exists():
        cfg = read_json(cfg_path)
        assert cfg["source_root"] == str(source) and cfg["full_manifest"] == str(full_manifest)
        postflight(root)
        print("Existing frozen same-source cohorts retained")
        return
    assert read_json(source / "release/FINAL_CPFM_ONLY_ABLATION_STATUS_V1.json")["status"] == "HISTORICAL_CPFM_ONLY_ABLATION_COMPLETED_GITHUB_VERIFIED"
    source_cfg = read_json(source / "protocol/FROZEN_HISTORICAL_CPFM_ABLATION_CONFIG_V1.json")
    pair_path = source / "manifests/ORIGINAL_ON_VS_CPFM_OFF_EXACT_PAIR_MAP_V1.parquet"
    master_path = source / "manifests/MASTER_HISTORICAL_SYNTHETIC_NODES_V1.parquet"
    lineage_path = source / "audit/HISTORICAL_SLOT_AND_SEED_LINEAGE_V1.csv"
    real_slots_path = source / "manifests/TRAIN_K6_SOURCE_SLOT_MAP_V1.parquet"
    real_records_path = source / "manifests/FROZEN_REAL_HISTORY_RECORDS_V1.json"
    pairs = pd.read_parquet(pair_path)
    master = pd.read_parquet(master_path, columns=["patient_id", "annual_slot_index", "query_date", "node_id", "seeds", "reference_on_paths"])
    for col in ("patient_key_private", "historical_node_key"): pairs[col] = pairs[col].astype(str)
    master.patient_id = master.patient_id.astype(str)
    master.node_id = master.node_id.astype(str)
    assert len(pairs) == 9270 and len(master) == 1545
    assert not pairs.duplicated(["historical_node_key", "image_index"]).any()
    assert set(pairs.historical_node_key) == set(master.node_id)
    by_node = {r.node_id: r for r in master.itertuples()}
    lineage = pd.read_csv(lineage_path, dtype={"historical_node_key": str, "patient_key_private": str})
    identity_columns = ["historical_node_key", "patient_key_private", "nominal_slot_id", "image_index", "master_seed", "reference_ON_image_path", "new_OFF_image_path"]
    assert pairs[identity_columns].astype(str).equals(lineage[identity_columns].astype(str))
    pairs = pairs.sort_values(["patient_key_private", "nominal_slot_id", "historical_node_key", "image_index"], kind="stable").reset_index(drop=True)
    patients = sorted(pairs.patient_key_private.unique())
    assert len(patients) == 860
    patient_codes = {p: f"P{i+1:04d}" for i, p in enumerate(patients)}
    nodes = list(dict.fromkeys(pairs.historical_node_key))
    node_codes = {n: f"N{i+1:04d}" for i, n in enumerate(nodes)}
    paths_to_protect = [pair_path, master_path, lineage_path, real_slots_path, real_records_path,
                        source / "protocol/FROZEN_HISTORICAL_CPFM_ABLATION_CONFIG_V1.json",
                        source / "release/FINAL_CPFM_ONLY_ABLATION_STATUS_V1.json"]
    rows = []
    for row in pairs.itertuples():
        m = by_node[row.historical_node_key]
        assert m.patient_id == row.patient_key_private and int(m.annual_slot_index) == int(row.nominal_slot_id)
        assert int(row.master_seed) == int(m.seeds[int(row.image_index)])
        assert row.reference_ON_image_path == m.reference_on_paths[int(row.image_index)]
        for branch, path in (("OP_CPFM_ON", row.reference_ON_image_path), ("OP_CPFM_OFF", row.new_OFF_image_path)):
            if not Path(path).is_file(): raise FileNotFoundError("Primary image absent; do not silently discard node")
            paths_to_protect.append(path)
            rows.append({"image_id": f"{branch}:{node_codes[row.historical_node_key]}:{int(row.image_index)}",
                         "branch": branch, "anonymous_patient_key": patient_codes[row.patient_key_private],
                         "anonymous_node_key": node_codes[row.historical_node_key], "image_index": int(row.image_index),
                         "seed": int(row.master_seed), "clinical_role": "historical_synthetic_not_target_truth",
                         "patient_key_private": row.patient_key_private, "historical_node_key_private": row.historical_node_key,
                         "source_date_private": str(m.query_date), "real_exam_key_private": "", "png_abs_path": path,
                         "comparability": "EXACT_ON_OFF_IMAGE_SEED_MATCH"})
    slots = pd.read_parquet(real_slots_path, columns=["patient_id", "source_type", "exam_date", "image_paths_json"])
    slots.patient_id = slots.patient_id.astype(str)
    assert set(slots.patient_id) == set(patients)
    native = read_json(real_records_path)
    valid_native = {(str(pid), str(exam["exam_date"]), str(Path(p).resolve()))
                    for pid, record in native.items() for exam in record["history"] for p in exam["image_paths"]}
    views = {}
    for row in slots[slots.source_type == "REAL"].itertuples():
        for path in json.loads(row.image_paths_json):
            resolved = str(Path(path).resolve())
            if (row.patient_id, str(row.exam_date), resolved) not in valid_native:
                raise RuntimeError("REAL role/date/path not found in original native history")
            if not Path(path).is_file(): raise FileNotFoundError("Native reference image absent")
            key = (row.patient_id, str(row.exam_date), resolved)
            views.setdefault(key, path)
    for i, ((pid, date, resolved), path) in enumerate(sorted(views.items())):
        paths_to_protect.append(path)
        rows.append({"image_id": f"REAL_HISTORY_REFERENCE:R{i+1:06d}", "branch": "REAL_HISTORY_REFERENCE",
                     "anonymous_patient_key": patient_codes[pid], "anonymous_node_key": "", "image_index": -1,
                     "seed": -1, "clinical_role": "native_real_history_distribution_reference_not_query_truth",
                     "patient_key_private": pid, "historical_node_key_private": "", "source_date_private": date,
                     "real_exam_key_private": pid + "::" + date, "png_abs_path": path,
                     "comparability": "NATIVE_REAL_DATE_AND_ROLE_VERIFIED"})
    full_status = "NOT_FOUND"
    full_matches = 0
    full_rows_total = None
    if full_manifest and Path(full_manifest).is_file():
        paths_to_protect.append(full_manifest)
        full = pd.read_csv(full_manifest, usecols=["patient_id", "k", "query_date", "generated_image_path", "image_index", "seed"], dtype={"patient_id": str})
        full_rows_total = len(full)
        assert not full.duplicated(["patient_id", "k", "image_index"]).any()
        lookup = {(r.patient_id, int(r.k), int(r.image_index)): r for r in full.itertuples()}
        for row in pairs.itertuples():
            found = lookup.get((row.patient_key_private, int(row.nominal_slot_id), int(row.image_index)))
            if found is None or int(found.seed) != int(row.master_seed): continue
            m = by_node[row.historical_node_key]
            if str(found.query_date)[:10] != str(m.query_date)[:10] or not Path(found.generated_image_path).is_file(): continue
            full_matches += 1
            paths_to_protect.append(found.generated_image_path)
            rows.append({"image_id": f"FULL_B16V1_HIST:{node_codes[row.historical_node_key]}:{int(row.image_index)}",
                         "branch": "FULL_B16V1_HIST", "anonymous_patient_key": patient_codes[row.patient_key_private],
                         "anonymous_node_key": node_codes[row.historical_node_key], "image_index": int(row.image_index),
                         "seed": int(found.seed), "clinical_role": "auxiliary_historical_route_not_single_variable",
                         "patient_key_private": row.patient_key_private, "historical_node_key_private": row.historical_node_key,
                         "source_date_private": str(found.query_date), "real_exam_key_private": "",
                         "png_abs_path": found.generated_image_path, "comparability": "EXACT_IMAGE_SEED_MATCH_AUXILIARY_ROUTE"})
        full_status = "EXACT_IMAGE_SEED_MATCH" if full_matches == len(pairs) else "PARTIAL_MATCHED_METADATA_SUPPORT"
    # Plain is resolved only from an explicit historic manifest supplied after source discovery.
    plain_status = "SOURCE_CONTRACT_FOUND_MANIFEST_NOT_YET_QUALIFIED"
    if plain_manifest:
        plain_status = "NOT_INCLUDED_PENDING_NATIVE_NODE_SEED_SCHEMA_QUALIFICATION"
    frame = pd.DataFrame(rows)
    real = frame[frame.branch == "REAL_HISTORY_REFERENCE"].copy()
    common = sorted(set(real.anonymous_patient_key) & set(frame[frame.branch == "OP_CPFM_ON"].anonymous_patient_key))
    real_counts = real.groupby("anonymous_patient_key").size()
    m = min(6, int(real_counts.loc[common].min()))
    assert m >= 1
    rng = np.random.default_rng(20261012)
    balanced_ids = []
    for patient in common:
        on = frame[(frame.branch == "OP_CPFM_ON") & (frame.anonymous_patient_key == patient)].sort_values("image_id")
        selected = rng.choice(len(on), m, replace=False)
        on_ids = on.iloc[selected].image_id.tolist()
        balanced_ids.extend(on_ids)
        balanced_ids.extend(x.replace("OP_CPFM_ON:", "OP_CPFM_OFF:") for x in on_ids)
        fids = [x.replace("OP_CPFM_ON:", "FULL_B16V1_HIST:") for x in on_ids]
        balanced_ids.extend(x for x in fids if x in set(frame.image_id))
        real_ids = real[real.anonymous_patient_key == patient].sort_values("image_id").image_id.tolist()
        balanced_ids.extend(real_ids[j] for j in rng.choice(len(real_ids), m, replace=False))
    frame["patient_balanced_selected"] = frame.image_id.isin(balanced_ids)
    panel_indices = np.linspace(0, len(nodes)-1, 12, dtype=int)
    panel_rows = [{"panel_node_code": f"V{i+1:02d}", "anonymous_node_key": node_codes[nodes[j]],
                   "historical_node_key_private": nodes[j], "display_section": "MAIN_FIXED_FIRST6" if i < 6 else "SUPPLEMENT_FIXED_NEXT6"}
                  for i, j in enumerate(panel_indices)]
    write_csv(root / "manifests/PRECOMMITTED_VISUAL_PANEL_NODES_V1.csv", panel_rows)
    atomic_parquet(root / "manifests/OP_ON_OFF_EXACT_IMAGE_PAIRS_V1.parquet", pairs)
    atomic_parquet(root / "manifests/REAL_HISTORY_REFERENCE_V1.parquet", real)
    atomic_parquet(root / "manifests/QUALITY_EVAL_FROZEN_QUEUE_V1.parquet", frame)
    atomic_json(root / "manifests/PRIVATE_ANONYMOUS_PATIENT_MAP_V1.json", patient_codes)
    counts = [{"branch": b, "images": len(f), "patients": f.anonymous_patient_key.nunique(),
               "nodes": f[f.anonymous_node_key != ""].anonymous_node_key.nunique(),
               "native_exams": f[f.real_exam_key_private != ""].real_exam_key_private.nunique(),
               "patient_balanced_images": int(f.patient_balanced_selected.sum()), "alignment": full_status if b == "FULL_B16V1_HIST" else "PRIMARY"}
              for b, f in frame.groupby("branch", sort=False)]
    write_csv(root / "source_audit/SOURCE_IDENTITY_AND_COMPARABILITY_V1.csv", counts)
    base_protection = read_json(source / "audit/PROTECTED_SOURCE_METADATA_V1.json")
    paths_to_protect.extend(r["path"] for r in base_protection)
    protect(root, paths_to_protect)
    contract = {"task": "B16V1_HISTORICAL_ORIGINAL_PROGRESSION_GENERATION_QUALITY_V1", "source_root": str(source),
                "full_manifest": str(full_manifest), "plain_manifest": str(plain_manifest) if plain_manifest else None,
                "ON_images": len(pairs), "OFF_images": len(pairs), "matched_nodes": len(nodes), "matched_patients": len(patients),
                "real_unique_views": len(real), "real_unique_exams": int(real.real_exam_key_private.nunique()),
                "real_unique_patients": int(real.anonymous_patient_key.nunique()), "common_patients": len(common),
                "full_status": full_status, "full_matched_images": full_matches, "full_manifest_rows": full_rows_total,
                "plain_status": plain_status, "branch_counts": counts,
                "balanced_policy": {"m": m, "definition": "min(6,minimum native REAL unique view count among common patients)",
                                    "seed": 20261012, "common_patients": common, "ON_OFF_same_selected_pairs": True},
                "visual_selection": "12 equal-spaced indices in stable anonymous-patient/slot/node order; first6 main, next6 supplement; all6 seeds",
                "paired_pixel_preprocess": "native generated RGB512, no histogram alteration",
                "texture_preprocess": "common whole-image RGB512 PIL bilinear; aspect-ratio/resampling nuisance explicitly reported; native shape kept",
                "inception_preprocess": "torch-fidelity0.4.0 TF1-compatible internal bilinear299/no antialias and (uint8-128)/128; native RGB sizes grouped",
                "ssim": {"space": "BT.601 grayscale float64", "range": 255, "gaussian_weights": True, "sigma": 1.5,
                         "win_size": 11, "use_sample_covariance": False},
                "kid": {"degree": 3, "gamma": "1/2048", "coef0": 1, "subsets": 50, "subset_size": 512, "seed": 20261012, "negative_values_retained": True},
                "bootstrap": {"repeats": 2000, "seed": 20261012, "unit": "patient", "FID_CI": "NOT_RUN_RESOURCE_BOUNDED_DECLARATION",
                              "paired_technical_CI": True, "balanced_full_U_statistic_KID_delta_CI": True, "KID_MC_SD_is_not_CI": True},
                "no_quality_filtering": True, "clinical_expert_review": "NOT_RUN", "medical_visual_transfer_authorized": False,
                "new_training": 0, "new_SD_forward": 0, "official_test_access": 0, "frozen_before_quality_scores_utc": now()}
    atomic_json(cfg_path, contract)
    for name, text in {
        "REAL_REFERENCE_SELECTION_POLICY_V1.md": "# Real reference\n\nOnly native REAL rows with exact patient/date/resolved-path membership in frozen original history are included. LOCF duplicates are excluded by unique native-view identity. Reference images are historical context, not query-year truth. Outcomes and reports are not used. Patient-balanced selection uses the common support and objective minimum view count before metrics.\n",
        "PRECOMMITTED_VISUAL_SAMPLE_SELECTION_V1.md": "# Precommitted panels\n\nTwelve equally spaced nodes in stable source ordering were saved before any score. First six are main and next six supplementary, with all six seeds retained. Medical panels never leave the server. No previous viewing permission is reused.\n",
        "COMPARISON_INTERPRETATION_RULES_V1.md": "# Boundaries\n\nON/OFF differences measure injection-path dependence, not ground-truth reconstruction. FID/KID are natural-image feature distribution descriptors. Historical Full and Plain are auxiliary routes, not single-variable effects. Anatomy/clinical expert review and independent HT validation are not established.\n"
    }.items(): (root / "protocol" / name).write_text(text, encoding="utf-8")
    postflight(root)
    state(root, "COHORTS_FROZEN_BEFORE_METRICS", counts=counts)
    print(json.dumps({k: contract[k] for k in ("ON_images", "OFF_images", "matched_nodes", "matched_patients", "real_unique_views", "real_unique_exams", "real_unique_patients", "full_status", "full_matched_images", "plain_status")}, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", required=True)
    p.add_argument("--source-root", required=True)
    p.add_argument("--full-manifest", required=True)
    p.add_argument("--plain-manifest")
    a = p.parse_args()
    main(a.output_root, a.source_root, a.full_manifest, a.plain_manifest)
