"""Frozen H main comparison. Old six-branch values are copied, never relabeled."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from cpfm_common import atomic_json, now, protected_postflight, read_csv, read_json, write_csv
from downstream_execution import BRANCH
from downstream_metric_utils import binary_metrics, paired_patient_bootstrap


def main(root):
    root = Path(root)
    cfg = read_json(root / "protocol/FROZEN_HISTORICAL_CPFM_ABLATION_CONFIG_V1.json")
    assert cfg["comparison_route"] == "H" and cfg["downstream_metrics_observed_before_route"] == 0
    seed_frames = []
    summaries = []
    for seed in (42, 3407, 2026):
        run = root / "downstream/runs" / BRANCH / f"seed_{seed}"
        assert read_json(run / "RUN_COMPLETED.json")["status"] == "RUN_COMPLETED"
        assert read_json(run / "independent_reload_qa.json")["status"] == "PASS"
        frame = pd.read_parquet(run / "val_patient_probabilities.parquet").assign(patient_id=lambda d: d.patient_id.astype(str)).sort_values("patient_id").reset_index(drop=True)
        assert len(frame) == frame.patient_id.nunique() == 189
        meta = read_json(run / "best_checkpoint_metadata.json")
        threshold = read_json(run / "val_decision_threshold.json")["threshold"]
        summaries.append({"branch_id": BRANCH, "seed": seed, "best_epoch": meta["best_epoch"], "formal_epochs": meta["formal_epochs_completed"],
                          "independent_reload_status": "PASS", **binary_metrics(frame.label, frame.prob_ht1, threshold),
                          "checkpoint_policy": "ROC_AUC,AP,-Val_Loss,-epoch", "threshold_policy": "original seed roc_curve argmax Youden"})
        seed_frames.append(frame)
    ids = seed_frames[0].patient_id.tolist()
    y = seed_frames[0].label.to_numpy(dtype=int)
    assert all(frame.patient_id.tolist() == ids and np.array_equal(frame.label, y) for frame in seed_frames)
    off = np.mean(np.stack([frame.prob_ht1.to_numpy(dtype=float) for frame in seed_frames]), axis=0)
    replay = Path(cfg["paths"]["replay_root"])
    on_frame = pd.read_csv(replay / "predictions_private/B16V1_PLUS_ORIGINAL_PROGRESSION/ensemble_val_probabilities.csv", dtype={"patient_id": str}, float_precision="round_trip").sort_values("patient_id").reset_index(drop=True)
    assert on_frame.patient_id.tolist() == ids and np.array_equal(on_frame.label, y)
    on = on_frame.prob_ht1.to_numpy(dtype=float)
    on_metrics, off_metrics = binary_metrics(y, on), binary_metrics(y, off)
    old_reference = read_csv(Path(cfg["paths"]["paper_root"]) / "tables/Table01_Historical_Six_Branch_Ensemble.csv")
    assert len(old_reference) == 6
    reference_on = next(r for r in old_reference if r["branch_id"] == "B16V1_PLUS_ORIGINAL_PROGRESSION")
    for name, value in on_metrics.items():
        if name in reference_on:
            assert abs(float(reference_on[name]) - value) < 1e-12, (name, "Archived ON arithmetic differs")
    private_out = root / "downstream/private_patient_probabilities"
    private_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"patient_id": ids, "label": y, "ON_prob_ht1": on, "OFF_prob_ht1": off}).to_parquet(private_out / "PAIRED_ON_OFF_ENSEMBLE_V1.parquet", index=False)
    write_csv(root / "tables/Table01_Historical_Six_Branch_Reference.csv", old_reference)
    write_csv(root / "tables/Table02_New_CPFM_OFF_Three_Seeds.csv", summaries)
    ensembles = [{"branch_id": "HIST_ORIGINAL_P16_CPFM_ON", "display_name": "Original Progression CPFM ON (historical matched reference)", **on_metrics},
                 {"branch_id": BRANCH, "display_name": "Original Progression CPFM OFF (new)", **off_metrics}]
    for row in ensembles:
        row.update(comparison_route="H", completed_seed_count=3, protocol="historical internal Val", threshold_rule="Val Youden J; smallest threshold on ties")
    write_csv(root / "tables/Table03_New_ON_OFF_Ensemble.csv", ensembles)
    bootstrap = paired_patient_bootstrap(y, on, off, n_resamples=2000, seed=20261012)
    write_csv(root / "tables/Table04_Paired_Patient_Bootstrap.csv", bootstrap)
    # Generator units stay separate: images -> nodes -> patients, then equal patient weights.
    db = sqlite3.connect(f"file:{root / 'generation/ledger.sqlite'}?mode=ro", uri=True)
    full = pd.DataFrame([json.loads(row[0]) for row in db.execute("SELECT metrics FROM images WHERE state='COMPLETE'").fetchall()])
    assert len(full) == 9270 and full.patient_id_private.nunique() == 860
    db.close()
    private_engineering = ["on_off_pixel_mae", "on_off_pixel_rmse", "on_off_generated_f16_rmse", "on_off_generated_f16_mae"]
    generator_rows = []
    for unit, frame in (("image", full), ("node", full.groupby("node_key_private")[private_engineering].mean()),
                        ("patient_equal_weight", full.groupby("patient_id_private")[private_engineering].mean())):
        for metric in private_engineering:
            values = frame[metric].to_numpy(dtype=float)
            generator_rows.append({"unit": unit, "metric": metric, "support_n": len(values), "mean": float(values.mean()),
                                   "median": float(np.median(values)), "q25": float(np.percentile(values, 25)), "q75": float(np.percentile(values, 75)),
                                   "minimum": float(values.min()), "maximum": float(values.max()), "measured_scope": "full exact matched RGB/F16 support"})
    pilot = pd.DataFrame(read_csv(root / "pilot/metrics/FIXED_36_PAIR_METRICS_V1.csv"))
    for metric in ("on_off_latent_rmse", "early_unet_uncond_rmse", "early_unet_cond_rmse"):
        values = pilot[metric].astype(float)
        generator_rows.append({"unit": "pilot_image", "metric": metric, "support_n": 36, "mean": float(values.mean()),
                               "median": float(values.median()), "q25": float(values.quantile(.25)), "q75": float(values.quantile(.75)),
                               "minimum": float(values.min()), "maximum": float(values.max()), "measured_scope": "fixed36 pilot, not full9270 latent replay"})
    write_csv(root / "tables/Table05_Generator_ON_OFF_Engineering_Differences.csv", generator_rows)
    public_engine = full[["on_off_pixel_mae", "on_off_pixel_rmse", "on_off_generated_f16_mae", "on_off_generated_f16_rmse", "on_off_pixel_bitexact"]]
    write_csv(root / "figures/source_csv/GENERATION_IMAGE_DIFFERENCES_ANONYMOUS_V1.csv", public_engine.to_dict("records"))
    update_fields = ["branch_id", "display_name", "ROC_AUC", "AP", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "Val_N", "TN", "FP", "FN", "TP", "threshold", "comparison_route", "new_run", "single_variable_match"]
    updated = [{**{name: r.get(name, "") for name in update_fields}, "comparison_route": "historical six-branch reference",
                "new_run": False, "single_variable_match": "not claimed for old six branches"} for r in old_reference]
    updated.append({**ensembles[1], "new_run": True, "single_variable_match": "H matched original P16 CPFM ON; ONLY original B14 gate"})
    write_csv(root / "tables/Table06_Updated_Module_Ablation_Table.csv", updated, update_fields)
    limitations = [
        {"topic": "Identity", "supported_claim": "Original P16 seed20260921 epoch120; old F16/B14 and historical K6", "prohibited_claim": "new Clean/Strict/S2 experiment"},
        {"topic": "CPFM injection", "supported_claim": "Original enabled gate bypass; P16 computation unchanged", "prohibited_claim": "P16 remains actively injected in OFF"},
        {"topic": "Module utility", "supported_claim": "Background-specific injection-path necessity/system dependence", "prohibited_claim": "fully independent P16 and CPFM causal decomposition"},
        {"topic": "ON parity", "supported_claim": "36/36 true ON pixels exact plus full source/seed/slot lineage", "prohibited_claim": "all9270 ON images resampled or exact old runtime revision recovered"},
        {"topic": "Statistics", "supported_claim": "189 patient-level paired stratified bootstrap2000; mean three-seed probabilities", "prohibited_claim": "567 independent patients or image-level clinical bootstrap"},
        {"topic": "Evaluation", "supported_claim": "historical internal Val previously used for checkpoint/threshold selection", "prohibited_claim": "independent clinical Test validation"},
        {"topic": "Medical realism", "supported_claim": "numerical, decode and distribution QA; all poor/zero outputs retained", "prohibited_claim": "medical expert reviewed progression validity"}]
    write_csv(root / "tables/Table07_Source_And_Claim_Limits.csv", limitations)
    fixed_threshold = binary_metrics(y, off, on_metrics["threshold"])
    atomic_json(root / "metrics/OFF_AT_OLD_ON_THRESHOLD_SENSITIVITY_V1.json", {"role": "additional descriptive sensitivity, not primary", **fixed_threshold})
    protected = protected_postflight(root)
    status = {"status": "MATCHED_INTERNAL_VAL_ABLATION_COMPLETED", "comparison_route": "H", "ON": on_metrics, "OFF": off_metrics,
              "ON_minus_OFF": {name: float(on_metrics[name] - off_metrics[name]) for name in ("ROC_AUC", "AP", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "TN", "FP", "FN", "TP")},
              "paired_bootstrap": bootstrap, "completed_runs": 3, "Val_N": 189, "Train_N": 860, "formal_images": 9270,
              "old_six_branch_values_unchanged": True, "protected_source_status": protected["status"], "intervention_path_dependent": True,
              "independent_clinical_validation": False, "test_access": 0, "medical_review": "NOT_RUN", "completed_utc": now()}
    atomic_json(root / "release/FINAL_METRICS_STATUS_V1.json", status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    a = p.parse_args()
    main(a.root)
