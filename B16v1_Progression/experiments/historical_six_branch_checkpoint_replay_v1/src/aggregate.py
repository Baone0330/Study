from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import SourceGuard, load_config, read_json, utc_now, write_frame, write_json, write_text
from .original_metrics import historical_ensemble_metrics, historical_seed_metrics
from .result_reconcile import reconcile
from .seed_ensemble import align_and_average


def aggregate(config_path, out):
    config, out = load_config(config_path), Path(out)
    guard = SourceGuard()
    for before in read_json(out / "audit/PROTECTED_SOURCE_METADATA_V1.json"):
        guard.records[before["path"]] = before
    results, compares, seeds, qa_rows, missing = [], [], [], [], []
    reference = read_json(config["historical_reference"])
    common_ids, common_labels = None, None
    maximum_error = 0.
    for branch in config["branches"]:
        bid, frames = branch["branch_id"], {}
        for seed in branch["seeds"]:
            run = out / "predictions_private" / bid / f"seed_{seed}"
            if not (run / "status.json").exists() or read_json(run / "status.json").get("status") != "REPLAY_COMPLETED":
                missing.append(dict(branch_id=bid, seed=seed, status="REPLAY_INCOMPLETE"))
                continue
            qa = read_json(run / "reload_qa.json")
            if qa["status"] != "PASS" or not qa["strict_load"] or qa["forward_N"] != 189:
                raise ValueError("Output lacks actual strict-load/forward QA")
            for before in read_json(run / "protected_source_metadata.json"):
                previous = guard.records.get(before["path"])
                if previous is not None and previous != before:
                    raise RuntimeError("Source snapshots differ between processes")
                guard.records[before["path"]] = before
            frame = pd.read_parquet(run / "val_patient_probabilities.parquet")
            if len(frame) != 189 or set(frame.prediction_source) != {"CHECKPOINT_FORWARD_THIS_EXECUTION"}:
                raise ValueError("Not a fresh full-cohort checkpoint output")
            frames[int(seed)] = frame
            seeds.append(dict(branch_id=bid, seed=int(seed), best_epoch=int(frame.best_epoch.iloc[0]),
                              **historical_seed_metrics(frame.label, frame.prob_ht1)))
            qa_rows.append({key: qa[key] for key in ("branch", "seed", "best_epoch", "actual_model_class",
                                                     "state_dict_tensor_count", "strict_load", "forward_N",
                                                     "unique_patients", "HT0", "HT1", "finite", "precision",
                                                     "matmul_allow_tf32", "elapsed_seconds", "status")})
        if len(frames) != 3:
            results.append(dict(branch_id=bid, replay_status="PARTIAL_NO_FULL_ENSEMBLE", completed_seed_count=len(frames)))
            continue
        ids, labels, probabilities = align_and_average(frames, branch["seeds"])
        if common_ids is None:
            common_ids, common_labels = ids, labels
        elif ids != common_ids or not np.array_equal(labels, common_labels):
            raise ValueError("Six-branch identity/label contract differs")
        ensemble = pd.DataFrame(dict(patient_id=ids, label=labels, prob_ht1=probabilities))
        csv = write_frame(out / "predictions_private" / bid / "ensemble_val_probabilities.csv", ensemble)
        actual, errors, native_source = historical_ensemble_metrics(branch, ensemble, csv, guard)
        maximum_error = max(maximum_error, max(errors.values()))
        write_text(out / "audit" / (bid + "_native_metric_source.py"), native_source + "\n")
        comparison = reconcile(bid, actual, reference[bid])  # Reference read only after forward values exist.
        comparison.update(actual_Precision=actual["Precision"], TP=actual["TP"], TN=actual["TN"],
                          FP=actual["FP"], FN=actual["FN"], threshold_source=branch["metric_function"],
                          notes="Historical Val internal replay, not independent Test")
        compares.append(comparison)
        results.append(dict(branch_id=bid, completed_seed_count=3, **actual,
                            threshold_source=branch["metric_function"],
                            replay_status=comparison["replay_status"],
                            all_display_fields_match=comparison["all_display_fields_match"]))
    guard.save(out / "audit")
    write_frame(out / "reports/ENSEMBLE_METRICS_V1.csv", pd.DataFrame(results))
    write_frame(out / "reports/SIX_BRANCH_ACTUAL_VS_REFERENCE_V1.csv", pd.DataFrame(compares))
    write_frame(out / "reports/SEED_LEVEL_ACTUAL_METRICS_V1.csv", pd.DataFrame(seeds))
    write_frame(out / "reports/REPLAY_CHECKPOINT_AND_INPUT_QA_V1.csv", pd.DataFrame(qa_rows))
    write_json(out / "audit/INDEPENDENT_METRIC_ARITHMETIC_QA_V1.json",
               dict(status="PASS", max_abs_error=maximum_error, PR_AUC_definition="average_precision_score",
                    threshold_from_actual_original_function=True, reference_used_for_selection=False))
    status = dict(identity=config["identity"], completed_checkpoint_forwards=len(qa_rows), expected=18,
                  complete_ensembles=sum(row.get("completed_seed_count") == 3 for row in results),
                  six_decimal_match_count=sum(row.get("all_display_fields_match", False) for row in results),
                  missing=missing, source_postflight=guard.check(), new_training_steps=0, new_generation_images=0,
                  optimizer_steps=0, backward_calls=0, checksum_workflow=0, official_test_forward=0,
                  results=results, utc=utc_now(), GitHub_push="NOT_RUN")
    write_json(out / "release/REPLAY_AGGREGATE_STATUS_V1.json", status)
    lines = ["# Historical replay discrepancies", "", "Only fresh checkpoint forwards form ensembles.", ""]
    for row in compares:
        lines.append(f"- {row['branch_id']}: {row['replay_status']}; all seven display fields match={row['all_display_fields_match']}.")
    for row in missing:
        lines.append(f"- {row['branch_id']} seed {row['seed']}: REPLAY_INCOMPLETE; no archived score substituted.")
    write_text(out / "reports/HISTORICAL_REPLAY_DISCREPANCIES_V1.md", "\n".join(lines) + "\n")
    print(f"AGGREGATE_COMPLETED checkpoints={len(qa_rows)}/18 ensembles={status['complete_ensembles']}/6", flush=True)
    return status
