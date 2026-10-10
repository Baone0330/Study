from __future__ import annotations

from pathlib import Path

from .common import DISPLAY, SourceGuard, read_json, write_frame, write_json


def discover(config, out=None):
    """Selection is proven from original saved logs, never from replay scores."""
    import pandas as pd
    import yaml
    guard = SourceGuard()
    rows, upstream, exclusions = [], [], []
    for branch, display in zip(config["branches"], DISPLAY):
        root = Path(branch["root"])
        upstream_config = branch.get("upstream_config")
        identity = read_json(guard.track(upstream_config)) if upstream_config else {"generator": "NONE"}
        upper = branch.get("upstream_checkpoint")
        if upper:
            guard.track(upper)  # Provenance only, never model input or state.
        upstream.append(dict(branch=branch["branch_id"], original_name=branch["internal_name"],
                             config_path=upstream_config, checkpoint_path=upper,
                             original_progression_training_seed=branch.get("upstream_training_seed"),
                             identity=identity, upstream_forward_executed=False))
        selected = set()
        for seed in branch["seeds"]:
            run = root / "runs" / branch["internal_name"] / f"seed_{seed}"
            checkpoint = run / "best_checkpoint.pt"
            selected.add(checkpoint.resolve())
            row = dict(branch=branch["branch_id"], branch_display_name=display,
                       original_branch_name=branch["internal_name"], seed=int(seed),
                       source_run_dir=str(run), checkpoint_path=str(checkpoint),
                       model_source=config["model_runner"], eval_script=branch["eval_script"],
                       historical_manifest=branch["val_manifest"],
                       historical_val_ids=branch["cohort_manifest"],
                       original_log=str(run / "val_metrics_by_epoch.csv"),
                       upstream_variant=branch["upstream_variant"],
                       matmul_allow_tf32=bool(branch["matmul_allow_tf32"]),
                       existence=checkpoint.is_file(), best_epoch=None, evidence_notes="UNKNOWN",
                       eligibility="UNRESOLVED")
            try:
                meta = read_json(guard.track(run / "best_checkpoint_metadata.json"))
                completed = read_json(guard.track(run / "RUN_COMPLETED.json"))
                cfg_path = guard.track(run / "resolved_config.yaml")
                cfg = yaml.safe_load(cfg_path.read_text())
                log = pd.read_csv(guard.track(row["original_log"]), float_precision="round_trip")
                best = max(log.to_dict("records"), key=lambda x: (
                    x["ROC_AUC"], x["PR_AUC"], -x["Val_Loss"], -x["epoch"]))
                if (cfg["seed"] != seed or completed["status"] != "RUN_COMPLETED"
                        or int(meta["best_epoch"]) != int(completed["best_epoch"])
                        or int(best["epoch"]) != int(meta["best_epoch"])):
                    raise ValueError("Original seed or selected epoch evidence disagrees")
                if cfg["precision"] != "fp32" or cfg["K"] != 6 or cfg["T"] != 6:
                    raise ValueError("Historical input/precision contract differs")
                guard.track(row["eval_script"])
                guard.track(row["historical_manifest"])
                guard.track(row["historical_val_ids"])
                guard.track(checkpoint)
                row.update(best_epoch=int(meta["best_epoch"]), resolved_config_path=str(cfg_path),
                           original_model_hparams=cfg, eligibility="SELECTED_BEST_PROVEN",
                           evidence_notes="Original resolved seed + completed marker + saved best metadata + full epoch log")
            except (OSError, ValueError, KeyError) as error:
                row["evidence_notes"] = repr(error)
            rows.append(row)
        for candidate in (root / "runs" / branch["internal_name"]).rglob("*.pt"):
            if candidate.resolve() not in selected:
                exclusions.append(dict(branch=branch["branch_id"], candidate_path=str(candidate),
                                       reason="Not the historically selected-best checkpoint; never loaded"))
    if out is not None:
        out = Path(out)
        write_json(out / "configs_private/HISTORICAL_RUN_INDEX_V1.json", rows)
        write_frame(out / "audit/HISTORICAL_SOURCE_OF_TRUTH_V1.csv",
                    pd.DataFrame([{k: v for k, v in row.items() if k != "original_model_hparams"} for row in rows]))
        write_frame(out / "audit/CHECKPOINT_CANDIDATE_EXCLUSIONS_V1.csv", pd.DataFrame(exclusions))
        write_json(out / "audit/OLD_UPSTREAM_WEIGHT_MAPPING_V1.json", upstream)
        write_frame(out / "audit/OLD_UPSTREAM_WEIGHT_MAPPING_V1.csv", pd.DataFrame([
            {k: v for k, v in row.items() if k != "identity"} for row in upstream]))
        guard.save(out / "audit")
    return rows, guard
