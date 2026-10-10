# Aggregate results only

`historical_six_branch_metrics_public.csv` is populated ONLY after actual
checkpoint-forward inference and original three-seed ensemble evaluation.
`replay_coverage_public.json` reports actual execution coverage, not the
existence of archived predictions. The reference JSON in configs is NOT a new
measured result. No patient-level record, source path, log or checkpoint belongs
in this directory.
