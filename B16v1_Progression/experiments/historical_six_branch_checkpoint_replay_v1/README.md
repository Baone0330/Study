# Historical B16v1 six-branch CLIMATv2 checkpoint replay

Identity: `HISTORICAL_FIXED_CLIMATV2_INTERNAL_VAL_SIX_BRANCH_REPLAY`.

This is inference-only reproduction of six HISTORICAL branches, using each
branch's own three previously selected classifiers. It does not train models,
generate ultrasound, replace original weights, search checkpoints by replay AUC,
or use the later clean/Strict Pre-Target experiment.

Original clinical data, 18 classifier checkpoints, generator weights, original
model/helper snapshots and private configuration are NOT redistributed here.
You need legal access to those original assets. The public path template is
deliberately incomplete and fails closed; it is not a fake clinical dataset.
Original CLIMATv2 code remains an external import, not a newly invented model.

From this directory, using the already validated historical environment:

```bash
python -B -m unittest discover -s tests -v
python -B scripts/inspect_historical_runs.py --help
python -B scripts/inspect_historical_runs.py --config /secure/local_private.yaml --out /secure/replay --dry-run
python -B scripts/replay_six_branches.py --config /secure/local_private.yaml --out /secure/replay --replay
python -B scripts/aggregate_historical_six.py --config /secure/local_private.yaml --out /secure/replay --aggregate
python -B scripts/validate_release.py --repo-dir /secure/Study --work-dir /secure/replay --require-staged
```

The real validation status and aggregate metrics live under `results/`; source
locations and patient outputs stay in the private replay root. Commands require
`--config` and `--out`; no server-specific absolute path is a public default.
Exit 0 means the command finished. Individual inference failure is recorded in
private `status.json`; aggregation reports incomplete ensembles, never fills
them from archived predictions. A CUDA device failure stops further GPU work.

CLI `--dry-run` inspects source paths, original selected-epoch evidence and
metadata without importing torch or opening image pixels. A real replay is one
independent process per branch/seed, strict full-state load, FP32, batch four,
eval/frozen parameters and `torch.inference_mode()`. Successful outputs can be
resumed only under the same source identity; failed runs remain inspectable.

Three positive-class probabilities are aligned by patient key and averaged.
PR-AUC means `average_precision_score`, NOT trapezoidal PR integration.
Threshold policies differ by historical experiment and are executed from their
original source functions. The six-decimal reference is comparison-only; all
new numbers come from new checkpoint forwards.

Important: "Remove CPFM / Plain-SD" is a historical display label. The old
Plain-SD route disables additional B16-specific conditions and is not an
isolated CPFM intervention. The historical Full branch here is the
longitudinal History+Temporal+CPFM route, not an earlier report/pathology Full
model. Original Progression is not a later S2/clean replacement.

Historical Val was used for model and threshold selection. Numerical
reproduction does not establish independent external clinical validity,
end-to-end leakage freedom, or strict causal future-ultrasound prediction.
See `docs/REPRODUCTION_STEPS.md`, `SOURCE_AND_VERSION_MAPPING.md` and
`KNOWN_LIMITATIONS.md` for details.
