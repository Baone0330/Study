# Source symbols and unchanged numerical behavior

| Component | Original source symbol | Release behavior |
|---|---|---|
| Resolved model | original fixed-K6 runner `model_cfg`; historical model-config snapshot | Exact in-memory dictionary comparison, no new hyperparameters |
| Model layers | original `CLIMATv2HTWrapper` and external native CLIMATv2 core | Original class, strict full checkpoint load; no vendored/reimplemented architecture |
| Exam pixels | original `FixedDataset._load` and `__getitem__` | Original RGB/antialiased resize256, /255, ImageNet normalization, all original views |
| Batch | original `BranchCollator` | Original masks, six visits, zero report/time inputs, no shuffle |
| REAL/Full threshold | Experiment 1 `best_youden` | Original CSV function, max(J, sensitivity, specificity, -threshold), >= positive |
| Plain-SD threshold | Experiment 2 `_binary_metrics` | Original array function, max(J, -threshold), >= positive |
| History/Temporal/Progression threshold | original ablation `metrics` | Original array function, max(J, -threshold), >= positive |
| Per-seed threshold | original fixed-loop `roc_curve` first max J | First maximum, same finite/clamp fallback; not the ensemble tie rule |
| ROC/AP QA | independent rank ROC and tied-score AP | Checked against sklearn and original confusion-based metrics |

Private source CSV/JSON preserve actual absolute locations and original
resolved configurations. Public documentation uses symbolic sources only.
Import paths and output globals are supplied by the caller; changing an output
destination is not changing forward computation. The original sources are not
edited. Constructor optimizer suppression is an inference-only runtime change,
not a new model or parameter initialization scheme.

Backend contract: cudnn deterministic=true, benchmark=false, cudnn TF32=true;
matmul TF32=false except the old Plain-SD route (true). The Plain-SD setting is
recovered from the original generation-to-training source chain and continuous
stage log. There was no independently captured historical flag snapshot; this
is explicitly source-evidenced recovery, not direct flag telemetry or an AUC
search. The restored setting is frozen BEFORE this task's first new forward.

The old Original Progression generator condition checkpoint is provenance-only;
it is not one of the 18 downstream classifiers and is not loaded into them.
Its historical upstream training seed is 20260921, subject to the private source
evidence check. No S2, clean F16, clean CPFM or new Strict assets are substituted.

This package does not redistribute original third-party/model snapshots or
their checkpoints. Original licenses and clinical access restrictions remain
applicable when supplying those external dependencies.
