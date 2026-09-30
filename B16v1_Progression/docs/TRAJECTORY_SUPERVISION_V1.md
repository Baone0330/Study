# Trajectory-priority supervision research candidate

This optional release provides three full-development S2 ProgressionCondition
weights. It **does not replace the default inference weights** and is not an
independently validated or clinical model. Development cross-fit is an internal
representation study, not independent generalization evidence.

Only coefficients of the existing teacher MSE/cosine, historical next-delta
MSE/cosine, and compatibility MSE terms change. Raw feature differences,
16→32→LayerNorm→GELU→16 Delta Encoder, valid-pair masked mean, condition fusion,
16-D adapter, ZC transmission, CPFM, diffusion model, medical LoRA, and
downstream predictor are unchanged. No new trajectory loss is introduced.

S2 uses `(0.25, 0.05, 1.20, 0.30, 0.10)` in that term order; the sum is 1.90.
Equal coefficient sums do not imply equal gradient norms. The direction-first
candidate has an alignment/amplitude trade-off and is not claimed to dominate
the original supervision across all representation metrics.

## Model loading

With `B16v1_Progression/src` on the Python path:

```python
from b16v1_progression.trajectory_supervision import load_trajectory_supervision_candidate

model = load_trajectory_supervision_candidate(
    "B16v1_Progression/weights/trajectory_supervision_v1/"
    "trajectory_supervision_candidate_seed42.pt"
)
# before/after/temporal: [batch, 16]
# deltas: [batch, historical_pairs, 16]
# delta_mask: [batch, historical_pairs], valid pairs = 1, padding = 0
# condition16, progression16, fused64 = model(before, after, temporal, deltas, delta_mask)
```

The release also includes seed 3407 and seed 2026; no best-seed selection was
performed. All files are raw parameter state_dicts, without optimizer state,
patient identifiers, split membership, clinical paths, or logs.

## Supervision interface

`existing_supervision_terms` and `weighted_supervision_loss` reproduce the five
original loss components with the fixed S0–S3 coefficients. Teacher and
compatibility targets must come from the unchanged frozen upstream condition
construction; historical next-delta is `F_(k+1) - F_k`. This module neither builds
data nor launches training. Final-target exam representations or clinical
labels must not be used to construct these supervision inputs.

Full-development fitting uses all legal historical pre-target examples and
frozen epoch budgets 48/54/37 for seeds 42/3407/2026. The public config records
the optimization policy; private patient splits, OOF representations,
diagnostics, bootstrap results, and execution scripts are deliberately absent.

Future work should hold S2 fixed when testing a single architectural factor,
retain S0 as a reference, and obtain independent holdout/external/future-cohort
evidence before making generalization or clinical claims.
