# DACP: experimental / not selected

This module archives the tested parameter-free Direction-Aware Consensus
Pooling (DACP) rule. **DACP was not selected to replace masked mean.** Retain
S2 supervision with the original masked mean for the existing research
candidate. No formal full-development DACP candidate weights are released,
and default inference, ZC/CPFM/diffusion/medical LoRA/downstream are unchanged.

The development mechanism experiment found that less vector cancellation
did not imply better historical next-direction alignment. This is internal
development evidence, not independent or clinical validation. Do not infer
that a larger pooled-vector norm establishes disease-progression correctness.

## Exact rule

For valid encoded transitions E_i, compute u_i=E_i/(norm(E_i)+1e-8), then
s_i=mean_{j!=i}(u_i dot u_j), a_i=(1+s_i)/2+1e-6, w_i=a_i/sum(a), and
P=sum(w_i E_i). Scoring uses directions; aggregation retains original
magnitudes. Gradients flow through the scores and weights. Padding is excluded.
There is no trainable pooling parameter, attention/query token, additional
network, temperature, recency weighting, or explicit history-count feature.

Zero valid transitions return zero; one returns E_1 exactly. For two, the
pairwise scores are symmetric, so both weights are 1/2 and the operator reduces
to mean. Symmetric direction sets cannot be forced to choose a direction.

Operator parity means **the same encoded inputs** produce identical outputs
for these boundary cases. Independently trained encoders can differ even
on one-transition examples; that is not a pooling-parity failure.

## Optional research integration

With `B16v1_Progression/src` on the Python path:

```python
from b16v1_progression.progression_condition import ProgressionCondition
from b16v1_progression.direction_aware_pooling import make_direction_aware_progression_class

ExperimentalDACP = make_direction_aware_progression_class(ProgressionCondition)
model = ExperimentalDACP()
# before/after/temporal: [batch, 16]
# deltas: [batch, historical_pairs, 16]
# delta_mask: [batch, historical_pairs], valid=1, padding=0
# condition16, progression16, fused64 = model(before, after, temporal, deltas, delta_mask)
```

The subclass inherits the original Delta Encoder, fusion, and adapter;
only encoded-transition aggregation changes. It adds no parameters and
retains state_dict keys/shapes. `load_direction_aware_progression_state` is an
explicit tensor-only **experimental** loader, not an automatic model upgrade.
Strict loading alone does not validate a mean-trained state under DACP.

Use only lawful historical pre-target representations and fixed S2 supervision.
Do not use final-target representations or labels to derive weights. The
public config captures the exact tested formula and optimization contract.
Private launchers, splits, diagnostics, OOF representations, bootstrap results,
and training checkpoints are deliberately not published.
