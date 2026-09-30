# Fixed H1 and experimental H2 endpoint construction

H1 remains the default. H2 is experimental / not selected; no H2 replacement
weights are supplied. Existing architecture, default configuration and loaders
are unchanged. These utilities are training-target integration, not a launcher.

For a strictly screened historical prefix `F_0..F_k`, define a single fixed
supervision endpoint `E=F_(k+h)`, with `h=1` or experimental `h=2`. Comparisons
must use exactly the anchors with `k+2<=L`; H1-only anchors cannot be added to
the H1 comparison arm.

`construct_endpoint_bundle` in
`b16v1_progression.fixed_horizon_target_bundle_v1` rebuilds the complete bundle:

- Delta target: `E-F_k`.
- Teacher: frozen `Bridge(before, E, query_h)`.
- Compatibility: frozen `Bridge(before, F_k, query_h)`.
- Student: identical `before`, `F_k`, prefix raw adjacent deltas and valid mask.
- Query: endpoint date and zero-based endpoint index `k+h`, preserving the
  recovered query encoder's `annual_index` semantics. First/before/after dates
  remain prefix dates. This expected temporal difference is part of the horizon
  intervention, not an undisclosed non-temporal input change.

The caller must supply the screened, frozen prefix-before representation. In
the recovered training semantics it is the image-count-weighted mean of the
historical before representations preceding the current anchor, or the frozen
empty-before representation at `k=0`. It must never contain an intermediate
future exam. `F_(k+1)` is NOT part of the H2 student prefix. The endpoint feature
is supervision-only. No pixel, report, path, label, or patient-file loading is
performed by this module.

Both teacher and delta use the same endpoint. Compatibility is not a future
target: its original last-visible-feature semantics are retained. Neither
targets nor query metadata are normalized by elapsed time. No new network
layer, horizon embedding, pooling, attention, or loss is introduced.

S2 weights remain `.25 teacher MSE + .05 teacher cosine + 1.20 delta MSE +
.30 delta cosine + .10 compatibility MSE`. The progression network uses the
original DeltaEncoder and masked mean. At a zero-delta prefix the original P
is identically zero, numerical cosine is zero, and directional interpretation
is undefined. Report informative nonempty-prefix diagnostics separately.

`balanced_h12_values` is a no-gradient selection helper. Compute it only on
TUNE, average anchors within each patient, then average patients; both arms
use the same criterion and exact ties retain the earlier epoch. It is NOT a
joint H1/H2 training loss. Cross-fit evaluation, outcome labels and H3 must not
select checkpoints. Do not use these utilities to make a clinical validation
claim. Any generation-path validation requires separate authorization.

The fixed-horizon experiment did not justify replacing H1. Publicly archived
here are only reusable tensor-level target construction, its configuration,
and these definitions; no split, registry, representations, diagnostic results,
training logs, launchers or clinical data are included.
