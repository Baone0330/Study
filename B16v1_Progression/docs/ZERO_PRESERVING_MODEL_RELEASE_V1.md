# Zero-Preserving Progression model release v1

This directory contains model-only code and release-safe tensor weights for
the ZP-FIT (228-patient) Train-only control. It does not include patient
manifests, predictions, results, generated images, training launchers or logs.

`R_seed_*.pt` is the ZP-FIT-refit R condition backbone;
`P_source_seed_*.pt` is the unchanged DeltaEncoder progression source refit on
the same new split. For each ZC, ZP and ZCP arm and each seed 42, 3407 and
2026, one adapter weight file serves both P_TRUE and P_ZERO. The release-safe
weights contain only an architecture version and tensor state dictionaries.

The new residual is `Linear(16,16,bias=False) -> SiLU ->
Linear(16,16,bias=False)` with a sigmoid gate initialized at 0.10. Thus
`r(0)=0` structurally. ZC adds it after the frozen R condition. ZP adds it to
the frozen B14CPFM patient latent before the unchanged gamma/beta heads. ZCP
uses both. No post-residual tanh, LayerNorm or affine is added. The original
zero-condition reference remains progression-free.

Load with `b16v1_progression.zero_preserving_release_loader_v1.load_zero_preserving_bundle`.
Pass the existing release-safe B14CPFM base weight separately. The caller is
responsible for the frozen pre-query REAL history encoder, SD1.5/medical LoRA,
PNDM epsilon scheduler and unchanged U-Net hooks. This release does not claim
clinical validity or independent Test performance.
