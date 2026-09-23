# Equations and losses

For each historical image, `x ∈ R^14` is the concatenation of eight grayscale statistics and six report count features. Formal Progression generation uses `report = 0`. The exam set representation is `mean_i Linear16(GELU(LayerNorm32(Linear32(x_i))))` for independent Before and After encoders.

The temporal vector is `T = Linear16(GELU(LayerNorm24(Linear24(q9))))`. Consecutive real history representations give `ΔF_i = F_i − F_(i−1)`. The Progression encoder applies `Linear(16,32) → LayerNorm(32) → GELU → Linear(32,16)` to each difference and computes a masked mean. `[Before16, After16, Temporal16, Progression16]` is concatenated to 64 dimensions, then passes through `Linear(64,128) → LayerNorm(128) → GELU → Linear(128,64) → Linear(64,16) → tanh`.

The formal training objective is `L_prog = L_teacher-MSE + 0.20 L_teacher-cos + 0.50 L_delta-MSE + 0.10 L_delta-cos + 0.10 L_compat-MSE`. The training script implements these coefficients unchanged.

For each U-Net down block, CPFM computes condition and zero condition gamma/beta directions, uses their difference, normalizes by RMS, applies learned magnitude and per layer gate, and modifies normalized activations with gamma and beta. The frozen diffusion model predicts epsilon; no SD1.5 training objective or base weights are included in this release.
