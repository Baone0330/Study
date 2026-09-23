# Architecture

The historical encoder derives eight grayscale image statistics and six report count features per image, producing a 14 dimensional input. Formal Progression inference passes no report text, so the report dimensions are zero. Independent Before and After `ExamSetEncoder` modules apply `Linear(14,32)`, `LayerNorm(32)`, `GELU`, `Linear(32,16)`, then mean pool within each exam set.

The relative annual query encoder maps nine numeric time features through `Linear(9,24)`, `LayerNorm(24)`, `GELU`, `Linear(24,16)`. A frozen longitudinal bridge consumes Before16, After16, and Temporal16 and supplies the scale control. The Progression module encodes consecutive real historical exam differences and fuses Before16 + After16 + Temporal16 + Progression16 into a 16 dimensional CPFM condition.

`B14CPFM` computes condition dependent gamma and beta directions, subtracts their zero condition response, normalizes by RMS, applies magnitude and safety heads, then modulates the first ResNet inputs in U-Net down blocks D0 to D3. `B14Hooks` registers those four forward pre hooks. The formal wrapper loads SD1.5 and medical LoRA, uses PNDM with epsilon prediction, 30 denoising steps, CFG 7.0, and 512 × 512 VAE decoding.

The formal active path is historical images + query time + Progression + B14CPFM/B14Hooks + SD1.5/LoRA. Report condition, structure, pathology, and texture are disabled. The old Full B16v1 paths are not active modules of this release.
