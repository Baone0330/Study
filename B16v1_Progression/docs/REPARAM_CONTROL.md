# B16v1-Reparam-Control

This is the model-only release of the `REPARAM_ONLY` mechanism control. It is
not a replacement for the formal B16v1 Progression model and does not establish
clinical or external validation.

`ReparamOnly` receives three 16-dimensional frozen representations (Before,
After, Temporal). It appends a zero 16-vector in place of progression and maps
the concatenated 64-vector through Linear(64,128), LayerNorm, GELU,
Linear(128,64), Linear(64,16), and tanh. The released weights are state dicts
only. The model has no patient identifiers, clinical data, image files,
optimizer, scheduler, or server paths.

The control was trained with teacher-condition MSE, 0.20×teacher cosine loss,
and 0.10×compatibility MSE. It did not use progression-delta loss. Seeds 42,
3407, and 2026 are separate retained model weights; select a seed explicitly.

Example:

```python
from b16v1_progression.reparam_control import load_reparam_control

model = load_reparam_control('weights/reparam_control/reparam_only_seed_42.pt')
condition16 = model(before16, after16, temporal16)
```

The external encoder and CPFM implementations must be kept consistent with
the frozen B16v1 stack. No Stable Diffusion, medical LoRA, or ultrasound data
is distributed with this control.
