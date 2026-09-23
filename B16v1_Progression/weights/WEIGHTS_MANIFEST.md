# Weight manifest

| Weight | Role | Required | Released | Format | Size (bytes) |
|---|---|---:|---:|---|---:|
| Stable Diffusion 1.5 | Frozen base | Yes | No, external | Provider model | — |
| Medical LoRA | Frozen ultrasound adaptation | Yes | No, redistribution rights unverified | safetensors | 6,414,992 at source |
| Longitudinal condition | Before/after/time encoding | Yes | Yes | model only `.pt` | 32,441 |
| Progression condition | Delta and fusion | Yes | Yes | model only `.pt` | 81,877 |
| B14CPFM | U-Net down block modulation | Yes | Yes | model only `.pt` | 6,858,038 |

Stable Diffusion 1.5 base weights are not included. The medical LoRA is a `REQUIRED_EXTERNAL_WEIGHT` until ownership and redistribution terms are established. Source checkpoint metadata and training state were removed from the released `.pt` files. Each released file contains `model_state_dict`, a non private `architecture` label, and `version`.
