# B16v1 + Disease Progression Condition

Patient specific longitudinal ultrasound generation with historical exam encoding, relative time encoding, a disease progression condition, B14CPFM modulation, and a frozen Stable Diffusion 1.5 backbone.

## Release scope

This package contains the audited active model definitions, a portable generation wrapper, the Progression training entry, and three model only custom checkpoints. The formal Progression generation configuration has **report condition, structure, pathology, and texture disabled**. The six dimensional report feature implementation remains in the historical encoder for compatibility; formal Progression inference passes empty report inputs. Proxy and attribute heads exist in the longitudinal model but do not condition the formal generation path.

The SD1.5 base weights and medical LoRA are external. The medical LoRA used by the formal runtime was located and checked for tensor metadata, but public redistribution rights were not established. Supply an authorized compatible LoRA yourself. The clinical ultrasound dataset is not included.

## Install

Use a CUDA capable environment compatible with the versions in [ENVIRONMENT.md](docs/ENVIRONMENT.md), then:

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
```

## Synthetic smoke test

```bash
python scripts/smoke_test.py \
  --longitudinal_ckpt weights/longitudinal_condition/release_safe_longitudinal_condition.pt \
  --cpfm_ckpt weights/b14_cpfm/release_safe_b14_cpfm.pt \
  --progression_ckpt weights/progression_condition/release_safe_progression_condition.pt
```

This test creates and removes a synthetic image in a temporary directory. It strictly loads all three checkpoints and runs the condition and CPFM forward paths.

## Inference

`scripts/generate_ultrasound.py` accepts a caller supplied, authorized `history.json` with a `history` array. Each entry contains `exam_date` and `image_paths`. The script filters exams to dates strictly before `--query_date`; keep patient data and output outside this repository.

```bash
python scripts/generate_ultrasound.py \
  --base_model /path/to/stable-diffusion-v1-5 \
  --medical_lora /path/to/authorized/medical_lora \
  --longitudinal_ckpt weights/longitudinal_condition/release_safe_longitudinal_condition.pt \
  --cpfm_ckpt weights/b14_cpfm/release_safe_b14_cpfm.pt \
  --progression_ckpt weights/progression_condition/release_safe_progression_condition.pt \
  --history_json /path/to/authorized/history.json \
  --query_date 2026-01-01 --annual_index 2 \
  --output_png /path/to/output.png
```

The external base and LoRA must match the formal runtime for identical results. No clinical output example is supplied.

## Training

`scripts/train_progression.py` retains the formal self supervision objective and patient level inner train/dev split. It requires an authorized Train only history JSONL and the released longitudinal checkpoint. Write the output outside this repository because training records include patient level manifests.

```bash
python scripts/train_progression.py \
  --train_history_jsonl /path/to/authorized/train_history.jsonl \
  --longitudinal_ckpt weights/longitudinal_condition/release_safe_longitudinal_condition.pt \
  --output_dir /path/outside/repository/progression_training
```

See [ARCHITECTURE.md](docs/ARCHITECTURE.md), [EQUATIONS_AND_LOSSES.md](docs/EQUATIONS_AND_LOSSES.md), and [WEIGHTS_MANIFEST.md](weights/WEIGHTS_MANIFEST.md).

## Reproducibility boundary

Model code and three custom checkpoint components are available. Inference additionally needs external SD1.5 and an authorized medical LoRA. Full original training data reproduction requires authorized clinical data. No claim of clinical validation or independent external validation is made.
