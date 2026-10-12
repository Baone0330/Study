# Historical Original Progression CPFM injection ablation

This is the new route-H CPFM-OFF comparison, not the old Plain-SD branch. The six historical reference rows are preserved. Read `results/public_replay_status.json` for measured outcomes and `docs/INTERPRETATION_BOUNDARIES.md` before using the numbers. The model components and six-image sampling settings are original; only the B14 enabled gate is bypassed.

## Private dependencies and execution

Clinical history/slot manifests, original weights, SD1.5 and medical LoRA, original native CLIMATv2 sources and the original local ImageNet state must be supplied through a private data root. They are not downloadable from this public directory. No medical images, private mappings, probabilities or weights are distributed. Public requirements describe the observed current environment, not a recovered exact historical dependency lock.

Use an isolated private root whose basename is `b16v1_original_progression_cpfm_only_ablation_historical_v1`. Run the real CLIs with `--help` first. The preflight reconstructs the exact old support from the private project tree; missing/changed sources are hard engineering errors, never replaced by guessed seeds.

```bash
python scripts/preflight.py --root "$PRIVATE_OUTPUT" --data-root "$PRIVATE_DATA_ROOT"
python scripts/pilot.py --root "$PRIVATE_OUTPUT"
python scripts/pilot.py --root "$PRIVATE_OUTPUT" --repeat-only
python scripts/freeze_route.py --root "$PRIVATE_OUTPUT"
python scripts/sample_cpfm_off.py --root "$PRIVATE_OUTPUT"
python scripts/train_climatv2.py --root "$PRIVATE_OUTPUT" --stage build
python scripts/train_climatv2.py --root "$PRIVATE_OUTPUT" --stage train --seed 42
python scripts/train_climatv2.py --root "$PRIVATE_OUTPUT" --stage reload --seed 42
# Repeat train/reload for seeds 3407 and 2026, serially, without changing the run.
python scripts/aggregate.py --root "$PRIVATE_OUTPUT"
python scripts/draw_figures.py --root "$PRIVATE_OUTPUT"
python -m unittest discover -s tests
python scripts/validate_release.py
```

`generation/ledger.sqlite` commits original seed/config-bound per-image states. Partial-node recovery retains the original full six-image batch. Completed outputs are verified, not overwritten. Native training equations are source-derived with additive epoch-boundary model/optimizer/scheduler/RNG/loader-state recovery; an incomplete epoch is replayed from its last completed boundary. The original source files and historical outputs are not edited.

## Review and statistical scope

The primary unit is the189 internal-Val patients, not567 seed-patient replicates or9270 images. AP denotes average precision, and each arm uses its own historical Youden threshold. Paired bootstrap uses2000 stratified patient draws, seed20261012. Val informed checkpoint and threshold selection. Medical expert review and Official Test access were NOT_RUN. Automatic figure QA and actual visual inspection of all five nonmedical figures passed; the separate records are included under docs. Numerical image changes are not proof of disease progression or anatomical realism. Refer to the complete bilingual method and captions for scope and denominators.
