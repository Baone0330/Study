# Historical Original Progression–CPFM saved-image quality evaluation

This release contains reproducible offline analysis code, aggregate source tables, eight nonmedical figures with editable text, and bilingual scientific interpretation. It does not contain medical images, patient identities/dates, pair maps, embeddings, reports, checkpoints, or clinical labels.

Primary controlled comparison: saved historical Original Progression CPFM ON vs CPFM OFF, same patient/node/image index/seed, 9270 image pairs across1545 nodes and860 patients. Reference:13200 native historical views from3154 exams and859 patients, not target-year ground truth. Full and Plain SD are exact image/seed-aligned historical auxiliary routes, not isolated CPFM interventions. All source images, including duplicate pixels and flagged artifacts, are retained.

## Execution interfaces

Use a private server directory named `b16v1_historical_original_progression_generation_quality_v1`. Supply actual authorized frozen source paths on the command line; do not copy private source manifests into this repository. The historical source-inspection entry point accepts `--output-root`, `--source-root`, and `--full-manifest`; the extension entry point accepts the legitimate `--plain-manifest`. `prepare_quality_models.py` prepares task-local public evaluation weights. Every executable entry point supports `--help`.

After source freezes, run `check_decode_and_image_stats.py`, then the single locked `run_measurement_pipeline.py` (feature extraction, paired changes/diversity, FID/KID, patient aggregation), then `draw_public_quality_figures.py`, `draw_private_medical_panels.py`, `prepare_blinded_expert_review_template.py`, and `build_quality_report.py`. All accept `--output-root`. Source images are read-only. A failed pipeline retains its own identity-checked chunks for the same-protocol resume; never start a second process while the lock is held.

## Environment and tests

Measured environment: Python3.10.20, PyTorch2.11.0+cu128 on RTX5090, torch-fidelity0.4.0, LPIPS0.1.4, scikit-image0.25.2, NumPy2.2.6, pandas2.3.3, SciPy1.15.3, Matplotlib3.10.9. Additional LPIPS/scikit-image support was installed only under the private task's vendor directory, without modifying shared CUDA/driver/PyTorch. Do not install the listed Torch build blindly on another host; first verify its driver/device compatibility. Read the runtime and encoder contract implementation.

Anonymous contract tests: `python scripts/test_quality_contracts.py`. Only numerical toy arrays are used. Feature weights and all real intermediate features stay private. Cached LPIPS implementation was compared against the official calibrated forward on anonymous data. Medical diagrams are restricted server outputs, never public exports. REAL display pixels remain withheld pending verified de-identification.

## Interpretation

FID/KID describe natural-image feature distribution distance; SSIM/LPIPS quantify ON/OFF change, not fidelity to a future real target. Whole-image contrast/edge/frequency and overlay flags are technical proxies with unknown clinical specificity. Patient bootstrap uncertainty is not independent-image uncertainty; KID MC SD is not a patient CI. Balanced FID is finite but rank-limited; its CI was prespecified NOT_RUN. No expert ratings, thyroid ROI certification, new image generation, new generator/downstream training, or official Test access occurred. See the complete aggregate tables and bilingual limitations for negative/mixed findings.
