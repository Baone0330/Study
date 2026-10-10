# Historical six-branch paper package

This is an additive presentation archive of the [existing checkpoint replay](../../README.md), not a new clinical experiment. Identity: HISTORICAL_FIXED_CLIMATV2_INTERNAL_VAL_SIX_BRANCH_REPLAY. Existing replay base commit: d97379a9e46cd462c4132f5173a5f319d4f570dd.

Six complete three-seed positive-probability ensembles on historical internal Val n=189 (HT0=131, HT1=58), unchanged native thresholds, AP=average_precision_score. Original Progression has the highest observed ROC-AUC/AP here; Plain-SD above Full is retained. No independent Test, CI/p-value, isolated causal component benefit or clinical image-realism claim is added.

The historical Full route is longitudinal-only, not an earlier report/pathology Full-B16. Original P16 uses upstream seed20260921 and the original delta/fusion architecture, not S2/Clean. The P16 adapter is not B14 CPFM. Plain-SD changes multiple condition routes, so its historical name does not certify a one-factor CPFM ablation.

Files: five safe tables; eight nonmedical figure groups (editable SVG, RGB600dpi PNG, PDF); per-figure CSV and bilingual captions; original-source layer inventory; Chinese Methods/Results/Discussion drafts. CPU source/schema inspection did not run a new model forward, training or diffusion generation. Patient records, clinical images, original model weights, private locators and report text are excluded.

## Rebuild figures without private clinical data

Use a Python environment with matplotlib, numpy and Pillow:

```bash
python figure_source/draw_paper_figures.py --help
python figure_source/draw_paper_figures.py --root .
```

The provided aggregate CSV and verified architecture facts are sufficient; this command does not require checkpoints or read clinical data. Re-exporting figures does not establish a new experimental result. Actual agent visual review and automated checks are recorded in qa; human author review remains NOT_RUN. Journal formatting and asset permissions require author verification before submission.

原权重与患者资料不公开。本目录仅新增，不替换现有重放代码、模型或已发布文件。完整结果定位与边界见HISTORICAL_PROTOCOL_AND_EVALUATION.md和LIMITATIONS.md。
