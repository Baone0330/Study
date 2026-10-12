"""Prescore auxiliary and technical definitions; retains the original freeze unchanged."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from b16_quality_common import setup, read_json, atomic_json, atomic_parquet, now, postflight, stat_record, write_csv


def main(output_root, plain_manifest):
    root = setup(output_root)
    main_cfg = read_json(root / 'protocol/FROZEN_QUALITY_EVALUATION_CONTRACT_V1.json')
    extension = root / 'protocol/PREMEASUREMENT_EXTENSION_V1.json'
    if extension.exists():
        assert read_json(extension)['plain_manifest'] == str(plain_manifest)
        postflight(root)
        print('Same frozen extension retained')
        return
    assert not list((root / 'metrics').glob('*')), 'Do not redefine metrics after scores'
    p = Path(plain_manifest)
    source = p.parents[2]
    config_path = source / 'plain_sd_generation/configs/PLAIN_SD_CAP2_GENERATION_CONFIG_V1.json'
    release_path = source / 'plain_sd_generation/release/PLAIN_SD_CAP2_GENERATION_RELEASE_SUMMARY_V1.json'
    config, release = read_json(config_path), read_json(release_path)
    assert release['contract_status'] == 'PASS' and release['synthetic_images'] == 9270
    assert config['train_only_generation'] == 1 and config['test_access'] == 0
    f = pd.read_csv(p, dtype={'patient_id': str})
    assert len(f) == 9270 and set(f.split) == {'train'} and set(f.node_status) == {'COMPLETE'}
    assert not f.duplicated(['patient_id', 'query_date', 'image_index']).any()
    for col in ['target_year_read', 'target_report_read', 'target_label_used', 'patient_history_used', 'b16_specific_condition_used']:
        assert (f[col] == 0).all()
    q = pd.read_parquet(root / 'manifests/QUALITY_EVAL_FROZEN_QUEUE_V1.parquet')
    on = q[q.branch == 'OP_CPFM_ON']
    merged = on.merge(f, left_on=['patient_key_private', 'source_date_private', 'image_index'],
                      right_on=['patient_id', 'query_date', 'image_index'], validate='1:1', how='left')
    assert len(merged) == 9270 and merged.generated_image_path.notna().all()
    assert (merged.seed_x == merged.seed_y).all()
    paths = [p, config_path, release_path]
    rows = []
    for r in merged.itertuples():
        assert Path(r.generated_image_path).is_file()
        paths.append(r.generated_image_path)
        d = {col: getattr(r, col) for col in on.columns if col != 'seed'}
        d.update(image_id=r.image_id.replace('OP_CPFM_ON:', 'PLAIN_SD_HIST:'), branch='PLAIN_SD_HIST',
                 seed=int(r.seed_y), png_abs_path=r.generated_image_path,
                 clinical_role='generic_prompt_auxiliary_not_new_CPFM_OFF',
                 comparability='EXACT_PATIENT_NODE_INDEX_SEED_AUXILIARY_NOT_SINGLE_VARIABLE')
        rows.append(d)
    plain = pd.DataFrame(rows)
    combined = pd.concat([q, plain], ignore_index=True)
    assert combined.image_id.is_unique
    atomic_parquet(root / 'manifests/PLAIN_SD_COMPARABILITY_V1.parquet', plain)
    atomic_parquet(root / 'manifests/COMBINED_QUALITY_EVAL_FROZEN_QUEUE_V1.parquet', combined)
    atomic_parquet(root / 'manifests/FULL_B16V1_COMPARABILITY_V1.parquet', q[q.branch == 'FULL_B16V1_HIST'])
    # Independent append-only protection supplement: never reset the original baseline.
    atomic_json(root / 'source_audit/AUXILIARY_PROTECTED_SOURCE_METADATA_V1.json',
                [stat_record(x) for x in sorted(set(map(str, paths)))])
    counts = []
    for b, g in combined.groupby('branch', sort=False):
        counts.append(dict(branch=b, images=len(g), patients=g.anonymous_patient_key.nunique(),
                           nodes=g[g.anonymous_node_key != ''].anonymous_node_key.nunique(),
                           native_exams=g[g.real_exam_key_private != ''].real_exam_key_private.nunique(),
                           balanced_images=int(g.patient_balanced_selected.sum()),
                           role='PRIMARY_PAIRED' if b.startswith('OP_') else 'REFERENCE' if b.startswith('REAL_') else 'AUXILIARY_ROUTE_ONLY'))
    write_csv(root / 'manifests/COHORT_COUNTS_AND_PATIENT_OVERLAP_V1.csv', counts)
    ext = dict(status='FROZEN_BEFORE_QUALITY_MEASUREMENT', frozen_utc=now(), plain_manifest=str(p),
               plain_status='EXACT_IMAGE_SEED_ALIGNED_AUXILIARY_GENERIC_PROMPT',
               plain_images=len(plain), full_status=main_cfg['full_status'], original_contract_retained=True,
               texture_flags={'low_contrast_gray_std_lt': 10, 'very_smooth_block_std_median_lt': 3,
                              'extreme_black_fraction_gt': 0.5, 'extreme_white_fraction_gt': 0.1,
                              'high_frequency_fraction_gt': 0.5,
                              'canny_thresholds': [50,150], 'hough_votes': 50,
                              'hough_min_length_px':128, 'hough_max_gap_px':10,
                              'line_flag': 'at least one Hough line length>=128',
                              'cross_flag': 'one near-horizontal and one near-vertical segment intersect within both extents; angle tolerance10deg'},
               duplicate_policy='All images: fixed raw RGB 9x9 coordinate samples group candidates; full native RGB ndarray equality verifies duplicates; no file digest; no removal',
               real_patient_aggregation='native views mean within exam; exam mean within patient; equal patients',
               generated_patient_aggregation='six images mean within node; node mean within patient; equal patients',
               balanced_images_per_patient=main_cfg['balanced_policy']['m'],
               figure_contract={'groups':8,'formats':['PNG_RGB_600dpi','SVG_EDITABLE_TEXT','PDF_EMBEDDED_FONTS'],
                                'backend':'Python Matplotlib','minimum_font_pt':8,'medical_agent_review':'NOT_RUN'},
               roi_model_status='NOT_AVAILABLE_NO_VALIDATED_THYROID_ROI_MODEL_IN_SUPPLIED_FROZEN_ASSETS',
               decode_workers=6, feature_batch_size=16, lpips_node_batch_size=6,
               compute_dtype='float32 FP32 evaluation; TF32 off; no mixed precision',
               no_score_filter=True)
    atomic_json(extension, ext)
    (root / 'protocol/QUALITY_METRIC_DEFINITIONS_V1.md').write_text('''# Frozen metric definitions

All image fingerprints contain sampled pixel values, not file hashes. Duplicate candidates are proved by full native RGB array equality and retained. Decoding failures are recorded and stop the pipeline rather than silently changing denominators.

Texture uses whole RGB image resized to 512x512 with PIL bilinear (native shape/mode and native gray mean/std retained). BT.601 gray =0.299R+0.587G+0.114B. The 32x32 non-overlapping blocks tile the full image without padding. Sobel is OpenCV ksize3 gradient mean; Laplacian is OpenCV ksize3 variance. FFT is demeaned gray times separable Hann window, DC excluded, radial high band >=0.25 cycles/pixel, maximum radius sqrt(0.5^2+0.5^2). These are image-statistic proxies, not pathology scores. Black<5 and white>250 use gray. Border black is fraction in the outer 5% strip; border UI proxy is that strip's Canny edge fraction. Fixed flags are saved in PREMEASUREMENT_EXTENSION_V1.json; they have unknown clinical false-positive/false-negative rates.

ON/OFF MAE and RMSE use all native RGB uint8 channels; max difference is 0..255; bit-exact tests use ndarray equality. SSIM uses BT.601 gray float64, data_range255, Gaussian sigma1.5, window11, population covariance. LPIPS uses offline official pretrained AlexNet plus v0.1 calibration, RGB uint8/127.5-1. Higher LPIPS means more pair change, not lower medical realism. Full vs ON is auxiliary route change, not single-module causality.

Inception is torch-fidelity0.4.0 inception-v3-compat 2048, actual official evaluation weights, native RGB grouped by dimensions, internal TF1-compatible299 bilinear without antialias, (uint8-128)/128. FID uses float64 sample covariance and PSD Bures trace; low-rank patient-balanced covariance is explicitly limited. KID is unbiased cubic polynomial MMD squared, gamma1/2048, coef1; 50 fixed-seed subsets512 report MC mean and SD, not clinical confidence intervals. Negative unbiased estimates are retained. For paired distance uncertainty, balanced full U-statistic KID OFF-minus-ON uses 2000 paired patient-cluster bootstrap replicates. FID CI is NOT_RUN as declared before measurement.

Image-to-node-to-patient means and 2000 paired patient bootstrap estimates are reported separately from image distributions. REAL uses native exam-to-patient means, not LOCF slots. No images are discarded by a quality threshold, no target-year reference is claimed, and no expert ratings are generated.
''', encoding='utf-8')
    write_csv(root / 'writing/TERMINOLOGY_LEDGER_V1.csv', [
        {'term_en':'real historical reference','term_zh':'真实历史参考','boundary':'not query-year ground truth'},
        {'term_en':'paired structural change','term_zh':'配对结构变化','boundary':'not reconstruction accuracy'},
        {'term_en':'artifact proxy','term_zh':'可疑伪影统计代理','boundary':'not medical diagnosis'},
        {'term_en':'natural-image feature distance','term_zh':'自然图像特征距离','boundary':'not thyroid authenticity certification'},
        {'term_en':'auxiliary historical route','term_zh':'历史辅助路线','boundary':'not isolated CPFM effect'}])
    postflight(root)
    print({'frozen_images': len(combined), 'plain_seed_matched': len(plain), 'balanced_m': ext['balanced_images_per_patient']}, flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-root', required=True); p.add_argument('--plain-manifest', required=True)
    a=p.parse_args(); main(a.output_root,a.plain_manifest)
