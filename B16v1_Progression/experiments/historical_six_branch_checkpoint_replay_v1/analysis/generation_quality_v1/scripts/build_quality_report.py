"""Measured Chinese handoff, bilingual manuscript materials, and privacy-safe public package."""
from __future__ import annotations
import argparse
import os
import shutil
import re
from pathlib import Path
import pandas as pd
from b16_quality_common import setup, read_json, atomic_json, now, postflight, write_csv, state

RESULT='B16V1_HISTORICAL_ORIGINAL_PROGRESSION_GENERATION_QUALITY_RESULT_V1.md'
DISCLAIMER='本轮生成图像的自动质量统计与视觉对照用于评价统计分布、伪影和结构可视性，不等价于获得医生认证的医学真实性，也不构成新的独立下游 HT 预测验证。'


def table(frame,columns=None):
    f=frame if columns is None else frame[columns]
    def cell(v):
        if pd.isna(v):return 'NA'
        if isinstance(v,float):return format(v,'.7g')
        return str(v).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(cell(v) for v in row)+' |' for row in f.itertuples(index=False,name=None))+'\n'


def bilingual(*pairs):return '\n\n'.join(en+'\n\n'+cn for en,cn in pairs)+'\n'


def public_package(root,private_report,status):
    from review_github_publication_scope import SCRIPTS,WRITING,PROTOCOL,scan
    package=root/'public_package';package.mkdir(exist_ok=True)
    def copy(path):
        d=package/path.relative_to(root);d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,d)
    for name in SCRIPTS:
        p=root/'scripts'/name
        if p.exists():copy(p)
    for folder,names in [('writing',WRITING),('protocol',PROTOCOL)]:
        for name in names:
            p=root/folder/name
            if p.exists():copy(p)
    for p in sorted((root/'tables').glob('Table*.csv')):copy(p)
    for folder in ['nonmedical','source_data','captions']:
        for p in sorted((root/'figures'/folder).glob('Fig0*')):copy(p)
    copy(root/'figures/export_qa/NONMEDICAL_FIGURE_EXPORT_QA_V1.csv')
    for name in ['FID_KID_NUMERICAL_TOY_V1.json','ANONYMOUS_TEST_RESULTS_V1.json']:copy(root/'tests'/name)
    text=private_report.replace(str(root),'<private output root>')
    cfg=read_json(root/'protocol/FROZEN_QUALITY_EVALUATION_CONTRACT_V1.json')
    text=text.replace(cfg['source_root'],'<frozen original ON/OFF source root>')
    text=re.sub(r'；Git 独立状态：`[^`]+`。','；公开包的 Git 标识以仓库提交历史为准。',text)
    first,sep,last=text.partition('## 16. GitHub 发布')
    if sep:
        _,sep2,tail=last.partition('## 17. 资源、恢复与原资产保护')
        text=first+'''## 16. GitHub 发布

公开子目录为 `B16v1_Progression/experiments/historical_six_branch_checkpoint_replay_v1/analysis/generation_quality_v1/`。包已通过严格白名单及实际私有值扫描，只有代码、匿名测试、聚合表/非医学图与双语材料。医学影像、患者身份/日期、逐患者特征/概率、权重及临床报告不在包中。为避免将自己的 commit 标识嵌入其本身造成循环自引用，公开包的真实版本以仓库提交历史为准；最终 push 与远端核验在发布后记录于单独执行结果报告，不能以包组装时的状态代替。

'''+sep2+tail
    # This public report never contains raw per-image paths, identifiers, clinical dates or features.
    (package/'PUBLIC_QUALITY_RESULT_V1.md').write_text(text,encoding='utf-8')
    safe={k:v for k,v in status.items() if k not in {'private_output_root','private_medical_panel_manifest','input_source_root',
                                                     'git_publication_status','git_commit','git_remote_verified'}}
    safe['publication_provenance']='Repository history is source of truth; actual push verification is recorded after publishing, outside this self-referencing commit.'
    atomic_json(package/'release/PUBLIC_FINAL_STATUS_V1.json',safe)
    copy(root/'release/NEXT_STAGE_PAPER_HANDOFF_V1.md')
    (package/'README.md').write_text('''# Historical Original Progression–CPFM saved-image quality evaluation

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
''',encoding='utf-8')
    (package/'requirements-evaluation.txt').write_text('''# Existing compatible Torch/torchvision must be verified, not blindly replaced.
numpy==2.2.6
pandas==2.3.3
scipy==1.15.3
opencv-python==4.13.0.92
Pillow==12.2.0
matplotlib==3.10.9
torch-fidelity==0.4.0
lpips==0.1.4
scikit-image==0.25.2
pyarrow==24.0.0
''',encoding='utf-8')
    audit=scan(package,root);atomic_json(root/'release/PUBLICATION_SCOPE_AUDIT_V1.json',audit)
    if audit['status']!='PASS':raise RuntimeError('Public privacy audit failed; package stays private and cannot be pushed')
    return audit


def main(output_root,build_package=False):
    root=setup(output_root)
    assert read_json(root/'release/MEASUREMENT_COMPLETED_V1.json')['status']=='MEASUREMENT_COMPLETED'
    c=read_json(root/'protocol/FROZEN_QUALITY_EVALUATION_CONTRACT_V1.json');ext=read_json(root/'protocol/PREMEASUREMENT_EXTENSION_V1.json')
    protection=postflight(root)
    counts=pd.read_csv(root/'tables/Table01_Branch_Support_V1.csv');dist=pd.read_csv(root/'tables/Table03_Distribution_Distances_V1.csv')
    change=pd.read_csv(root/'tables/Table04_Paired_Changes_V1.csv');tech=pd.read_csv(root/'tables/Table05_Technical_Proxies_V1.csv')
    div=pd.read_csv(root/'tables/Table06_Node_Seed_Diversity_V1.csv');flags=pd.read_csv(root/'tables/Table07_Artifact_Proxy_Review_Scope_V1.csv')
    ci=pd.read_csv(root/'tables/Table09_Paired_KID_Patient_CI_V1.csv');na=pd.read_csv(root/'tables/Table10_Unavailable_Metrics_V1.csv')
    duplicates=pd.read_csv(root/'tables/Table12_Exact_Pixel_Duplicates_V1.csv');qa=pd.read_csv(root/'figures/export_qa/NONMEDICAL_FIGURE_EXPORT_QA_V1.csv')
    panels=read_json(root/'figures/export_qa/PRIVATE_MEDICAL_PANEL_TECHNICAL_QA_V1.json');expert=read_json(root/'figures/private_medical_panels/EXPERT_TEMPLATE_STATUS_V1.json')
    runtime=read_json(root/'image_features/FEATURE_RUNTIME_CONTRACT_V1.json');tests=read_json(root/'tests/ANONYMOUS_TEST_RESULTS_V1.json')
    assert tests['status']=='PASS' and len(qa)==8
    pubfile=root/'release/GIT_PUBLICATION_STATUS_V1.json'
    git=read_json(pubfile) if pubfile.exists() else dict(status='PUSH_NOT_ATTEMPTED',commit=None,remote_verified=False)
    final='QUALITY_EVALUATION_COMPLETED_WITH_LIMITATIONS' if qa.agent_visual_review_status.eq('AGENT_VISUAL_PASS').all() else 'QUALITY_EVALUATION_MEASUREMENTS_COMPLETE_VISUAL_QA_PENDING'
    status=dict(FINAL_STATUS=final,task=c['task'],updated_utc=now(),ON_found_images=9270,OFF_found_images=9270,
                exact_matched_pairs=9270,matched_nodes=1545,matched_unique_patients=860,real_reference_unique_views=13200,
                real_reference_unique_exams=3154,real_reference_patients=859,distribution_common_patients=859,
                full_historical_status=c['full_status'],plain_sd_status=ext['plain_status'],fid_status='MEASURED_10_COMPARISONS_WITH_RANK_LIMITATION',
                kid_status='MEASURED_MC_AND_PAIRED_2000_PATIENT_BOOTSTRAP',paired_image_stats_status='COMPLETE_9270',
                clinical_expert_review_status='NOT_RUN',medical_agent_visual_review_status='NOT_RUN',thyroid_ROI_status='NOT_AVAILABLE',
                medical_REAL_display_status='WITHHELD_PENDING_VERIFIED_DEIDENTIFICATION',private_medical_panels_uploaded_or_public=False,
                new_generator_training_steps=0,new_image_generation=0,new_climatv2_training=0,official_test_access=0,
                original_source_metadata_postflight=protection['status'],protected_file_records=protection['files'],
                aggregate_tables=len(list((root/'tables').glob('*.csv'))),nonmedical_figure_groups=len(qa),
                nonmedical_agent_visual_review='PASS' if qa.agent_visual_review_status.eq('AGENT_VISUAL_PASS').all() else 'NOT_COMPLETED',
                git_publication_status=git['status'],git_commit=git.get('commit'),git_remote_verified=git.get('remote_verified',False),
                private_output_root=str(root),input_source_root=c['source_root'],
                private_medical_panel_manifest='figures/private_medical_panels/PRIVATE_MEDICAL_PANEL_MANIFEST_V1.csv')
    atomic_json(root/'release/FINAL_QUALITY_EVALUATION_STATUS_V1.json',status)
    primary=change[change.comparison=='ON_OFF'].set_index('metric');main=dist[dist.scope=='all_available_common_support'].set_index('branch')
    fid_delta=float(main.loc['OP_CPFM_OFF','fid']-main.loc['OP_CPFM_ON','fid']);kid_delta=float(main.loc['OP_CPFM_OFF','kid_mc_mean']-main.loc['OP_CPFM_ON','kid_mc_mean'])
    kval=ci.iloc[0]
    lineflags=flags[flags.proxy=='long_line_flag'].set_index('branch').patient_equal_fraction
    # Outcome wording is determined by measured values, never by expected direction.
    methods=bilingual(
        ('# Image quality methods / 图像质量方法',''),
        (f'We evaluated saved historical Original Progression–CPFM ON/OFF images, with {c["ON_images"]:,} exact node/index/seed pairs from {c["matched_nodes"]:,} nodes and {c["matched_patients"]} patients. Original Progression refers to the frozen20260921 epoch120 historical route, not Clean/S2. Full and generic-prompt Plain SD were image/seed-aligned auxiliary routes, not single-variable CPFM contrasts.',
         f'评价已保存的历史 Original Progression–CPFM ON/OFF 图像：{c["ON_images"]:,} 对精确节点/图序/种子配对，来自 {c["matched_nodes"]:,} 节点和 {c["matched_patients"]} 名患者。Original Progression 指冻结的 20260921 epoch120 历史路线，不是 Clean/S2。Full 与通用提示词 Plain SD 为图像/种子对齐的辅助路线，不是 CPFM 单变量对照。'),
        (f'The reference contained {c["real_unique_views"]:,} unique native historical views from {c["real_unique_exams"]:,} exams and {c["real_unique_patients"]} patients, after identity-based removal of repeated LOCF slot references. These images were not query-year ground truth. Distribution comparisons primarily used {c["common_patients"]} common patients. The prespecified balanced sensitivity selected2 views per patient without quality/outcome filtering.',
         f'REAL 参考为 {c["real_unique_views"]:,} 张独立原始历史 view，来自 {c["real_unique_exams"]:,} Exam 和 {c["real_unique_patients"]} 名患者；按身份去除重复 LOCF slot 引用，未剔除像素重复图。这些不是 query 年真值。主分布比较限于 {c["common_patients"]} 名共同患者；事前冻结的均衡敏感性分析每人选 2 图，不按质量/结局筛选。'),
        ('Distribution features were extracted in frozen FP32 evaluation using torch-fidelity0.4.0 inception-v3-compat2048 with official evaluation weights, native RGB dimension-grouped batches16, internal TF1-compatible299 bilinear resize without antialias and (uint8−128)/128. FID used float64 sample covariance and the PSD Bures formula. KID was unbiased cubic polynomial MMD squared (gamma1/2048, intercept1),50 fixed-seed subsets512; negative estimates were retained. [Implementation](https://torch-fidelity.readthedocs.io/en/latest/).',
         '分布特征采用冻结 FP32 评价：torch-fidelity0.4.0 的 inception-v3-compat2048 和官方评价权重；原始 RGB 按尺寸分组、batch16，内部 TF1 兼容 299 双线性 resize，无 antialias，值域为 (uint8−128)/128。FID 使用 float64 样本协方差和 PSD Bures 公式；KID 是无偏三次多项式 MMD²（gamma1/2048、常数1），固定种子 50 次子集、每次 512 图，保留负估计。[实现来源](https://torch-fidelity.readthedocs.io/en/latest/)。'),
        ('ON/OFF native RGB512 change used MAE, RMSE and bit-exact equality; gray SSIM used BT.601 float64/data_range255/Gaussian sigma1.5/window11/population covariance. Calibrated AlexNet LPIPSv0.1 used RGB[-1,1], not clinical truth. Cached normalized-layer calculations matched the official forward on anonymous toy data. [LPIPS source](https://github.com/richzhang/PerceptualSimilarity).',
         'ON/OFF 原始 RGB512 变化采用 MAE、RMSE 和逐像素恒等；灰度 SSIM 使用 BT.601 float64/data_range255/高斯 sigma1.5/窗口11/总体协方差。AlexNet LPIPSv0.1 使用 RGB[-1,1]，不以其代表临床真值；缓存标准化层计算已与官方前向在匿名数据上核验一致。[LPIPS 来源](https://github.com/richzhang/PerceptualSimilarity)。'),
        ('Texture proxies used whole-image512 PIL bilinear resize,32×32 nonoverlapping block contrast,3×3 Sobel/Laplacian, demeaned Hann-window FFT with DC removed and radial high band≥0.25 cycles/pixel, black<5/white>250 and fixed line/cross candidates. Native sizes and modes were retained. None is an anatomical ROI or composite medical realism score.',
         '纹理代理采用全图 512 PIL 双线性 resize、32×32 无重叠局部对比度、3×3 Sobel/Laplacian、去均值 Hann 窗 FFT（去 DC，径向高频≥0.25 周期/像素）、灰度黑<5/白>250及固定直线/十字候选。保留原始尺寸和 mode；这些不是解剖 ROI，也未合成为医学真实度总分。'),
        ('Image-to-node-to-patient means used equal patient weights; native REAL used view-to-exam-to-patient means. Patient-cluster percentile bootstrap used2000 replicates/seed20261012. The balanced full-U KID OFF-minus-ON contrast resampled the same patient weights in REAL/ON/OFF. KID subset Monte Carlo SD is not a patient CI. FID CI was prespecified NOT_RUN; balanced1718-image covariance is rank-limited relative to2048 features.',
         '采用 image→node→patient 均值和患者等权；原始 REAL 采用 view→Exam→patient 均值。患者聚类百分位 bootstrap 为 2,000 次/seed20261012。均衡 full-U KID 的 OFF−ON 差异对 REAL/ON/OFF 同时重采样同一患者权重。KID 子集蒙特卡洛标准差不是患者区间；FID CI 在测量前声明未做。均衡 1,718 图相对 2,048 特征维数存在协方差秩限制。'))
    results=bilingual(
        ('# Measured results / 实测结果',''),
        (f'All50,280 frozen images decoded successfully;101 images beyond the first member of exact-pixel equivalence groups were retained. Feature extraction,9,270 primary changes,9,270 Full/ON auxiliary changes and46,350 within-node seed-pair comparisons were completed. Zero-response primary pairs: {int(primary.loc["bit_exact","zero_response_images"])}.',
         f'全部 50,280 张冻结图像成功解码；像素等价组中首张以外的 101 张重复图保留。已完成特征提取、9,270 对主变化、9,270 对 Full/ON 辅助路线变化和 46,350 对节点内种子比较。主配对零响应为 {int(primary.loc["bit_exact","zero_response_images"])} 对。'),
        (f'Under all-available common support, ON FID={main.loc["OP_CPFM_ON","fid"]:.6g} and OFF FID={main.loc["OP_CPFM_OFF","fid"]:.6g}; OFF−ON={fid_delta:.6g}. KID MC means were {main.loc["OP_CPFM_ON","kid_mc_mean"]:.6g} and {main.loc["OP_CPFM_OFF","kid_mc_mean"]:.6g}, respectively (difference={kid_delta:.6g}). These are descriptive distances, without a computed FID confidence interval.',
         f'在全量共同患者支持下，ON FID={main.loc["OP_CPFM_ON","fid"]:.6g}、OFF FID={main.loc["OP_CPFM_OFF","fid"]:.6g}，OFF−ON={fid_delta:.6g}；KID 蒙特卡洛均值分别为 {main.loc["OP_CPFM_ON","kid_mc_mean"]:.6g} 和 {main.loc["OP_CPFM_OFF","kid_mc_mean"]:.6g}，差值={kid_delta:.6g}。这是描述性距离；未计算 FID 置信区间。'),
        (f'Balanced full-U KID OFF−ON={kval.point_estimate:.6g}, paired patient95% interval [{kval.ci_low:.6g},{kval.ci_high:.6g}]. This estimand differs from the Monte Carlo subset mean and remains conditional on the frozen balanced reference. Pixel-change image means: MAE={primary.loc["rgb_mae","image_mean"]:.6g}, RMSE={primary.loc["rgb_rmse","image_mean"]:.6g}, gray SSIM={primary.loc["gray_ssim","image_mean"]:.6g}, LPIPS={primary.loc["lpips","image_mean"]:.6g}.',
         f'均衡 full-U KID OFF−ON={kval.point_estimate:.6g}，配对患者 95% 区间 [{kval.ci_low:.6g},{kval.ci_high:.6g}]。该估计目标不同于蒙特卡洛子集均值，且以冻结均衡参考为条件。逐图变化均值：MAE={primary.loc["rgb_mae","image_mean"]:.6g}、RMSE={primary.loc["rgb_rmse","image_mean"]:.6g}、灰度 SSIM={primary.loc["gray_ssim","image_mean"]:.6g}、LPIPS={primary.loc["lpips","image_mean"]:.6g}。'),
        ('Every branch and sensitivity result, uncertainty type, denominator and low-level proxy is retained in Tables01–12, including nonzero flags, duplicates, rank limitations and unavailable metrics. No quality threshold was used to alter the cohort or choose the visual examples.',
         '表 01–12 保留全部分支、敏感性结果、不确定性类型、分母和低层代理，包括非零标记、重复图、秩限制和不可用指标。没有利用质量阈值改变群体或选择图例。'),
        (f'ON had a lower natural-image distribution distance than OFF, yet the prespecified long-line candidate fraction was {lineflags["OP_CPFM_ON"]:.6g} for ON, {lineflags["OP_CPFM_OFF"]:.6g} for OFF, and {lineflags["REAL_HISTORY_REFERENCE"]:.6g} for REAL under equal-patient aggregation. Thus the different evidence layers do not warrant a blanket statement that ON is clinically superior; line candidates have unknown medical specificity.',
         f'ON 的自然图像分布距离较 OFF 小，但预定义长直线候选的患者等权比例为 ON={lineflags["OP_CPFM_ON"]:.6g}、OFF={lineflags["OP_CPFM_OFF"]:.6g}、REAL={lineflags["REAL_HISTORY_REFERENCE"]:.6g}。不同证据层次不支持笼统声称 ON 临床更优；直线候选的医学特异性未知。'))
    results+='\n'+table(dist,['scope','branch','fid','kid_mc_mean','kid_mc_sd','real_images','generated_images','generated_patients','covariance_rank_limited'])+'\n'+table(ci,['point_estimate','ci_low','ci_high','patients','images_per_patient','repeats'])
    discussion=bilingual(
        ('# Discussion and limitations / 讨论与限制',''),
        ('A controlled CPFM ON/OFF pixel response establishes dependence on the injection path at the saved-image level. It does not identify which branch is clinically more realistic, reconstruct a future thyroid exam, or by itself establish an independent progression benefit. Full and Plain comparisons remain auxiliary because their conditioning and model routes differ.',
         '受控 CPFM ON/OFF 像素响应可表明已保存图像对注入路径的依赖，但不确定哪一臂临床更真实、不重建未来甲状腺 Exam，也不单独建立 Progression 的独立获益。Full/Plain 的条件与模型路线不同，比较仍是辅助性质。'),
        ('Inception and AlexNet were developed on natural-image/perceptual tasks; their domain mismatch, acquisition overlays, aspect ratios and resize artifacts can affect ultrasound distances. Image counts are not patient counts. Rank-limited balanced FID, Monte Carlo KID variation, patient-cluster uncertainty and exact duplicate pixels are separate limitations, not interchangeable uncertainty measures.',
         'Inception 与 AlexNet 源于自然图像/感知任务；域偏差、采集覆盖层、宽高比及 resize 伪影均可能影响超声距离。图数不是患者数。秩受限的均衡 FID、KID 蒙特卡洛变化、患者聚类不确定性和像素重复是不同限制，不能互相替代。'),
        ('No verified thyroid ROI/annotation asset was available in the supplied frozen sources. No clinician or medical visual reviewer evaluated the images; the blinded template has no scores. REAL pixels were withheld from display because verified de-identification was unavailable. Generated comparison panels remain restricted server files and were not transferred or published.',
         '所提供冻结源中没有已验证的甲状腺 ROI/标注资产。未开展医生或医学视觉评阅，盲评模板无任何分数。REAL 像素因未取得已验证脱敏版本而暂不显示；生成对照图板仍为服务器限制文件，未传出或发布。'),
        ('Offline quality statistics and visual comparisons assess distribution, artifacts and structural visibility. They do not amount to physician-certified medical authenticity or a new independent downstream HT validation.',DISCLAIMER))
    writings={'IMAGE_QUALITY_METHODS_CN_EN_V1.md':methods,'IMAGE_QUALITY_RESULTS_CN_EN_V1.md':results,
              'IMAGE_QUALITY_DISCUSSION_AND_LIMITATIONS_CN_EN_V1.md':discussion}
    for filename,text in writings.items():(root/'writing'/filename).write_text(text,encoding='utf-8')
    captions=[]
    for r in qa.itertuples():captions.append((root/r.caption_path).read_text())
    for i,desc in enumerate(['Branch image/patient/node support','Native real reference and preprocessing','All feature-distance comparisons and limitations',
                              'Exact ON/OFF and Full/ON paired changes','Whole-image technical proxies','Six-seed node diversity',
                              'Prespecified artifact proxy flags and absent medical review','Claim-to-evidence boundaries','Paired patient KID uncertainty',
                              'Unavailable or unperformed metrics','Native image sizes and modes','Exact pixel duplicate groups'],1):
        captions.append(bilingual((f'Table{i:02d}. {desc}. Values originate from measured aggregate CSVs; no patient identities or clinical dates are included.',
                                   f'表{i:02d}。对应聚合表呈现{["分支支持","真实参考及预处理","分布距离及限制","严格配对变化","全图技术代理","六种子多样性","伪影代理及评阅范围","结论—证据边界","配对患者 KID 不确定性","不可用/未开展指标","原始尺寸与 mode","像素重复等价组"][i-1]}。数值来自实际聚合 CSV，不含患者身份或临床日期。')))
    (root/'writing/FIGURES_AND_TABLES_CAPTIONS_CN_EN_V1.md').write_text('\n\n'.join(captions),encoding='utf-8')
    handoff=bilingual(('Add this quality subsection after the frozen seven-branch CLIMATv2 outcome analysis as complementary, non-independent image evidence. Keep the original outcome tables and checkpoints unchanged. Cite exact matching and common-reference support, present the complete auxiliary and sensitivity results, and keep clinical authenticity/target-truth claims out of the conclusions.',
                        '将本质量小节接在冻结的七分支 CLIMATv2 结局分析之后，作为补充而非新的独立图像证据。保留原结局表和 checkpoint 不变，说明严格匹配与共同参考支持，呈现完整辅助/敏感性结果，不把临床真实性或目标真值主张写进结论。'))
    (root/'writing/PAPER_EXPERIMENT_INTEGRATION_V1.md').write_text(handoff,encoding='utf-8')
    (root/'release/NEXT_STAGE_PAPER_HANDOFF_V1.md').write_text(handoff+'\n'+DISCLAIMER+'\n',encoding='utf-8')
    metrics_columns=['comparison','metric','images','nodes','patients','image_mean','image_median','image_sd','image_q25','image_q75','image_p025','image_p975','patient_equal_mean','patient_ci_low','patient_ci_high','zero_response_images']
    text=f'''# 历史 Original Progression–CPFM 生成图像质量评价结果 V1

## 1. 身份与状态

任务：{c['task']}。机器：5090-lin / RTX5090。更新时间：{status['updated_utc']}。实验状态：`{final}`；Git 独立状态：`{git['status']}`。本报告仅使用实际完成值，不设置质量性能成功阈值。

## 2. 输入根及模型身份

只读主源：`{c['source_root']}`；独立输出：`{root}`。采用历史 Original Progression 20260921 epoch120、旧冻结 F16/B14 及历史 ON；新 OFF 为同种子 CPFM 零注入，不是 Plain SD，也不是 Clean/Strict/S2。历史采样为 PNDM30、CFG7、RGB512、medical LoRA。本轮没有重新采样旧 ON、OFF 或辅助图。前轮只有36对固定历史ON重采样通过解码像素严格恒等，不能改称全部9,270张ON曾被重新采样核验。

## 3. 全量计数

{table(counts)}
ON/OFF 各 9,270 张，精确主配对 9,270 对；1,545 节点×每节点6图；860 名唯一患者。五个图像集合共 50,280 张均已解码，失败0。唯一 seed 以每图 master_seed 配对核验，不把技术种子计作独立患者。

## 4. 匹配范围及失败/缺失

患者、节点、relative slot、image index、确切 seed 和现存 PNG 已逐一核验。主配对缺失0、错误0。Full/Plain 各 9,270 图与主清单具有同患者、同节点、同图序、同种子；但路线和条件不同，不能据此声称单模块贡献。主图像配对分母始终保持 860 人/9,270 对。

## 5. REAL 参考的合法选择

仅选 K6 source_type=REAL 且在冻结原始历史中有精确患者/原始日期/路径身份的 native views；剔除重复 LOCF 引用，不按疾病结局、报告或图像分数筛选。参考为 859 人、3,154 Exam、13,200 唯一原始 view；有1名生成患者没有原始 REAL slot 支持。参考不是 query 年真实目标，主分布比较使用共同 859 人；另完整报告 all-available native-support 敏感性。均衡方案测量前冻结为每人2图，共1,718图/分支，ON/OFF 选同一批配对。

## 6. 辅助组身份

Full：`{c['full_status']}`，从历史 full manifest 中匹配 9,270/12,036 图；属于历史完整路线，不是 Original Progression 单变量对照。Plain：`{ext['plain_status']}`，9,270 图通用提示词、相同节点与种子；历史 manifest/config/release 显示 patient_history_used=0、b16_specific_condition_used=0、Train-only/Test0。两辅助组只解释路线背景，不替代本轮 OFF。

## 7. 预处理、软件与数值自检

全图技术统计：PIL bilinear 512×512，保留 native尺寸/mode；无对比度均衡或医学 ROI。特征：torch-fidelity0.4.0 标准 Inception2048/官方权重、原 RGB 按尺寸分组、内部 TF1兼容299/无 antialias/(uint8−128)/128；FP32/TF32关闭。Torch `{runtime['torch_version']}`，运行时 CUDA `{runtime['torch_cuda_runtime']}`，不是把 nvidia-smi 的驱动最大 CUDA 当作训练 runtime。LPIPS0.1.4/AlexNetv0.1 冻结权重与校准；SSIM0.25.2/BT.601gray/data_range255/高斯1.5/window11/总体协方差。五类匿名测试通过；FID/KID 排列不变、均值平移、自比与无偏负 KID、bootstrap核聚合等式均已验证。LPIPS缓存/官方前向 toy最大误差0；匿名同输入 Inception 微小 FP32舍入差约3.22e−6，不影响或冒充旧 ON 的解码像素恒等要求。

## 8. FID/KID 实测完整矩阵

{table(dist,['scope','branch','fid','kid_mc_mean','kid_mc_sd','real_images','generated_images','real_patients','generated_patients','covariance_rank_limited'])}
KID 为原始 MMD²尺度，无乘1000或截断。MC均值/SD来自50×512子集；SD不是患者CI。全量共同支持 OFF−ON：FID={fid_delta:.7g}、KID_MC={kid_delta:.7g}。均衡 full-U KID 差异及患者区间：

{table(ci,['point_estimate','ci_low','ci_high','bootstrap_sd','patients','images_per_patient','repeats'])}
2000次同患者配对重采样，full-U目标不同于MC子集均值。均衡协方差秩受限（1,718图<2,048维）；数值有效不等于稳定临床结论。FID CI按测量前资源声明未计算，不将距离大小直接称为临床真实性或自动声称 FID 显著。

## 9. 严格图像变化与零响应

{table(change,metrics_columns)}
SSIM越低/LPIPS越高仅表示 ON/OFF 变化更大。两臂均不是ground truth；Full/ON只为辅助路线差异。零响应按全 native RGB array_equal，不用文件哈希。

## 10. 技术代理与可疑结构标记

技术 CSV：`metrics/TEXTURE_AND_ARTIFACT_DESCRIPTIVES_V1.csv`；含灰度mean/std/p1/p99、动态范围、32×32局部std、Sobel、Laplacian、Hann/DC排除FFT高频、黑白/边框/UI边缘比例。以下列出主要患者等权统计；所有15项完整精度留在表05。

{table(tech[tech.metric.isin(['gray_std','sobel_gradient_mean','high_frequency_fraction','border_black_fraction'])],['branch','metric','images','nodes','exams','patients','patient_equal_mean','patient_ci_low','patient_ci_high'])}
{table(flags,['branch','proxy','flagged_images','images','image_flag_fraction','patient_equal_fraction'])}
低对比/平滑/极端黑白/频谱/长直线/十字阈值事前固定；这类代理可能误检真实界面或采集UI，没有临床校准。未合成为医学真实度总分，未用其删图。ROI/病理认证：NOT_AVAILABLE。

## 11. 重复、节点多样性与统计单位

{table(duplicates)}
全队列固定原始像素坐标数组找候选、全 native RGB ndarray比较证实：首张以外101张像素重复；没有执行SHA/MD5/CRC/文件校验和，也没有移除重复图。

{table(div,['branch','metric','seed_pairs','nodes','patients','images','pair_mean','node_mean','patient_equal_mean','patient_ci_low','patient_ci_high'])}
每臂23,175技术种子配对，总46,350；每节点全6图/15对。image→node→patient等权；REAL为view→Exam→patient。技术重复不作为独立患者，2000患者bootstrap区间与图像分布/MC变化分开。

## 12. 固定图板及隐私

固定12等距节点：前6主图、后6补充，不再择优。四组医学图板共{panels['node_panels']}个节点级PNG，所在绝对目录：`{root}/figures/private_medical_panels/`；清单 `PRIVATE_MEDICAL_PANEL_MANIFEST_V1.csv`。MedicalPlate01/02为ON/OFF全部6seed；03为统一0–255差值图及预定中心crop（非ROI）；04为Full/Plain辅助图。原图不改、不锐化、不造毫米尺度；目录700/PNG600权限。没有传出/下载/公开任何医学图片。REAL图板像素因缺乏已验证脱敏显示流程而WITHHELD；保留背景说明和未开展原因，不能写作完成了真实医学可视对照。

## 13. 非医学图件与源数据

8组真实聚合图：每组sourceCSV、RGB600dpi PNG、可编辑文字SVG、嵌字PDF、英汉caption。自动实际绘制文字越界/导出检查PASS，图件清单如下；实际agent/human视觉状态单列，不冒充人工检查。

{table(qa,['filename','source_rows','image_dimensions','dpi','auto_export_status','agent_visual_review_status','human_review_status'])}
绝对目录：`{root}/figures/nonmedical/`、`{root}/figures/source_data/`、`{root}/figures/captions/`。聚合表{status['aggregate_tables']}份位于 `{root}/tables/`，数值保留完整浮点精度，报告仅格式化显示。

## 14. 实际评阅状态

非医学agent视觉复核：`{status['nonmedical_agent_visual_review']}`；human_review=NOT_RUN。医学agent复核/医师盲评均NOT_RUN。空白盲评模板{expert['template_rows']}行，医生数0、实际评分0；仅服务器私有随机映射，不生成假分数。

{table(na)}

## 15. 英汉 Methods/Results/Discussion

目录 `{root}/writing/` 中五份：IMAGE_QUALITY_METHODS_CN_EN_V1.md、IMAGE_QUALITY_RESULTS_CN_EN_V1.md、IMAGE_QUALITY_DISCUSSION_AND_LIMITATIONS_CN_EN_V1.md、FIGURES_AND_TABLES_CAPTIONS_CN_EN_V1.md、PAPER_EXPERIMENT_INTEGRATION_V1.md。每段英文后紧跟中文，术语ledger独立保存。采用图表与科学写作skill约束：评价证据与医学/下游主张分开，呈现全部混合/负面结果，不套用性能门槛。

## 16. GitHub 发布

仅允许新增 `B16v1_Progression/experiments/historical_six_branch_checkpoint_replay_v1/analysis/generation_quality_v1/`，代码、匿名toy、聚合表/非医学图及双语材料；禁止医学PNG、pair-map、患者ID/日期/逐患者特征与概率、权重和临床报告。当前状态 `{git['status']}`；commit `{git.get('commit')}`；真实远端核验 `{git.get('remote_verified',False)}`。作者 study / l3026714004@outlook.com。Git状态与实验完成状态独立，不以本地commit冒充push。

## 17. 资源、恢复与原资产保护

GPU冻结评价batch16/FP32，特征阶段自身峰值allocated={runtime.get('allocated_peak_mib')}MiB；CPU解码6worker、LPIPS/SSIM4worker，未动其他作业/共享Torch/驱动。唯一主链PID锁防重启，逐批缓存核验同 frozen queue/encoder/runtime后可恢复；曾因汇总输出中重复pair_mean字段退出，修复后仅恢复聚合阶段，未重算已有FID/KID、配对或特征，历史失败日志保留且不是当前状态。原源和模型/manifest/已冻结split等元数据postflight `{protection['status']}`：{protection['files']}个保护记录；比较path/resolved_path/size/mtime_ns/inode，非内容hash，不能把元数据核验夸为全部文件内容byte-proof。新生成训练0、新SD采样0、新CLIMATv2训练0、Official Test访问0。

## 18. 负面/混合结果与论文交接

全量与均衡/原支持敏感性矩阵及辅助路线结果完整保留；ON的特征距离较OFF小，但长直线候选患者等权比例 ON={lineflags['OP_CPFM_ON']:.7g}、OFF={lineflags['OP_CPFM_OFF']:.7g}、REAL={lineflags['REAL_HISTORY_REFERENCE']:.7g}，不能给ON统一“临床更优”结论。有101张像素重复、1名患者无REAL支持、均衡FID秩限制、技术伪影代理未知特异性、无FID CI/医学ROI/专家认证、REAL显示缺少已验证脱敏。不是“所有指标越高越好”，不因方向或数值不理想换样本/权重/编码器。接入既有七分支论文时只新增离线质量小节，不修改封存结局表或checkpoint；可补合法专家阅片，但须实际资源、授权和盲评流程，不能从自动指标推断医师结论。

{DISCLAIMER}
'''
    (root/RESULT).write_text(text,encoding='utf-8')
    if build_package:
        audit=public_package(root,text,status);print({'public_scope':audit['status'],'public_files':audit['file_count']},flush=True)
    recovery=root/'release/MEASUREMENT_RECOVERY_V1.json'
    if recovery.exists():
        rec=read_json(recovery);rec.update(status='RESOLVED_SAME_INPUT_AGGREGATION_COMPLETED',completion_marker='MEASUREMENT_COMPLETED_V1.json')
        atomic_json(recovery,rec)
    state(root,final,git_publication_status=git['status'],result_document=RESULT)
    print({'result_document':RESULT,'experiment_status':final,'git':git['status']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',required=True);p.add_argument('--build-public-package',action='store_true');a=p.parse_args();main(a.output_root,a.build_public_package)
