"""Aggregate-only seven tables and evidence-bounded bilingual supplement text."""
import argparse
import shutil
import pandas as pd
from review_common import setup,read_json,csv_out,json_out


def bi(en,cn):return en+'\n\n'+cn+'\n\n'


def main(output_root):
    root=setup(output_root);c=read_json(root/'protocol/FROZEN_VISUAL_AND_ARTIFACT_REVIEW_CONTRACT_V1.json')
    assert read_json(root/'release/TECHNICAL_AGGREGATION_COMPLETE_V1.json')['status']=='COMPLETE'
    cohort=[dict(cohort='FULL_PAIRED_PRIMARY',pairs=9270,images=18540,nodes=1545,patients=860,exams=None,meaning='two arms, no population filtering'),
            dict(cohort='REAL_NATIVE_REFERENCE',pairs=None,images=13200,nodes=None,patients=859,exams=3154,meaning='historical views, NOT target-year truth'),
            dict(cohort='CORE_FIXED12',pairs=72,images=144,nodes=12,patients=12,exams=None,meaning='inherited nodes, all six indices'),
            dict(cohort='FLAG_STRATIFIED',pairs=48,images=96,nodes=48,patients=48,exams=None,meaning='one distinct patient per pair; no calibrated population weights'),
            dict(cohort='CORE_PLUS_STRATIFIED',pairs=120,images=240,nodes=60,patients=60,exams=None,meaning='zero patient/pair overlap'),
            dict(cohort='CLINICIAN_CANDIDATES_SUBSET_OF_CORE',pairs=36,images=72,nodes=12,patients=12,exams=None,meaning='indices0,2,5; zero actual expert ratings'),
            dict(cohort='UNASSIGNED_OFF_ONLY_FULL_QUEUE_RETAINED',pairs=420,images=840,nodes=None,patients=None,exams=None,meaning='not covered by four sampling strata; retained in full-queue analysis')]
    cov=pd.read_csv(root/'metrics/REVIEW_SAMPLE_COVERAGE_AND_SELECTION_V1.csv')
    for r in cov.to_dict('records'):cohort.append(dict(cohort=r['stratum'],pairs=r['selected_pairs'],images=2*r['selected_pairs'],nodes=r['selected_pairs'],patients=r['selected_patients'],
                                                    all_queue_pairs=r['all_queue_pairs'],eligible_pairs=r['eligible_noncore_pairs'],meaning=r['inclusion_probability']))
    csv_out(root/'tables/Table01_Review_Cohort_and_Coverage.csv',cohort)
    joint=pd.read_csv(root/'metrics/FULL_QUEUE_PAIRED_PROXY_JOINT_COUNTS_V1.csv')
    joint['reference_annotation_status']='NOT_RUN';joint['TP']=None;joint['FP']=None;joint['TN']=None;joint['FN']=None
    csv_out(root/'tables/Table02_Flag_vs_Visual_Category.csv',joint)
    shutil.copyfile(root/'metrics/FULL_QUEUE_REGION_LEVEL_AGGREGATES_V1.csv',root/'tables/Table03_Region_Artifact_Distribution.csv')
    cis=pd.read_csv(root/'metrics/PAIR_CLUSTER_BOOTSTRAP_V1.csv');cis['human_structure_visibility_comparison']='NOT_RUN';cis['clinician_score_comparison']='NOT_RUN'
    csv_out(root/'tables/Table04_CPFM_ON_OFF_Qualitative_Comparison.csv',cis)
    limitations=[
      dict(item='regions',status='TECHNICAL_ONLY',value='outer26; center102:410; intermediate; multizone',boundary='position is not thyroid ROI; no mm scale'),
      dict(item='overlapping_regions',status='RETAINED',value='segments can cross regions; crosses union both arms',boundary='region proportions need not sum to one'),
      dict(item='cross_heuristic',status='ORIGINAL_DEFINITION_REUSED',value='average coordinates and projected segment extents',boundary='not guaranteed exact physical intersection; multiple candidates per image'),
      dict(item='resize',status='FIXED',value='whole512 PIL bilinear; fixed center crop192:320',boundary='resampling/aspect ratio and acquisition composition can affect counts'),
      dict(item='48_case_weights',status='NOT_ESTABLISHED',value='four strata with distinct-patient priority',boundary='not a prevalence estimator; no quota filling duplicates'),
      dict(item='four_strata_residual',status='RETAINED',value='420 OFF-only pairs',boundary='not silently forced intoS4 or removed'),
      dict(item='pixel_visual_review',status='NOT_RUN',value='zero actual observers',boundary='no observed false positives, anatomy or visibility grades'),
      dict(item='expert_scale_examples',status='NOT_RUN',value='no nonpatient example images or clinician scale training',boundary='prepared verbal anchors alone do not establish validated reliability'),
      dict(item='public_medical_images',status='NOT_APPROVED',value='0',boundary='private storage does not prove PHI clearance'),
      dict(item='clinical_or_future_truth',status='NOT_ESTABLISHED',value='no verified ROI/pathology/future-year matched truth',boundary='no Dice/IoU or clinical quality certification')]
    csv_out(root/'tables/Table05_Sensitivity_and_Unknowns.csv',limitations)
    shutil.copyfile(root/'metrics/OPTIONAL_EXPERT_RATING_STATUS_V1.csv',root/'tables/Table06_Expert_Rating_Status.csv')
    evidence=[dict(claim='Prior natural-image distances favored ON',evidence='prior completed quality CSV only; not recalculated',level='PREVIOUS_STATISTICAL',boundary='domain-biased features; not clinical realism'),
              dict(claim='All prior flags reproduce exactly',evidence='31740 current decoded images;0 mismatches',level='CURRENT_TECHNICAL',boundary='reproducibility does not establish anatomical specificity'),
              dict(claim='ON has more line/cross candidates',evidence='full primary9270pairs;860patient paired bootstrap',level='CURRENT_TECHNICAL',boundary='possible interface, UI, acquisition or generated geometry; origins unobserved'),
              dict(claim='Candidate positions quantified',evidence='whole512 geometric zones; multi-zone union retained',level='CURRENT_GEOMETRIC',boundary='not gland segmentation'),
              dict(claim='Fixed review assets prepared',evidence='72core+48stratified pairs;36expert candidate pairs',level='ENGINEERING_PREPARATION',boundary='no image prettiness selection; no actual viewing'),
              dict(claim='Visible thyroid structure or true artifact',evidence='NOT_RUN',level='NO_HUMAN_REFERENCE',boundary='technical candidates cannot fill observer fields'),
              dict(claim='Clinical authenticity',evidence='NOT_ESTABLISHED',level='NO_EXPERT_EVIDENCE',boundary='no doctors, no scores, no kappa or ICC'),
              dict(claim='Prediction benefit or disease information',evidence='NOT_EVALUATED_THIS_TASK',level='OUT_OF_SCOPE',boundary='no new HT evaluation/Test access, no quality-to-AUC inference')]
    csv_out(root/'tables/Table07_Claim_Evidence_Boundary.csv',evidence)
    one=cis.set_index('metric');repro=pd.read_csv(root/'metrics/ORIGINAL_FLAG_FULL_QUEUE_REPRODUCTION_V1.csv').set_index('branch')
    l=one.loc['line_candidates'];x=one.loc['cross_candidates'];lf=one.loc['old_long_line_flag'];cf=one.loc['old_cross_proxy_flag']
    methods=bi('We performed a frozen, offline positional audit of historical Original Progression with original CPFM injection ON versus OFF. All9,270 matched pairs (1,545 nodes,860 patients) shared authoritative patient/node, image-index and master-seed identity. No image was sampled anew, modified at source, quality-filtered or evaluated on Test.',
               '对历史 Original Progression 及原 CPFM 注入 ON/OFF 开展固定离线位置复核。全部9,270对来自1,545节点、860患者，依据权威患者/节点、图序及 master-seed 配对；无新采样、源图修改、质量筛选或 Test 评价。')
    methods+=bi('The prior fixed12 nodes were inherited unchanged; all six indices supplied72 core pairs. A seeded permutation selected12 distinct-patient pairs from each of four prespecified flag strata, excluding core nodes and previously selected patients. This yielded48 pairs with no core-patient overlap. The nonuniform selection has no calibrated inclusion probabilities. The420 OFF-only residual pairs remain in full-queue statistics rather than being assigned to a false stratum. Expert candidates were the precommitted core indices0,2,5 (36pairs/72images/12patients).',
                '原12个固定节点保持不变，全部六图序组成72核心配对。按预声明 flag 四层各选择12名互不重复患者，排除核心节点及此前入选患者，得到48对且与核心患者无重叠。该非均匀抽样缺少校准纳入概率；420个未被四层覆盖的 OFF-only 配对仍保留于全量统计。医生候选为核心预设图序0、2、5，共36对/72图/12患者。')
    methods+=bi('All saved ON/OFF and13,200 native REAL views were read on CPU. Prior whole512 bilinear grayscale/Canny50-150/Hough votes50/minlength128/gap10 flags were reused unchanged. LOCALIZATION_DIAGNOSTIC_V1 retained candidate segments, approximate cross intersections and coordinates privately. Zones were outer26pixels, central[102:410), and the remainder. Candidates spanning zones were retained in all intersected zones; cross support was the union of both arms. These are positional zones, not thyroid ROIs.',
                'CPU 读取全部既有 ON/OFF 及13,200个 REAL 原生 view；不改变原全图512双线性灰度、Canny50–150、Hough票数50/最短128/间隔10定义。新诊断定位器私有保存线段、近似十字交点和坐标。区域为外圈26像素、中心[102:410)及其余区域；跨区候选保留各区，十字按双臂并集。区域不是甲状腺 ROI。')
    methods+=bi('Image values were averaged within six-image nodes, then within patients; REAL views were averaged within exams then patients. Paired ON-minus-OFF contrasts used2,000 percentile-bootstrap resamples of860 patients with identical paired patient weights (seed20261012). No confirmatory multiplicity-adjusted claim was made. Case-level human fields stayed NOT_REVIEWED; observer roles were not inferred from machine calculations.',
                '生成图先在每节点六图内平均，再在患者内平均；REAL 先在 Exam 内再在患者内平均。ON−OFF 对比使用860患者同权配对的2,000次百分位 bootstrap，种子20261012；不作未经多重性校正的确证性推断。人工字段保持 NOT_REVIEWED，机器计算不自动获得人工角色。')
    results=bi('This task actually decoded31,740 existing images and reproduced the prior long-line flag, cross flag and line count with zero mismatches. ON contained237,968 line candidates and2,319 approximate cross candidates; OFF contained11,166 and2; REAL contained72,788 and4,181, respectively. Candidate counts, image hits and patients are different denominators.',
               '本轮实际解码31,740张既有图像，原长线 flag、十字 flag 及线段数逐图复现差异为0。ON 为237,968个线段/2,319个近似十字，OFF 为11,166/2，REAL 为72,788/4,181。候选数、命中图像数、患者数属于不同分母。')
    results+=bi(f'Equal-patient mean line-candidate counts were{l.on_patient_mean:.5f} ON and{l.off_patient_mean:.5f} OFF; the paired difference was{l.delta_on_minus_off:.5f} (95% interval{l.ci_low:.5f} to{l.ci_high:.5f}). The corresponding cross-candidate difference was{x.delta_on_minus_off:.6f} ({x.ci_low:.6f} to{x.ci_high:.6f}). The line-hit fraction contrast was{100*lf.delta_on_minus_off:.3f} percentage points; cross-hit contrast was{100*cf.delta_on_minus_off:.3f} points. These contrasts quantify detector responses, not improved or worsened clinical structure.',
                f'患者等权线段数 ON={l.on_patient_mean:.5f}、OFF={l.off_patient_mean:.5f}；配对差值{l.delta_on_minus_off:.5f}（95%区间{l.ci_low:.5f}至{l.ci_high:.5f}）。十字候选差值{x.delta_on_minus_off:.6f}（{x.ci_low:.6f}至{x.ci_high:.6f}）。长线命中比例差{100*lf.delta_on_minus_off:.3f}个百分点，十字差{100*cf.delta_on_minus_off:.3f}个百分点。这些数值仅反映检测器响应，不证明临床结构变好或变差。')
    results+=bi('All fixed core and stratified cases were retained, including high-candidate and unflagged cases. Private panel/card preparation did not constitute visual review. Actual approved human observers, clinicians and rated images were all zero. Visibility and candidate-origin categories were NOT_REVIEWED; proxy-to-human confusion counts and inter-rater agreement were NOT_RUN/NA. REAL pixel display and public medical-image approval remained withheld/pending.',
                '保留全部固定核心与分层病例，包括高候选数及无 flag 情形。私有图板/卡片准备不构成实际复核；真实获批人工、医生及评分图像均为0。可视性和候选来源保持 NOT_REVIEWED，机器与人工混淆数及评分者一致性为 NOT_RUN/NA；REAL 显示和医学图公开许可仍缺失。')
    discussion=bi('The earlier natural-image feature distances were lower for ON, whereas this audit found more line/cross detector responses for ON. The two findings are not contradictory measurements of the same clinical endpoint: feature distributions, geometric candidates, anatomical visibility and clinical utility are separate questions. Neither feature proximity nor a technically significant paired change establishes thyroid tissue authenticity.',
                  '上轮 ON 的自然图像特征距离较小，本轮 ON 的直线/十字检测器响应更多；两者不是对同一临床终点的矛盾测量。特征分布、几何候选、解剖可视性与临床效用是不同问题；特征接近或技术配对差异均不证明甲状腺组织真实性。')
    discussion+=bi('Candidates may correspond to genuine ultrasound interfaces, scanner rulers/text/UI, acquisition geometry or generated linear structures. Without actual local viewing these are competing explanations, not observed case classifications. OFF has fewer algorithmic candidates, but we cannot claim it looks more natural or has more identifiable thyroid anatomy. No annotated thyroid ROI or target-year ground truth was available; no external segmentation model was introduced.',
                   '候选可能来自真实超声界面、标尺/文字/UI、采集几何或生成线性结构；无人实际本地查看时，这些只是待检验解释，不是已观察分类。OFF 算法候选较少，但不能声称其更自然或甲状腺更可辨。缺少甲状腺 ROI 标注和目标年份真值，未引入外部分割模型。')
    discussion+=bi('Resampling, nonclinical approximate intersections, region overlap, differing native view/exam support and nonuniform case sampling constrain interpretation. The fixed48-case set is for future error analysis, not a population prevalence estimate. No human reference means sensitivity, specificity, false-positive rates, Kappa and ICC are unknown rather than zero. Subsequent qualified local review, PHI clearance and an independently justified scale/ROI reference are prerequisites for medical conclusions.',
                   '重采样、非临床近似交点、区域重叠、不同原生 view/Exam 支持以及非均匀抽样限制解释。固定48对用于后续错误分析，不是总体发生率估计。无人参考时灵敏度、特异度、假阳性率、Kappa、ICC 为未知而非0。真实合规本地医生复核、PHI 核准及独立论证的量表/ROI 参考是医学结论的前提。')
    integration=bi('Preserve prior quality Fig01-08 unchanged. Label this supplement Qual-Fig01-06 and Qual-Table01-07. Use Qual-Fig01 for sampling, Qual-Fig02/04 for technical geometry, Qual-Fig03/05 for honest missingness, and Qual-Fig06 for claim boundaries. Private MedicalPanel A-D and cards are pending approval, not public manuscript images; REAL panelE is WITHHELD. Do not select new visually favorable cases.',
                   '旧质量 Fig01–08 不修改；本补充使用 Qual-Fig01–06、Qual-Table01–07。Fig01 说明抽样，Fig02/04 说明技术几何，Fig03/05 展示真实缺位，Fig06 界定主张。私有 MedicalPanel A–D 和卡片待审批，不是公开论文医学图；REAL E 保持 WITHHELD，不新选好看病例。')
    integration+=bi('Suggested manuscript result: original CPFM changes saved image geometry under strict pairing, and lower natural-image distribution distance does not suffice to establish thyroid structural authenticity. Expert review and future-image validity remain open. No new HT training, AUC calculation or Test-content access was conducted in this supplemental audit.',
                    '建议论文表述：原 CPFM 在严格配对下改变既有图像几何；较小自然图像分布距离不足以证明甲状腺结构真实性。专家复核及未来影像有效性仍待验证。本补充未开展新的 HT 训练、AUC 计算或 Test 内容访问。')
    for name,text in [('QUALITATIVE_METHODS_CN_EN_V1.md',methods),('QUALITATIVE_RESULTS_CN_EN_V1.md',results),('ANATOMY_ARTIFACT_DISCUSSION_CN_EN_V1.md',discussion),('PAPER_FIGURE_INTEGRATION_V1.md',integration)]:
        (root/'writing'/name).write_text('# '+name.replace('_',' ')+'\n\n'+text,encoding='utf-8')
    plan=[dict(figure=f'Fig0{i}_{name}',conclusion=claim,archetype=archetype,evidence_source=source,review_risk=risk,medical_pixels=False) for i,name,claim,archetype,source,risk in [
      (1,'Review_Design_and_Cohorts','fixed unequal scopes; no population case estimate','study design','Table01','different denominators'),
      (2,'Line_Cross_Proxy_and_Regions','measured proxy hits and geometric overlap','bars','Table03','not thyroid ROI; overlaps'),
      (3,'Flag_Audit_And_Unknowns','reproduced flags but no human reference','status chart','Table02','no fake TP FP'),
      (4,'Paired_Artifact_Difference','ON minus OFF changes detector response','interval plots','Table04','not medical improvement'),
      (5,'Visibility_Observation_Distribution','240 candidate images have no actual visibility observation','missingness','Table06','unreviewed is not uncertain'),
      (6,'Claim_Evidence_Limitations','technical and medical evidence separated','evidence map','Table07','no authenticity composite')]]
    json_out(root/'protocol/PRE_DRAW_FIGURE_EVIDENCE_PLAN_V1.json',dict(backend='Python',figures=plan,exports='RGB600dpi PNG; editableSVG; embedded TrueType PDF',min_font_pt=8))
    print({'aggregate_tables':7,'bilingual_materials_prepared':4,'figure_plan_groups':6})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',required=True);a=p.parse_args();main(a.output_root)
