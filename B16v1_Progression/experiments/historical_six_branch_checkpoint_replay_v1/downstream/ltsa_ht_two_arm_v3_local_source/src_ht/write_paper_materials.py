"""Evidence-first bilingual Methods/Results/Limitations, no invented performance or novelty."""
import argparse
import pandas as pd
from common import config,setup,jread,cout,ARMS


def bi(en,cn):return en+'\n\n'+cn+'\n\n'


def main(path):
    c=config(path);root=setup(c['project_private']);counts=pd.read_csv(root/'tables/REAL_COHORT_NATIVE_SUPPORT_V3.csv')
    cout(root/'writing/TERMINOLOGY_LEDGER_V3.csv',[dict(canonical_en=a,canonical_zh=b,boundary=d) for a,b,d in [
        ('LTSA-HT adapted predictor','基于LTSA的HT适配预测器','not unmodified survival LTSA'),('REAL Exam','真实检查','observed native date, not LOCF'),('bounded synthetic annual-query token','有真实历史界限的合成年查询token','not observed Exam/future truth'),('average precision (AP)','平均精度','sklearn average_precision_score'),('internal validation','内部验证','not Official Test')]])
    methods=bi('We implemented an LTSA-based HT adapted predictor from the existing local LTSA source without downloading or modifying it. The shared ImageNet-initialized ResNet18 encoded every native ultrasound view; views were averaged within each observed examination. The original fixed sinusoidal temporal encoding, a one-layer four-head causal Transformer (512-dimensional ReLU feed-forward, post-norm, dropout0.25) and the latest real-history representation were used for a binary logit head. The survival and step-ahead heads were not used.',
               '基于现有本地LTSA源码实现HT适配预测器，未下载或改动原源码。共享ImageNet初始化ResNet18编码全部原生超声view，并在真实检查内均值聚合。保留原固定sin/cos时间编码、1层4头因果Transformer（FFN512/ReLU/postnorm/dropout0.25），用末真实历史表示经二分类logit头预测；不使用生存或step-ahead头。')
    methods+=bi('The frozen860Train and189Val patients and final-year0/1 labels were shared by both arms. Native histories included4,585/991 examinations and18,413/3,861 views in Train/Val. Only real dates before the target year were eligible. Relative times were binned by floor(days from first observed examination/30.4375), with no target-date gap. Target-year image/report contents and Official Test were not read.',
                '两臂共享冻结860Train/189Val患者及最终年度0/1标签，Train/Val原生历史分别4,585/991检查和18,413/3,861view。仅目标年之前真实日期合格；相对时间为floor(距首真实检查天数/30.4375)，不注入目标日期gap，不读目标年影像/报告内容或Official Test。')
    methods+=bi('Augmentation was fixed before formal validation:686 historical Original Progression CPFM-ON nodes (4,116saved images), anchored to the first observed real date and strictly bounded by first/last real-history examinations, were inserted as explicitly synthetic query tokens. Six views per node were averaged to one token.758target-year nodes and101unknown-future nodes after the latest real exam were excluded. All real exams and the same patient support were retained; Val was always REAL-only and the readout token remained the latest real exam.',
                '增强在正式验证前冻结：686个历史Original Progression CPFM-ON节点（4,116张已存图）由首真实日期锚定且严格在首末真实历史之间，作为明确合成查询token插入。每节点六图均值为一个token；排除758个目标年节点、101个末真实之后未知将来节点。保留全部真实检查和相同患者支持；Val始终REAL-only，读出token始终末真实检查。')
    methods+=bi('Both conditions used three fresh seeds42/3407/2026, FP32 AdamW (learning rate1e-4, weight decay1e-4), a maximum60epochs with5epoch warmup/cosine, patience12 and minimum15epochs, and effective patient batch8 (microbatch4/accumulation2). Selection prioritized Val ROC-AUC, AP, lower loss and earlier epoch. Every best checkpoint was independently strictly reloaded and the same189 ordered positive-class probabilities were verified to1e-6 tolerance, then averaged across seeds. Paired patient-level2,000resample percentile intervals evaluated OP-minus-REAL ROC-AUC/AP differences.',
                '两条件使用42/3407/2026三个fresh seed，FP32 AdamW（lr1e-4/wd1e-4），最多60epoch、5epoch warmup/cosine、patience12/min15，有效患者batch8（微批4/累积2）。固定选模次序Val AUC、AP、较低loss、较早epoch。各best独立strict重载，同189人有序正类概率1e-6容差核验后种子平均；OP−REAL的AUC/AP差用患者配对2,000次百分位重抽样区间评价。')
    comparison=root/'tables/LTSA_TWO_ARM_ENSEMBLE_COMPARISON_V3.csv';results=''
    if comparison.exists():
        d=pd.read_csv(comparison).set_index('arm');ci=pd.read_csv(root/'tables/LTSA_OP_MINUS_REAL_PAIRED_BOOTSTRAP_V3.csv').set_index('metric')
        a=d.loc[ARMS[0]];b=d.loc[ARMS[1]];r=ci.loc['ROC_AUC'];p=ci.loc['AP']
        results=bi(f'All six independent runs completed strict reload verification on the shared189-patient internal validation cohort. Probability ensembles achieved REAL-only ROC-AUC{a.ROC_AUC:.6f}/AP{a.AP:.6f}, and Original Progression ROC-AUC{b.ROC_AUC:.6f}/AP{b.AP:.6f}. The paired ROC-AUC contrast was{r.delta_OP_minus_REAL:+.6f} (95%interval{r.ci_low:+.6f} to{r.ci_high:+.6f}); the AP contrast was{p.delta_OP_minus_REAL:+.6f} ({p.ci_low:+.6f} to{p.ci_high:+.6f}).',
                   f'全部六个独立run在同189内部验证患者完成严格重载。概率集成REAL-only ROC-AUC{a.ROC_AUC:.6f}/AP{a.AP:.6f}，Original Progression ROC-AUC{b.ROC_AUC:.6f}/AP{b.AP:.6f}；配对AUC差{r.delta_OP_minus_REAL:+.6f}（95%区间{r.ci_low:+.6f}至{r.ci_high:+.6f}），AP差{p.delta_OP_minus_REAL:+.6f}（{p.ci_low:+.6f}至{p.ci_high:+.6f}）。')
        results+=bi('Threshold-dependent metrics were estimated with the prespecified Val Youden rule, with a fixed0.5 sensitivity table. These are internal-validation estimates, not independent test generalization; intervals crossing zero do not demonstrate significant improvement. No performance criterion was used to discard runs or redesign mapping.',
                    '阈值相关指标按预设Val Youden规则计算，并提供0.5阈值敏感性表。它们是内部验证估计而非独立Test泛化；跨0区间不证明显著改善，没有按性能弃run或重设计映射。')
    else:results=bi('NOT_RUN: the complete two-arm ensemble and paired comparison are pending. Started, partially trained or smoke-tested runs are not complete performance evidence.','NOT_RUN：完整双臂集成和配对对比待完成；启动、部分训练或烟测不是完整性能证据。')
    limitations=bi('This experiment changes training data exposure, not an isolated generator-module intervention. The bounded686-node augmentation and all-native variable-length LTSA histories differ from historical CLIMATv2 fixedK6/LOCF and full9,270-image augmentation. Within-predictor differences are more interpretable than absolute ranking across architectures, but do not demonstrate that a specific progression module caused an effect.',
                   '本实验改变训练数据暴露，不是单生成器模块消融。686有界合成节点及LTSA全部原生变长历史不同于旧CLIMAT固定K6/LOCF和完整9,270图增强。模型内差值比跨架构绝对排名更可解释，但不证明特定Progression模块导致效应。')
    limitations+=bi('The single internal Val cohort was also used for early stopping, checkpoint and threshold selection. Patient bootstrap intervals condition on selected ensembles and omit training/selection uncertainty; three seed variability is not a patient interval. Acquisition/UI cues, image-domain shift and variable observation support may affect predictions. Upstream historical generators were inherited rather than newly retrained or clinically certified; thyroid anatomy authenticity and external/Official Test generalization were not evaluated.',
                    '同一内部Val也用于早停、选checkpoint和阈值；患者bootstrap以已选集成为条件，不涵盖训练/选模不确定性，三个种子变异不是患者区间。采集/UI线索、图像域差异和观察支持可能影响预测。上游历史生成器继承而非重训或临床认证；未评价甲状腺解剖真实性或外部/Official Test泛化。')
    for n,text in [('LTSA_HT_METHODS_CN_EN_V3.md',methods),('LTSA_HT_RESULTS_CN_EN_V3.md',results),('LTSA_HT_LIMITATIONS_CN_EN_V3.md',limitations)]:
        (root/'writing'/n).write_text('# '+n.replace('_',' ')+'\n\n'+text,encoding='utf-8')
    (root/'writing/CLAIM_EVIDENCE_MAP_V3.md').write_text('# Claim-evidence map\n\nLocal architecture: actual source/smoke. Canonical support: native manifest audit. Synthetic temporal eligibility: frozen boundedquery ledger. Performance: actual six strictreloads+ensembles only. No claim of unmodified survival-paper reproduction, external generalization, clinical image authenticity or isolated Progression causality. Earlier CLIMAT numbers are supplied-runbook references, not recomputed values.\n',encoding='utf-8')
    print({'bilingual_materials':3,'performance_text':'ACTUAL' if comparison.exists() else 'NOT_RUN_PLACEHOLDER'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();main(a.config)
