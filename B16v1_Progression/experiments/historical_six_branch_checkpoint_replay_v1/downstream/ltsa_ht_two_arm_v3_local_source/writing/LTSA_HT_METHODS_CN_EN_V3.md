# LTSA HT METHODS CN EN V3.md

We implemented an LTSA-based HT adapted predictor from the existing local LTSA source without downloading or modifying it. The shared ImageNet-initialized ResNet18 encoded every native ultrasound view; views were averaged within each observed examination. The original fixed sinusoidal temporal encoding, a one-layer four-head causal Transformer (512-dimensional ReLU feed-forward, post-norm, dropout0.25) and the latest real-history representation were used for a binary logit head. The survival and step-ahead heads were not used.

基于现有本地LTSA源码实现HT适配预测器，未下载或改动原源码。共享ImageNet初始化ResNet18编码全部原生超声view，并在真实检查内均值聚合。保留原固定sin/cos时间编码、1层4头因果Transformer（FFN512/ReLU/postnorm/dropout0.25），用末真实历史表示经二分类logit头预测；不使用生存或step-ahead头。

The frozen860Train and189Val patients and final-year0/1 labels were shared by both arms. Native histories included4,585/991 examinations and18,413/3,861 views in Train/Val. Only real dates before the target year were eligible. Relative times were binned by floor(days from first observed examination/30.4375), with no target-date gap. Target-year image/report contents and Official Test were not read.

两臂共享冻结860Train/189Val患者及最终年度0/1标签，Train/Val原生历史分别4,585/991检查和18,413/3,861view。仅目标年之前真实日期合格；相对时间为floor(距首真实检查天数/30.4375)，不注入目标日期gap，不读目标年影像/报告内容或Official Test。

Augmentation was fixed before formal validation:686 historical Original Progression CPFM-ON nodes (4,116saved images), anchored to the first observed real date and strictly bounded by first/last real-history examinations, were inserted as explicitly synthetic query tokens. Six views per node were averaged to one token.758target-year nodes and101unknown-future nodes after the latest real exam were excluded. All real exams and the same patient support were retained; Val was always REAL-only and the readout token remained the latest real exam.

增强在正式验证前冻结：686个历史Original Progression CPFM-ON节点（4,116张已存图）由首真实日期锚定且严格在首末真实历史之间，作为明确合成查询token插入。每节点六图均值为一个token；排除758个目标年节点、101个末真实之后未知将来节点。保留全部真实检查和相同患者支持；Val始终REAL-only，读出token始终末真实检查。

Both conditions used three fresh seeds42/3407/2026, FP32 AdamW (learning rate1e-4, weight decay1e-4), a maximum60epochs with5epoch warmup/cosine, patience12 and minimum15epochs, and effective patient batch8 (microbatch4/accumulation2). Selection prioritized Val ROC-AUC, AP, lower loss and earlier epoch. Every best checkpoint was independently strictly reloaded and the same189 ordered positive-class probabilities were verified to1e-6 tolerance, then averaged across seeds. Paired patient-level2,000resample percentile intervals evaluated OP-minus-REAL ROC-AUC/AP differences.

两条件使用42/3407/2026三个fresh seed，FP32 AdamW（lr1e-4/wd1e-4），最多60epoch、5epoch warmup/cosine、patience12/min15，有效患者batch8（微批4/累积2）。固定选模次序Val AUC、AP、较低loss、较早epoch。各best独立strict重载，同189人有序正类概率1e-6容差核验后种子平均；OP−REAL的AUC/AP差用患者配对2,000次百分位重抽样区间评价。

