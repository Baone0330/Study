# Figure legends

## Fig01_LTSA_HT_Detailed_Architecture

Fig.1 | LTSA-HT adapted architecture. ResNet18 internal stages and residual basic-block flow, all-view grouping, fixed local LTSA temporal encoding, masked causal Transformer and last-real-token logit head are shown. Components were source-checked and GPU-smoke tested; neither original survival head nor step-ahead is used.

图1 | LTSA-HT适配结构。显示ResNet18内部stage/残差basic block、全部view聚合、原本地LTSA固定时间编码、掩码因果Transformer和末真实token分类头。结构经源码与GPU烟测核实；不使用原生存头或step-ahead。

## Fig02_Bounded_Synthetic_Token_Mapping

Fig.2 | Frozen augmentation eligibility. Synthetic annual queries were anchored to the first real-history date, not the actual target date. Only686 nodes lying strictly inside the observed history span were included;859 nodes were excluded. The597 augmented and263 nonaugmented Train patients remain the same860-person cohort.

图2 | 冻结增强资格。合成年查询由首真实历史日期锚定，不由目标实际日期回推。仅686节点严格位于已观察历史跨度内，排除859节点。有合成597人、无合成263人仍组成同860训练人群。

## Fig03_Two_Arm_Internal_Val_ROC_PR

Fig.3 | Two-arm internal-validation curves. Three independently reloaded seed probabilities were averaged for each of the same189patients (58positive/131negative). Dashed references show chance ROC and positive prevalence; AP is sklearn average precision, not trapezoidal area. This is not Official Test.

图3 | 双臂内部验证曲线。同189患者（58阳性/131阴性）的三个独立重载种子正类概率平均；虚线为随机ROC/阳性比例。AP为sklearn average precision，非梯形面积；不是官方Test。

## Fig04_Seed_And_Ensemble_Performance

Fig.4 | Seed and probability-ensemble performance. Filled points are the three independent downstream seeds42/3407/2026; open diamonds are average-probability ensembles, not average AUC. Seed variability is not a patient confidence interval. All use the same189 internal-Val patients.

图4 | 种子和概率集成表现。实心点为42/3407/2026三个独立下游种子，空心菱形为概率平均集成而非平均AUC。种子变异不是患者置信区间；均为同189内部Val患者。

## Fig05_OP_Minus_REAL_Paired_Intervals

Fig.5 | Paired augmentation contrasts. Intervals are95% percentile intervals from2,000 paired resamples of189 patients, seed20261013, conditional on the fixed selected ensembles. They do not include training/model-selection uncertainty; crossing0 does not demonstrate a significant gain. Two endpoints are exploratory.

图5 | 配对增强效应。固定已选集成条件下，同189患者、种子20261013、2,000次配对重抽样95%百分位区间；不包含训练/选模不确定性。跨0不证明显著增益；两终点为探索性。
