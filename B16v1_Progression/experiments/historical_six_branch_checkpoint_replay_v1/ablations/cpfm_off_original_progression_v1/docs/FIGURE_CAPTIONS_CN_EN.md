Fig. 1 | Original Progression CPFM injection intervention. a The historical F16 and P16 computations retain their original encoders, masked pooling, fusion and adapter. b Original B14 magnitude, direction and patient-direction paths are frozen. Only the original enabled gate changes. c ON applies the original residual; OFF returns the original activation. Counts are physical calls per six-image batch (31 UNet calls and four layer hooks), not six independent count replicates. Progression is still computed but has no active UNet path in OFF. This identifies injection-path dependence, not independent module causality. No medical image is shown.

图1 | 原Original Progression的CPFM注入干预。a 原F16/P16编码、掩码均值池化、融合和adapter保持不变。b 原B14幅度、方向及患者方向路径冻结，只切换原enabled门。c ON应用原残差，OFF直接返回原激活。物理调用数以六图batch为单位（31次UNet、4层hook），不能再乘以6当作独立重复。OFF仍计算Progression，但无活跃UNet通路；衡量的是注入路径依赖，不是独立模块因果贡献。图中不含医学图像。


Fig. 2 | Historical internal-validation references and new CPFM OFF. Bars show three-seed ensemble ROC-AUC and average precision (AP) for 189 patients (58 positive, 131 negative). Grey bars retain the old six-branch background; only original ON (blue) and new OFF (orange) form the route-H controlled comparison. The old Plain-SD branch is not the new OFF intervention. No seed spread or confidence interval is implied by the bars. Val was used for checkpoint selection.

图2 | 历史内部验证参考与新增CPFM OFF。条形为189名患者（阳性58/阴性131）的三seed ensemble ROC-AUC与average precision（AP）。灰色为旧路线背景，只有原ON（蓝）与新OFF（橙）构成路线H受控比较。旧Plain-SD不是新增OFF。条形不代表seed离散度或置信区间；Val用于选模。


Fig. 3 | Paired ON-minus-OFF differences. Points show the observed ROC-AUC/AP differences. Whiskers are two-sided percentile 95% intervals from 2,000 stratified paired patient resamples (seed 20261012), preserving 58 positive and 131 negative draws. The same draw indexes both arms after averaging three model seeds within each patient. The dashed line is zero. These intervals describe reused internal Val, not independent clinical uncertainty; no multiplicity-adjusted significance is asserted.

图3 | 配对ON减OFF差异。点为ROC-AUC/AP实测差值，误差线为2000次分层配对患者bootstrap（seed20261012）的双侧percentile95%区间，每次保留阳性58/阴性131。先在每名患者内平均三个模型seed，再以相同索引抽取两臂。虚线为零。区间描述复用内部Val的不确定性，不是独立临床区间，也不宣称经过多重校正的显著性。


Fig. 4 | Numerical ON/OFF transmission differences. Top panels show full-support RGB-pixel and generated-image F16 RMSE for 9,270 image pairs across 1,545 nodes and 860 patients. Bottom panels show final-latent and early conditional-UNet-noise RMSE for the predeclared 36 full-sampling pilot pairs only. Images are technical observations; neither 9,270 images nor 36 pilot images are independent patient samples. Zero-response outputs were retained. Differences do not establish pathology or anatomical realism. Public source data contain binned aggregates only.

图4 | ON/OFF数值传递差异。上方是1545节点、860患者、9270对图的全量RGB像素与生成图F16 RMSE；下方仅为事前固定36对完整采样pilot的最终latent和early conditional UNet noise RMSE。图像为技术观察，9270或36都不是独立患者样本数。零响应输出保留。差异不证明病理或解剖真实性；公开源数据仅为分箱聚合。


Fig. 5 | Model-seed stability and ensemble threshold metrics. Left dots show three original ON model seeds and three newly trained OFF seeds, not additional patient replicates; no seed confidence interval is shown. Right bars show ensemble F1, sensitivity and specificity on the same 189 internal-Val patients. Each arm uses its own historical Youden-J threshold with the smallest-threshold tie rule; thresholds were not optimized for F1. Blue denotes ON and orange OFF.

图5 | 模型seed稳定性与ensemble阈值指标。左侧点为三个原ON模型seed和三个新训练OFF seed，不是额外患者重复，也未给seed置信区间。右侧条形为同189内部Val患者的ensemble F1、敏感度和特异度。每臂按原Youden J及最小阈值tie规则取自身阈值，没有另调阈值优化F1。蓝色ON、橙色OFF。
