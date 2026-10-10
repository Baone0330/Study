# 历史六分支结果分析草稿

## 重放覆盖与数值一致性

上一阶段完成了六分支各三个 selected-best CLIMATv2 checkpoint 的18次独立新推理。每个分支按同患者键匹配42、3407和2026三个种子的正类预测概率，并等权平均形成189人历史内部验证集的 ensemble；其中HT0为131人、HT1为58人。六分支七项展示字段（包含样本量）共42项均与历史参考值在六位小数精度一致。这项结果证明指定权重和计算路径可重放，不等于患者概率全部逐比特相等，也不证明独立临床泛化。本次封存阶段未新增任何模型前向。

## 主结果与完整对照

在固定历史内部 Val 中，Original Progression 的ROC-AUC为0.871808，AP为0.788794，在六种观察条件中最高。相对Full B16v1的0.836141和0.715742，差值分别为0.035667和0.073052；相对REAL-only的0.763885和0.624627，差值为0.107923和0.164167。这里AP由average_precision_score计算，不是PR曲线梯形积分。相关主结果见Table01、Fig05，全部差值由原始完整精度表自动导出。

Plain-SD历史路由的ROC-AUC为0.846802、AP为0.753492，均高于Full B16v1，差值为0.010661和0.037750。Original Progression相对Plain-SD的ROC-AUC和AP差值则为0.025007和0.035302。Plain-SD并非只移除CPFM：它同时关闭患者历史、纵向查询和B16专用条件，并使用通用prompt的SD+医学LoRA生成路线。因此该排序不支持对CPFM单独主效应作因果判断；该负向对照信息在表格和差值图中完整保留。

history_only达到ROC-AUC0.833377、AP0.717396，接近Full的ROC-AUC且AP略高；temporal_only达到0.806528和0.695574，两项均低于Full但高于REAL-only。这些比较描述既定训练图像输入和原条件路线的内部评估，不用排名倒推某个模块已经形成有效病理机制。

## 阈值相关的权衡

沿用各分支原有Youden阈值策略，Original Progression的Accuracy为0.830688、Sensitivity为0.793103、Specificity为0.847328、F1为0.741935。history_only的Sensitivity为0.810345，高于Original Progression，但Specificity为0.748092；temporal_only的Specificity为0.893130最高，而Sensitivity为0.637931。因而最高ROC-AUC/AP并不意味着每项阈值指标均占优。Precision与TN/FP/FN/TP直接取自上一阶段真实聚合表，未依据四舍五入指标反推，见Table01和Fig06。

## 统计定位

本轮没有新运行患者级配对bootstrap或统计检验，因此报告仅为三seed概率ensemble的点估计与描述性差值，不提供95%CI、p值或“显著提高”措辞。阈值来自原历史内部Val策略，并非新的外部锁定阈值。其他Clean/Strict协议的既有结果不参与这张历史六分支表，也不会被本轮封存或较高的旧内部Val指标覆盖。

陈述—证据：18次重放/42字段匹配由既有release及QA支持；六分支观察排序由Table01支持；全部比较由Table02支持；独立临床改善和单模块因果效果均未被证明。结构选择先说明重放可信范围，再呈现完整主结果和阈值权衡，最后限定统计含义。
