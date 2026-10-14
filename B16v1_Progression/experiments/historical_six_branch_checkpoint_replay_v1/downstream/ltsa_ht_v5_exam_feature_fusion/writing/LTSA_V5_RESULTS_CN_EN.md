# V5 Results / 结果

The REAL-only ensemble achieved ROC-AUC 0.859042, AP 0.810942, accuracy 0.788360, precision 0.621622, sensitivity 0.793103, specificity 0.786260, and F1 0.696970 at its internal-Val Youden threshold 0.292721 (TN/FP/FN/TP 103/28/12/46).

REAL-only集成实测ROC-AUC 0.859042、AP 0.810942、准确率 0.788360、精确率 0.621622、敏感度 0.793103、特异度 0.786260、F1 0.696970，内部Val Youden阈值为0.292721（TN/FP/FN/TP 103/28/12/46）。

The Original Progression fusion ensemble achieved ROC-AUC 0.870624, AP 0.802829, accuracy 0.783069, precision 0.613333, sensitivity 0.793103, specificity 0.778626, and F1 0.691729 at its internal-Val Youden threshold 0.303098 (TN/FP/FN/TP 102/29/12/46).

Original Progression fusion集成实测ROC-AUC 0.870624、AP 0.802829、准确率 0.783069、精确率 0.613333、敏感度 0.793103、特异度 0.778626、F1 0.691729，内部Val Youden阈值为0.303098（TN/FP/FN/TP 102/29/12/46）。

LTSA_V5_REAL_ONLY, seed 42: best epoch 19, 31 completed epochs, AUC 0.835088, AP 0.765454, F1 0.657143; independent strict reload passed.

LTSA_V5_REAL_ONLY，seed 42：最佳19轮，实际31轮，AUC 0.835088、AP 0.765454、F1 0.657143；独立strict重载通过。

LTSA_V5_REAL_ONLY, seed 3407: best epoch 21, 33 completed epochs, AUC 0.837984, AP 0.765637, F1 0.696296; independent strict reload passed.

LTSA_V5_REAL_ONLY，seed 3407：最佳21轮，实际33轮，AUC 0.837984、AP 0.765637、F1 0.696296；独立strict重载通过。

LTSA_V5_REAL_ONLY, seed 2026: best epoch 9, 21 completed epochs, AUC 0.845222, AP 0.780548, F1 0.695652; independent strict reload passed.

LTSA_V5_REAL_ONLY，seed 2026：最佳9轮，实际21轮，AUC 0.845222、AP 0.780548、F1 0.695652；独立strict重载通过。

LTSA_V5_ORIGINAL_PROGRESSION_FUSION, seed 42: best epoch 6, 18 completed epochs, AUC 0.840353, AP 0.765171, F1 0.672131; independent strict reload passed.

LTSA_V5_ORIGINAL_PROGRESSION_FUSION，seed 42：最佳6轮，实际18轮，AUC 0.840353、AP 0.765171、F1 0.672131；独立strict重载通过。

LTSA_V5_ORIGINAL_PROGRESSION_FUSION, seed 3407: best epoch 18, 30 completed epochs, AUC 0.845354, AP 0.733872, F1 0.700855; independent strict reload passed.

LTSA_V5_ORIGINAL_PROGRESSION_FUSION，seed 3407：最佳18轮，实际30轮，AUC 0.845354、AP 0.733872、F1 0.700855；独立strict重载通过。

LTSA_V5_ORIGINAL_PROGRESSION_FUSION, seed 2026: best epoch 4, 16 completed epochs, AUC 0.835483, AP 0.781191, F1 0.718447; independent strict reload passed.

LTSA_V5_ORIGINAL_PROGRESSION_FUSION，seed 2026：最佳4轮，实际16轮，AUC 0.835483、AP 0.781191、F1 0.718447；独立strict重载通过。

OP−REAL differences were 0.011582 for ROC-AUC (95%CI -0.036498 to 0.061370) and -0.008114 for AP (95%CI -0.068648 to 0.051396); 2000/2,000 paired draws were valid and 0 single-class draws were skipped.

OP−REAL ROC-AUC差为0.011582（95%CI -0.036498至0.061370），AP差为-0.008114（95%CI -0.068648至0.051396）；2,000次配对重采样中2000次有效，跳过0次单类样本。

The frozen V3/V4 ensembles were REAL AUC0.859042/AP0.810942 and virtual-query OP AUC0.790208/AP0.698502. Their OP−REAL differences were −0.068834/−0.112441. Comparison with V5 is descriptive because the fusion gate and encoding path also differ; private REAL probability diagnostics are summarized separately.

冻结V3/V4集成为REAL AUC0.859042/AP0.810942、虚拟query OP AUC0.790208/AP0.698502，差值−0.068834/−0.112441。与V5仅作描述比较，因为gate和编码路径也变化；真实对照的私有概率一致性另表聚合报告。

The V5 REAL-only probabilities matched the frozen V3 and V4 REAL-only probabilities across all three seeds and the ensemble: the maximum absolute difference across the eight aligned comparisons was 0. This was a numerical implementation diagnostic, not a performance gate or a reason to retune the fusion arm.

V5 REAL-only与冻结V3/V4的三个seed及集成逐人概率保持一致；八项对齐比较的最大绝对差为0。这只是数值实现诊断，不是性能Gate，也不据此重新调节融合臂。
