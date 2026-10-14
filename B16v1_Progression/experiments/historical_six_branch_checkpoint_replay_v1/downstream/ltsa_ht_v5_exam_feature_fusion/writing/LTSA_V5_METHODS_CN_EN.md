# V5 Methods / 方法

The two V5 arms used the same 860 training patients (617 HT-negative and 243 HT-positive) and 189 internal-validation patients (131 negative and 58 positive), with all 4,585/991 pre-target REAL examinations and 18,413/3,861 native views. The 686 previously frozen Original Progression nodes (4,116 images; historical B14 CPFM ON) were retained without reselection or new generation. Validation used REAL history only; Official Test content was not accessed.

V5两臂使用相同860名训练患者（617阴性、243阳性）及189名内部验证患者（131阴性、58阳性），完整保留4,585/991次目标年度前真实检查和18,413/3,861张原生视图。既有686个Original Progression节点（4,116图，历史B14 CPFM ON）未重选或重生成；验证仅使用真实历史，未访问Official Test内容。

Before reading pixels, each frozen query was attached to the first strictly later REAL examination within the historical window. A shared trainable offline-ImageNet ResNet18 encoded every REAL view and six images per node. REAL views were averaged per Exam; synthetic views were averaged per node and then equally across nodes attached to the same Exam. No synthetic temporal token was added.

读取像素前，每个冻结query被附着到历史窗口内第一个严格晚于其日期的真实检查。共享、可训练、离线ImageNet初始化的ResNet18编码全部真实视图及每节点六图；真实视图检查内均值，合成图先节点内均值再对同Exam节点等权均值，不增加合成时间token。

For attached Exams, two separate LayerNorm512 modules formed h_r and h_s. A scalar sigmoid gate from concatenated 1,024-D features modulated a 512→512 projection of h_s−h_r, with fixed residual scale 0.1. Unattached Exams bypassed all fusion operations exactly. Both arms contained 13,020,738 trainable parameters; new-module initialization preserved global Python, NumPy and Torch CPU/CUDA RNG. Standard train BatchNorm was retained, so additional synthetic images could alter its running statistics.

有节点的Exam通过两个独立LayerNorm512形成h_r/h_s；拼接1024维输入的标量sigmoid门控调制h_s−h_r的512→512投影，残差系数固定0.1。无节点Exam严格旁路全部融合操作。两臂均有13,020,738可训练参数，新增模块初始化保持Python、NumPy、Torch CPU/CUDA随机状态；训练BN保持标准行为，新增合成图可能改变其运行统计。

The original month sinusoidal positional encoding, one 4-head causal Transformer layer (512-D FFN, ReLU, dropout 0.25) and last-REAL binary head were retained. Each arm was independently trained at seeds 42, 3407 and 2026 using FP32 AdamW (learning rate and weight decay 10⁻⁴), 5-epoch warmup and cosine decay to epoch60, microbatch4/accumulation2, and complete differentiable view chunks16. Early stopping used max60/min15/patience12; best selection followed ROC-AUC, AP, lower loss and earlier epoch. No V3/V4 or previous-arm checkpoint initialized a run.

保留原始月sinusoidal位置编码、单层四头因果Transformer（FFN512、ReLU、dropout0.25）及最后真实token二分类头。每臂分别以42、3407、2026独立训练，采用FP32 AdamW（学习率和weight decay均10⁻⁴）、5轮warmup后cosine至60轮、microbatch4/累积2、完整可微view chunk16。早停max60/min15/patience12，最佳顺序为AUC、AP、较低loss、较早epoch；未用V3/V4或前一臂权重初始化。

Cached RGB224 pixels were ImageNet-normalized without new stochastic image augmentation. Training minimized unweighted binary cross-entropy with logits once per patient. Microbatch 4 with two accumulation steps produced an effective batch of 8; the final short accumulation group was normalized by its actual patient count. FP32, TF32-off, deterministic algorithms and math-only scaled-dot-product attention followed the frozen V3 numerical contract.

已有RGB224缓存按ImageNet规范归一化，不增加随机影像增强。每患者一次无类别权重的BCEWithLogitsLoss监督。microbatch 4、累积2步对应有效batch 8；末尾不足8人的累积组按实际患者数归一化。FP32、关闭TF32、确定性算法及仅math SDPA均沿用冻结V3数值合同。

Every best checkpoint was independently strictly reloaded within a 10⁻⁶ probability tolerance on the same ordered 189 patients. Ensembles averaged three positive-class probability vectors. AP denotes average_precision_score, not trapezoidal PR area. Youden-J internal-Val thresholds and fixed0.5 sensitivity analyses were reported. Patient-paired bootstrap used 2,000 draws at seed20261013 with single-class skips recorded.

每个best checkpoint均独立strict重载，同189人顺序的概率容差10⁻⁶；集成为三种子正类概率均值。AP为average_precision_score而非梯形PR面积。报告内部Val Youden-J阈值及固定0.5敏感性分析；患者配对bootstrap为固定seed20261013的2,000次，并记录单类跳过数。
