# V5 captions / 图表注

Fig.1 | V5 matched internal-validation ROC and precision–recall. REAL-only blue solid lines and OP fusion amber dashed lines represent three-seed probability ensembles on n=189 patients (58 positive). AP is average precision, not trapezoidal PR area. Dashed gray references show chance ROC and positive prevalence for PR. Source data are provided as aggregate curve coordinates; no patient-level data are released.

图1 | V5匹配内部验证ROC与PR曲线。蓝色实线REAL及琥珀色虚线OP代表189人（58阳性）的三种子概率集成；AP为average precision而非梯形PR面积。灰色虚线表示随机ROC及PR阳性率基线。仅提供聚合曲线坐标作为源数据，不发布逐人数据。

Fig.2 | Descriptive cross-protocol ensemble comparison. Six rows compare V3/V4 virtual-query augmentation with V5 Exam fusion and their REAL controls. Bars are measured ensemble AUC/AP, not independent patient replicates or a causal token-count contrast.

图2 | 跨协议集成描述性比较。六行比较V3/V4虚拟query增强与V5检查级融合及其REAL对照；柱为集成AUC/AP，不是独立患者重复，也不是token数量因果对照。

Fig.3 | All V5 seeds and ensembles. Seeds42/3407/2026 are independent training initializations on the same split; no seed is omitted. Ensemble is arithmetic mean of probabilities, not mean of the plotted metrics.

图3 | V5全部seed与集成。42/3407/2026是相同拆分上独立训练初始化，不省略任何seed；集成是概率算术均值，不是图中指标的均值。

Fig.4 | Internal architecture of next-real Exam fusion. Anonymous REAL timeline and frozen query binding precede shared image encoding, node-equal averaging, separate LayerNorms, scalar sigmoid gate, 512-D difference projection and fixed0.1 residual. Empty synthetic masks follow strict identity. Original temporal masks and last-REAL readout are retained; no medical images or patient dates are depicted.

图4 | next-real检查融合内部结构。匿名时间线/query映射之后为共享编码、节点等权均值、独立LayerNorm、标量sigmoid gate、512维差值projection及固定0.1残差；无合成掩码走严格恒等支路，保留原时间掩码/最后REAL读出；无医学图像或患者日期。

Tables | Seed-level and ensemble metrics, paired bootstrap and frozen input support. Threshold-related metrics use internal-Val Youden-J or explicitly labeled fixed0.5 thresholds. Confidence intervals use 2,000 patient-paired resamples, fixed seed20261013 and reported single-class skips; no multiplicity-adjusted universal inference is made.

表 | 每seed/集成指标、配对bootstrap及冻结输入支持。阈值指标用内部Val Youden-J或明确标记的0.5；CI为固定seed20261013的2,000次患者配对重采样并报告单类跳过，不作多重校正后的普遍推断。
