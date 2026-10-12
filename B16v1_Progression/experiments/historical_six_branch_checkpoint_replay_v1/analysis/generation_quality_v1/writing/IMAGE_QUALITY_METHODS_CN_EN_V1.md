# Image quality methods / 图像质量方法



We evaluated saved historical Original Progression–CPFM ON/OFF images, with 9,270 exact node/index/seed pairs from 1,545 nodes and 860 patients. Original Progression refers to the frozen20260921 epoch120 historical route, not Clean/S2. Full and generic-prompt Plain SD were image/seed-aligned auxiliary routes, not single-variable CPFM contrasts.

评价已保存的历史 Original Progression–CPFM ON/OFF 图像：9,270 对精确节点/图序/种子配对，来自 1,545 节点和 860 名患者。Original Progression 指冻结的 20260921 epoch120 历史路线，不是 Clean/S2。Full 与通用提示词 Plain SD 为图像/种子对齐的辅助路线，不是 CPFM 单变量对照。

The reference contained 13,200 unique native historical views from 3,154 exams and 859 patients, after identity-based removal of repeated LOCF slot references. These images were not query-year ground truth. Distribution comparisons primarily used 859 common patients. The prespecified balanced sensitivity selected2 views per patient without quality/outcome filtering.

REAL 参考为 13,200 张独立原始历史 view，来自 3,154 Exam 和 859 名患者；按身份去除重复 LOCF slot 引用，未剔除像素重复图。这些不是 query 年真值。主分布比较限于 859 名共同患者；事前冻结的均衡敏感性分析每人选 2 图，不按质量/结局筛选。

Distribution features were extracted in frozen FP32 evaluation using torch-fidelity0.4.0 inception-v3-compat2048 with official evaluation weights, native RGB dimension-grouped batches16, internal TF1-compatible299 bilinear resize without antialias and (uint8−128)/128. FID used float64 sample covariance and the PSD Bures formula. KID was unbiased cubic polynomial MMD squared (gamma1/2048, intercept1),50 fixed-seed subsets512; negative estimates were retained. [Implementation](https://torch-fidelity.readthedocs.io/en/latest/).

分布特征采用冻结 FP32 评价：torch-fidelity0.4.0 的 inception-v3-compat2048 和官方评价权重；原始 RGB 按尺寸分组、batch16，内部 TF1 兼容 299 双线性 resize，无 antialias，值域为 (uint8−128)/128。FID 使用 float64 样本协方差和 PSD Bures 公式；KID 是无偏三次多项式 MMD²（gamma1/2048、常数1），固定种子 50 次子集、每次 512 图，保留负估计。[实现来源](https://torch-fidelity.readthedocs.io/en/latest/)。

ON/OFF native RGB512 change used MAE, RMSE and bit-exact equality; gray SSIM used BT.601 float64/data_range255/Gaussian sigma1.5/window11/population covariance. Calibrated AlexNet LPIPSv0.1 used RGB[-1,1], not clinical truth. Cached normalized-layer calculations matched the official forward on anonymous toy data. [LPIPS source](https://github.com/richzhang/PerceptualSimilarity).

ON/OFF 原始 RGB512 变化采用 MAE、RMSE 和逐像素恒等；灰度 SSIM 使用 BT.601 float64/data_range255/高斯 sigma1.5/窗口11/总体协方差。AlexNet LPIPSv0.1 使用 RGB[-1,1]，不以其代表临床真值；缓存标准化层计算已与官方前向在匿名数据上核验一致。[LPIPS 来源](https://github.com/richzhang/PerceptualSimilarity)。

Texture proxies used whole-image512 PIL bilinear resize,32×32 nonoverlapping block contrast,3×3 Sobel/Laplacian, demeaned Hann-window FFT with DC removed and radial high band≥0.25 cycles/pixel, black<5/white>250 and fixed line/cross candidates. Native sizes and modes were retained. None is an anatomical ROI or composite medical realism score.

纹理代理采用全图 512 PIL 双线性 resize、32×32 无重叠局部对比度、3×3 Sobel/Laplacian、去均值 Hann 窗 FFT（去 DC，径向高频≥0.25 周期/像素）、灰度黑<5/白>250及固定直线/十字候选。保留原始尺寸和 mode；这些不是解剖 ROI，也未合成为医学真实度总分。

Image-to-node-to-patient means used equal patient weights; native REAL used view-to-exam-to-patient means. Patient-cluster percentile bootstrap used2000 replicates/seed20261012. The balanced full-U KID OFF-minus-ON contrast resampled the same patient weights in REAL/ON/OFF. KID subset Monte Carlo SD is not a patient CI. FID CI was prespecified NOT_RUN; balanced1718-image covariance is rank-limited relative to2048 features.

采用 image→node→patient 均值和患者等权；原始 REAL 采用 view→Exam→patient 均值。患者聚类百分位 bootstrap 为 2,000 次/seed20261012。均衡 full-U KID 的 OFF−ON 差异对 REAL/ON/OFF 同时重采样同一患者权重。KID 子集蒙特卡洛标准差不是患者区间；FID CI 在测量前声明未做。均衡 1,718 图相对 2,048 特征维数存在协方差秩限制。
