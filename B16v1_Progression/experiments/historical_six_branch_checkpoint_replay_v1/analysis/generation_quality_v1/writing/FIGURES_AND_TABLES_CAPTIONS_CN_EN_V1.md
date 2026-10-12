Offline quality evaluation uses frozen saved images; auxiliary Full and Plain routes are not isolated CPFM interventions. No new training, sampling or Test access.

固定已保存图像的离线质量评价；Full 和 Plain 仅为辅助路线，不是 CPFM 单变量干预。无新训练、采样或 Test 访问。


Distances use 859 common patients. KID error bars are Monte Carlo SD over50 subsets512, not patient confidence intervals. Balanced covariance is rank-limited (1,718 images vs2,048 features); FID CI was not run. Neither distance certifies thyroid anatomy.

距离使用 859 名共同患者。KID 误差线是 50 次、每次 512 图的蒙特卡洛标准差，不是患者置信区间。均衡集 1,718 图小于 2,048 特征维数，协方差秩受限；未计算 FID 区间。两指标均不认证甲状腺解剖真实性。


Distributions aggregate six images within node, then nodes within patient; n=860. ON/OFF shares node, image index and exact seed. SSIM/LPIPS describe structural/perceptual change, not reconstruction against target truth. All responses, including zero and extremes, are retained.

先对每节点六图求均值，再对每患者节点求均值；n=860。ON/OFF 共享节点、图序和精确种子。SSIM/LPIPS 表示结构/感知变化，不是相对目标真值的重建准确性。保留零响应及极端值。


Whole-image512 proxies show equal-patient means with2000 patient-bootstrap95% percentile intervals. Native view counts and dimensions differ; resampling and acquisition may affect values. More contrast or high-frequency power is not necessarily more pathological detail.

全图 512 尺度代理显示患者等权均值与 2,000 次患者 bootstrap 的 95% 百分位区间。原始 view 数和尺寸不同，重采样及采集可能影响数值。更多对比度或高频功率不必然表示更好的病理细节。


All15 unordered pairs of six seeds per node were evaluated:23,175 seed pairs per arm,1,545 nodes,860 patients. Error bars use2000 patient-cluster bootstrap; these technical pairs are not independent clinical subjects.

每节点六种子的全部 15 个无序配对均被评价：每臂 23,175 对、1,545 节点、860 患者。误差线来自 2,000 次患者聚类 bootstrap；技术配对不是独立临床受试者。


Each generated arm has9270 images/1545 nodes/860 patients; REAL has13200 unique native views/3154 exams/859 patients. One generated patient lacks a native REAL slot reference. Primary distribution comparisons use common859, not an silently altered pair denominator.

各生成臂为 9,270 图/1,545 节点/860 患者；REAL 为 13,200 个原始 view/3,154 Exam/859 患者。有 1 名生成患者缺少原始 REAL slot 参考。主分布比较用共同 859 人，不暗改配对分母。


Flags are prespecified whole-image statistical/line candidates, not thyroid diagnoses. Bars are equal-patient mean fractions. Hough line/cross candidates can detect genuine interfaces or acquisition UI; clinical specificity is unknown. None were removed.

标记是预声明的全图统计/直线候选，不是甲状腺诊断。柱高为患者等权标记比例。Hough 直线/十字可能检出真实界面或采集 UI，临床特异性未知；没有据此剔图。


Measured statistical and technical evidence is separated from absent clinical or target-truth evidence. Offline quality, structural visibility and downstream HT utility are different questions.

已测量的统计/技术证据与缺失的临床/目标真值证据分开。离线质量、结构可视性与下游 HT 效用是不同问题。


Table01. Branch image/patient/node support. Values originate from measured aggregate CSVs; no patient identities or clinical dates are included.

表01。对应聚合表呈现分支支持。数值来自实际聚合 CSV，不含患者身份或临床日期。


Table02. Native real reference and preprocessing. Values originate from measured aggregate CSVs; no patient identities or clinical dates are included.

表02。对应聚合表呈现真实参考及预处理。数值来自实际聚合 CSV，不含患者身份或临床日期。


Table03. All feature-distance comparisons and limitations. Values originate from measured aggregate CSVs; no patient identities or clinical dates are included.

表03。对应聚合表呈现分布距离及限制。数值来自实际聚合 CSV，不含患者身份或临床日期。


Table04. Exact ON/OFF and Full/ON paired changes. Values originate from measured aggregate CSVs; no patient identities or clinical dates are included.

表04。对应聚合表呈现严格配对变化。数值来自实际聚合 CSV，不含患者身份或临床日期。


Table05. Whole-image technical proxies. Values originate from measured aggregate CSVs; no patient identities or clinical dates are included.

表05。对应聚合表呈现全图技术代理。数值来自实际聚合 CSV，不含患者身份或临床日期。


Table06. Six-seed node diversity. Values originate from measured aggregate CSVs; no patient identities or clinical dates are included.

表06。对应聚合表呈现六种子多样性。数值来自实际聚合 CSV，不含患者身份或临床日期。


Table07. Prespecified artifact proxy flags and absent medical review. Values originate from measured aggregate CSVs; no patient identities or clinical dates are included.

表07。对应聚合表呈现伪影代理及评阅范围。数值来自实际聚合 CSV，不含患者身份或临床日期。


Table08. Claim-to-evidence boundaries. Values originate from measured aggregate CSVs; no patient identities or clinical dates are included.

表08。对应聚合表呈现结论—证据边界。数值来自实际聚合 CSV，不含患者身份或临床日期。


Table09. Paired patient KID uncertainty. Values originate from measured aggregate CSVs; no patient identities or clinical dates are included.

表09。对应聚合表呈现配对患者 KID 不确定性。数值来自实际聚合 CSV，不含患者身份或临床日期。


Table10. Unavailable or unperformed metrics. Values originate from measured aggregate CSVs; no patient identities or clinical dates are included.

表10。对应聚合表呈现不可用/未开展指标。数值来自实际聚合 CSV，不含患者身份或临床日期。


Table11. Native image sizes and modes. Values originate from measured aggregate CSVs; no patient identities or clinical dates are included.

表11。对应聚合表呈现原始尺寸与 mode。数值来自实际聚合 CSV，不含患者身份或临床日期。


Table12. Exact pixel duplicate groups. Values originate from measured aggregate CSVs; no patient identities or clinical dates are included.

表12。对应聚合表呈现像素重复等价组。数值来自实际聚合 CSV，不含患者身份或临床日期。
