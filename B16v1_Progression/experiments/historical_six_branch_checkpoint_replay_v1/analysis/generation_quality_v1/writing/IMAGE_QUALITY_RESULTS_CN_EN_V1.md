# Measured results / 实测结果



All50,280 frozen images decoded successfully;101 images beyond the first member of exact-pixel equivalence groups were retained. Feature extraction,9,270 primary changes,9,270 Full/ON auxiliary changes and46,350 within-node seed-pair comparisons were completed. Zero-response primary pairs: 0.

全部 50,280 张冻结图像成功解码；像素等价组中首张以外的 101 张重复图保留。已完成特征提取、9,270 对主变化、9,270 对 Full/ON 辅助路线变化和 46,350 对节点内种子比较。主配对零响应为 0 对。

Under all-available common support, ON FID=70.3154 and OFF FID=118.77; OFF−ON=48.4542. KID MC means were 0.0597322 and 0.145596, respectively (difference=0.0858637). These are descriptive distances, without a computed FID confidence interval.

在全量共同患者支持下，ON FID=70.3154、OFF FID=118.77，OFF−ON=48.4542；KID 蒙特卡洛均值分别为 0.0597322 和 0.145596，差值=0.0858637。这是描述性距离；未计算 FID 置信区间。

Balanced full-U KID OFF−ON=0.0806263, paired patient95% interval [0.0754977,0.0855833]. This estimand differs from the Monte Carlo subset mean and remains conditional on the frozen balanced reference. Pixel-change image means: MAE=30.403, RMSE=41.2234, gray SSIM=0.67143, LPIPS=0.385836.

均衡 full-U KID OFF−ON=0.0806263，配对患者 95% 区间 [0.0754977,0.0855833]。该估计目标不同于蒙特卡洛子集均值，且以冻结均衡参考为条件。逐图变化均值：MAE=30.403、RMSE=41.2234、灰度 SSIM=0.67143、LPIPS=0.385836。

Every branch and sensitivity result, uncertainty type, denominator and low-level proxy is retained in Tables01–12, including nonzero flags, duplicates, rank limitations and unavailable metrics. No quality threshold was used to alter the cohort or choose the visual examples.

表 01–12 保留全部分支、敏感性结果、不确定性类型、分母和低层代理，包括非零标记、重复图、秩限制和不可用指标。没有利用质量阈值改变群体或选择图例。

ON had a lower natural-image distribution distance than OFF, yet the prespecified long-line candidate fraction was 0.915019 for ON, 0.625388 for OFF, and 0.538115 for REAL under equal-patient aggregation. Thus the different evidence layers do not warrant a blanket statement that ON is clinically superior; line candidates have unknown medical specificity.

ON 的自然图像分布距离较 OFF 小，但预定义长直线候选的患者等权比例为 ON=0.915019、OFF=0.625388、REAL=0.538115。不同证据层次不支持笼统声称 ON 临床更优；直线候选的医学特异性未知。

| scope | branch | fid | kid_mc_mean | kid_mc_sd | real_images | generated_images | generated_patients | covariance_rank_limited |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_available_common_support | OP_CPFM_ON | 70.31544 | 0.05973215 | 0.002369102 | 13200 | 9258 | 859 | False |
| all_available_common_support | OP_CPFM_OFF | 118.7696 | 0.1455958 | 0.004967753 | 13200 | 9258 | 859 | False |
| all_available_common_support | FULL_B16V1_HIST | 71.3605 | 0.06139092 | 0.002552177 | 13200 | 9258 | 859 | False |
| all_available_common_support | PLAIN_SD_HIST | 118.6576 | 0.14542 | 0.004981323 | 13200 | 9258 | 859 | False |
| patient_balanced_common_support | OP_CPFM_ON | 85.80477 | 0.05942833 | 0.001817194 | 1718 | 1718 | 859 | True |
| patient_balanced_common_support | OP_CPFM_OFF | 129.7719 | 0.1401125 | 0.004619047 | 1718 | 1718 | 859 | True |
| patient_balanced_common_support | FULL_B16V1_HIST | 86.61765 | 0.06030762 | 0.002280919 | 1718 | 1718 | 859 | True |
| patient_balanced_common_support | PLAIN_SD_HIST | 129.5945 | 0.1397679 | 0.004622227 | 1718 | 1718 | 859 | True |
| all_available_native_support_sensitivity | OP_CPFM_ON | 70.31761 | 0.05950324 | 0.002093467 | 13200 | 9270 | 860 | False |
| all_available_native_support_sensitivity | OP_CPFM_OFF | 118.786 | 0.1460658 | 0.004113387 | 13200 | 9270 | 860 | False |

| point_estimate | ci_low | ci_high | patients | images_per_patient | repeats |
| --- | --- | --- | --- | --- | --- |
| 0.08062634 | 0.0754977 | 0.08558327 | 859 | 2 | 2000 |
