# 历史 Original Progression–CPFM 生成图像质量评价结果 V1

## 1. 身份与状态

任务：B16V1_HISTORICAL_ORIGINAL_PROGRESSION_GENERATION_QUALITY_V1。机器：5090-lin / RTX5090。更新时间：2026-10-12T11:11:55.034334+00:00。实验状态：`QUALITY_EVALUATION_COMPLETED_WITH_LIMITATIONS`；公开包的 Git 标识以仓库提交历史为准。本报告仅使用实际完成值，不设置质量性能成功阈值。

## 2. 输入根及模型身份

只读主源：`<frozen original ON/OFF source root>`；独立输出：`<private output root>`。采用历史 Original Progression 20260921 epoch120、旧冻结 F16/B14 及历史 ON；新 OFF 为同种子 CPFM 零注入，不是 Plain SD，也不是 Clean/Strict/S2。历史采样为 PNDM30、CFG7、RGB512、medical LoRA。本轮没有重新采样旧 ON、OFF 或辅助图。前轮只有36对固定历史ON重采样通过解码像素严格恒等，不能改称全部9,270张ON曾被重新采样核验。

## 3. 全量计数

| branch | images | patients | nodes | native_exams | balanced_images | role |
| --- | --- | --- | --- | --- | --- | --- |
| OP_CPFM_ON | 9270 | 860 | 1545 | 0 | 1718 | PRIMARY_PAIRED |
| OP_CPFM_OFF | 9270 | 860 | 1545 | 0 | 1718 | PRIMARY_PAIRED |
| REAL_HISTORY_REFERENCE | 13200 | 859 | 0 | 3154 | 1718 | REFERENCE |
| FULL_B16V1_HIST | 9270 | 860 | 1545 | 0 | 1718 | AUXILIARY_ROUTE_ONLY |
| PLAIN_SD_HIST | 9270 | 860 | 1545 | 0 | 1718 | AUXILIARY_ROUTE_ONLY |

ON/OFF 各 9,270 张，精确主配对 9,270 对；1,545 节点×每节点6图；860 名唯一患者。五个图像集合共 50,280 张均已解码，失败0。唯一 seed 以每图 master_seed 配对核验，不把技术种子计作独立患者。

## 4. 匹配范围及失败/缺失

患者、节点、relative slot、image index、确切 seed 和现存 PNG 已逐一核验。主配对缺失0、错误0。Full/Plain 各 9,270 图与主清单具有同患者、同节点、同图序、同种子；但路线和条件不同，不能据此声称单模块贡献。主图像配对分母始终保持 860 人/9,270 对。

## 5. REAL 参考的合法选择

仅选 K6 source_type=REAL 且在冻结原始历史中有精确患者/原始日期/路径身份的 native views；剔除重复 LOCF 引用，不按疾病结局、报告或图像分数筛选。参考为 859 人、3,154 Exam、13,200 唯一原始 view；有1名生成患者没有原始 REAL slot 支持。参考不是 query 年真实目标，主分布比较使用共同 859 人；另完整报告 all-available native-support 敏感性。均衡方案测量前冻结为每人2图，共1,718图/分支，ON/OFF 选同一批配对。

## 6. 辅助组身份

Full：`EXACT_IMAGE_SEED_MATCH`，从历史 full manifest 中匹配 9,270/12,036 图；属于历史完整路线，不是 Original Progression 单变量对照。Plain：`EXACT_IMAGE_SEED_ALIGNED_AUXILIARY_GENERIC_PROMPT`，9,270 图通用提示词、相同节点与种子；历史 manifest/config/release 显示 patient_history_used=0、b16_specific_condition_used=0、Train-only/Test0。两辅助组只解释路线背景，不替代本轮 OFF。

## 7. 预处理、软件与数值自检

全图技术统计：PIL bilinear 512×512，保留 native尺寸/mode；无对比度均衡或医学 ROI。特征：torch-fidelity0.4.0 标准 Inception2048/官方权重、原 RGB 按尺寸分组、内部 TF1兼容299/无 antialias/(uint8−128)/128；FP32/TF32关闭。Torch `2.11.0+cu128`，运行时 CUDA `12.8`，不是把 nvidia-smi 的驱动最大 CUDA 当作训练 runtime。LPIPS0.1.4/AlexNetv0.1 冻结权重与校准；SSIM0.25.2/BT.601gray/data_range255/高斯1.5/window11/总体协方差。五类匿名测试通过；FID/KID 排列不变、均值平移、自比与无偏负 KID、bootstrap核聚合等式均已验证。LPIPS缓存/官方前向 toy最大误差0；匿名同输入 Inception 微小 FP32舍入差约3.22e−6，不影响或冒充旧 ON 的解码像素恒等要求。

## 8. FID/KID 实测完整矩阵

| scope | branch | fid | kid_mc_mean | kid_mc_sd | real_images | generated_images | real_patients | generated_patients | covariance_rank_limited |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_available_common_support | OP_CPFM_ON | 70.31544 | 0.05973215 | 0.002369102 | 13200 | 9258 | 859 | 859 | False |
| all_available_common_support | OP_CPFM_OFF | 118.7696 | 0.1455958 | 0.004967753 | 13200 | 9258 | 859 | 859 | False |
| all_available_common_support | FULL_B16V1_HIST | 71.3605 | 0.06139092 | 0.002552177 | 13200 | 9258 | 859 | 859 | False |
| all_available_common_support | PLAIN_SD_HIST | 118.6576 | 0.14542 | 0.004981323 | 13200 | 9258 | 859 | 859 | False |
| patient_balanced_common_support | OP_CPFM_ON | 85.80477 | 0.05942833 | 0.001817194 | 1718 | 1718 | 859 | 859 | True |
| patient_balanced_common_support | OP_CPFM_OFF | 129.7719 | 0.1401125 | 0.004619047 | 1718 | 1718 | 859 | 859 | True |
| patient_balanced_common_support | FULL_B16V1_HIST | 86.61765 | 0.06030762 | 0.002280919 | 1718 | 1718 | 859 | 859 | True |
| patient_balanced_common_support | PLAIN_SD_HIST | 129.5945 | 0.1397679 | 0.004622227 | 1718 | 1718 | 859 | 859 | True |
| all_available_native_support_sensitivity | OP_CPFM_ON | 70.31761 | 0.05950324 | 0.002093467 | 13200 | 9270 | 859 | 860 | False |
| all_available_native_support_sensitivity | OP_CPFM_OFF | 118.786 | 0.1460658 | 0.004113387 | 13200 | 9270 | 859 | 860 | False |

KID 为原始 MMD²尺度，无乘1000或截断。MC均值/SD来自50×512子集；SD不是患者CI。全量共同支持 OFF−ON：FID=48.4542、KID_MC=0.08586367。均衡 full-U KID 差异及患者区间：

| point_estimate | ci_low | ci_high | bootstrap_sd | patients | images_per_patient | repeats |
| --- | --- | --- | --- | --- | --- | --- |
| 0.08062634 | 0.0754977 | 0.08558327 | 0.002539462 | 859 | 2 | 2000 |

2000次同患者配对重采样，full-U目标不同于MC子集均值。均衡协方差秩受限（1,718图<2,048维）；数值有效不等于稳定临床结论。FID CI按测量前资源声明未计算，不将距离大小直接称为临床真实性或自动声称 FID 显著。

## 9. 严格图像变化与零响应

| comparison | metric | images | nodes | patients | image_mean | image_median | image_sd | image_q25 | image_q75 | image_p025 | image_p975 | patient_equal_mean | patient_ci_low | patient_ci_high | zero_response_images |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ON_OFF | rgb_mae | 9270 | 1545 | 860 | 30.40304 | 29.71417 | 6.44669 | 25.84069 | 34.29245 | 19.63259 | 44.86608 | 30.45698 | 30.31687 | 30.59499 | 0 |
| ON_OFF | rgb_rmse | 9270 | 1545 | 860 | 41.22342 | 40.63669 | 7.98251 | 35.65295 | 46.21188 | 27.28825 | 58.31079 | 41.26953 | 41.09697 | 41.44054 | 0 |
| ON_OFF | gray_ssim | 9270 | 1545 | 860 | 0.6714302 | 0.6768223 | 0.07417055 | 0.6236435 | 0.724509 | 0.5129139 | 0.8021094 | 0.6711032 | 0.6695186 | 0.6726313 | 0 |
| ON_OFF | lpips | 9270 | 1545 | 860 | 0.3858364 | 0.3718632 | 0.08969674 | 0.3225239 | 0.4334151 | 0.2471721 | 0.6038048 | 0.3864652 | 0.3843952 | 0.3884691 | 0 |
| ON_OFF | bit_exact | 9270 | 1545 | 860 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| ON_FULL_AUX | rgb_mae | 9270 | 1545 | 860 | 15.92673 | 14.895 | 5.526963 | 11.97379 | 18.78504 | 8.161649 | 29.25546 | 15.88729 | 15.75027 | 16.02798 | 0 |
| ON_FULL_AUX | rgb_rmse | 9270 | 1545 | 860 | 22.61494 | 21.45858 | 7.131463 | 17.46201 | 26.47237 | 12.03248 | 39.49399 | 22.54523 | 22.36292 | 22.72968 | 0 |
| ON_FULL_AUX | gray_ssim | 9270 | 1545 | 860 | 0.8333951 | 0.854368 | 0.08630716 | 0.7936798 | 0.8949482 | 0.6080801 | 0.9410076 | 0.8335831 | 0.8314968 | 0.8355795 | 0 |
| ON_FULL_AUX | lpips | 9270 | 1545 | 860 | 0.1986839 | 0.1740735 | 0.09993941 | 0.1289903 | 0.2410062 | 0.07499707 | 0.4572849 | 0.1984998 | 0.1962292 | 0.2008963 | 0 |
| ON_FULL_AUX | bit_exact | 9270 | 1545 | 860 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

SSIM越低/LPIPS越高仅表示 ON/OFF 变化更大。两臂均不是ground truth；Full/ON只为辅助路线差异。零响应按全 native RGB array_equal，不用文件哈希。

## 10. 技术代理与可疑结构标记

技术 CSV：`metrics/TEXTURE_AND_ARTIFACT_DESCRIPTIVES_V1.csv`；含灰度mean/std/p1/p99、动态范围、32×32局部std、Sobel、Laplacian、Hann/DC排除FFT高频、黑白/边框/UI边缘比例。以下列出主要患者等权统计；所有15项完整精度留在表05。

| branch | metric | images | nodes | exams | patients | patient_equal_mean | patient_ci_low | patient_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OP_CPFM_ON | gray_std | 9270 | 1545 | 0 | 860 | 57.60777 | 57.36946 | 57.83822 |
| OP_CPFM_ON | sobel_gradient_mean | 9270 | 1545 | 0 | 860 | 67.06293 | 66.8066 | 67.32528 |
| OP_CPFM_ON | high_frequency_fraction | 9270 | 1545 | 0 | 860 | 0.006129124 | 0.005964648 | 0.006312454 |
| OP_CPFM_ON | border_black_fraction | 9270 | 1545 | 0 | 860 | 0.00395812 | 0.003728516 | 0.004196011 |
| OP_CPFM_OFF | gray_std | 9270 | 1545 | 0 | 860 | 58.44371 | 58.25482 | 58.62401 |
| OP_CPFM_OFF | sobel_gradient_mean | 9270 | 1545 | 0 | 860 | 51.14462 | 50.95201 | 51.34557 |
| OP_CPFM_OFF | high_frequency_fraction | 9270 | 1545 | 0 | 860 | 0.00106563 | 0.001046036 | 0.001088976 |
| OP_CPFM_OFF | border_black_fraction | 9270 | 1545 | 0 | 860 | 0.04056954 | 0.03940379 | 0.04171437 |
| REAL_HISTORY_REFERENCE | gray_std | 13200 | 0 | 3154 | 859 | 41.36985 | 40.94321 | 41.81003 |
| REAL_HISTORY_REFERENCE | sobel_gradient_mean | 13200 | 0 | 3154 | 859 | 37.39351 | 36.86115 | 37.94328 |
| REAL_HISTORY_REFERENCE | high_frequency_fraction | 13200 | 0 | 3154 | 859 | 0.002245604 | 0.002125921 | 0.002376693 |
| REAL_HISTORY_REFERENCE | border_black_fraction | 13200 | 0 | 3154 | 859 | 0.1138821 | 0.1074371 | 0.1202303 |
| FULL_B16V1_HIST | gray_std | 9270 | 1545 | 0 | 860 | 50.42555 | 50.14114 | 50.71051 |
| FULL_B16V1_HIST | sobel_gradient_mean | 9270 | 1545 | 0 | 860 | 61.86244 | 61.56496 | 62.16566 |
| FULL_B16V1_HIST | high_frequency_fraction | 9270 | 1545 | 0 | 860 | 0.01201154 | 0.01145968 | 0.0125929 |
| FULL_B16V1_HIST | border_black_fraction | 9270 | 1545 | 0 | 860 | 0.002428251 | 0.002277837 | 0.002590541 |
| PLAIN_SD_HIST | gray_std | 9270 | 1545 | 0 | 860 | 58.43972 | 58.25148 | 58.62049 |
| PLAIN_SD_HIST | sobel_gradient_mean | 9270 | 1545 | 0 | 860 | 51.12822 | 50.93653 | 51.32915 |
| PLAIN_SD_HIST | high_frequency_fraction | 9270 | 1545 | 0 | 860 | 0.001065453 | 0.001045727 | 0.001088894 |
| PLAIN_SD_HIST | border_black_fraction | 9270 | 1545 | 0 | 860 | 0.04053144 | 0.039365 | 0.04167572 |

| branch | proxy | flagged_images | images | image_flag_fraction | patient_equal_fraction |
| --- | --- | --- | --- | --- | --- |
| OP_CPFM_ON | low_contrast_flag | 0 | 9270 | 0 | 0 |
| OP_CPFM_ON | very_smooth_flag | 35 | 9270 | 0.00377562 | 0.004069767 |
| OP_CPFM_ON | extreme_black_flag | 0 | 9270 | 0 | 0 |
| OP_CPFM_ON | extreme_white_flag | 31 | 9270 | 0.003344121 | 0.003100775 |
| OP_CPFM_ON | high_frequency_flag | 0 | 9270 | 0 | 0 |
| OP_CPFM_ON | long_line_flag | 8486 | 9270 | 0.9154261 | 0.9150194 |
| OP_CPFM_ON | cross_proxy_flag | 671 | 9270 | 0.07238403 | 0.07344961 |
| OP_CPFM_OFF | low_contrast_flag | 0 | 9270 | 0 | 0 |
| OP_CPFM_OFF | very_smooth_flag | 0 | 9270 | 0 | 0 |
| OP_CPFM_OFF | extreme_black_flag | 0 | 9270 | 0 | 0 |
| OP_CPFM_OFF | extreme_white_flag | 36 | 9270 | 0.003883495 | 0.003585271 |
| OP_CPFM_OFF | high_frequency_flag | 0 | 9270 | 0 | 0 |
| OP_CPFM_OFF | long_line_flag | 5777 | 9270 | 0.6231931 | 0.6253876 |
| OP_CPFM_OFF | cross_proxy_flag | 2 | 9270 | 0.0002157497 | 0.0001937984 |
| REAL_HISTORY_REFERENCE | low_contrast_flag | 1 | 13200 | 7.575758e-05 | 7.275902e-05 |
| REAL_HISTORY_REFERENCE | very_smooth_flag | 94 | 13200 | 0.007121212 | 0.004777751 |
| REAL_HISTORY_REFERENCE | extreme_black_flag | 291 | 13200 | 0.02204545 | 0.02223174 |
| REAL_HISTORY_REFERENCE | extreme_white_flag | 84 | 13200 | 0.006363636 | 0.005866317 |
| REAL_HISTORY_REFERENCE | high_frequency_flag | 0 | 13200 | 0 | 0 |
| REAL_HISTORY_REFERENCE | long_line_flag | 7069 | 13200 | 0.5355303 | 0.5381151 |
| REAL_HISTORY_REFERENCE | cross_proxy_flag | 472 | 13200 | 0.03575758 | 0.03635259 |
| FULL_B16V1_HIST | low_contrast_flag | 3 | 9270 | 0.0003236246 | 0.0002906977 |
| FULL_B16V1_HIST | very_smooth_flag | 97 | 9270 | 0.01046386 | 0.01007752 |
| FULL_B16V1_HIST | extreme_black_flag | 0 | 9270 | 0 | 0 |
| FULL_B16V1_HIST | extreme_white_flag | 17 | 9270 | 0.001833873 | 0.001647287 |
| FULL_B16V1_HIST | high_frequency_flag | 4 | 9270 | 0.0004314995 | 0.0003875969 |
| FULL_B16V1_HIST | long_line_flag | 8490 | 9270 | 0.9158576 | 0.9143411 |
| FULL_B16V1_HIST | cross_proxy_flag | 977 | 9270 | 0.1053937 | 0.1054264 |
| PLAIN_SD_HIST | low_contrast_flag | 0 | 9270 | 0 | 0 |
| PLAIN_SD_HIST | very_smooth_flag | 0 | 9270 | 0 | 0 |
| PLAIN_SD_HIST | extreme_black_flag | 0 | 9270 | 0 | 0 |
| PLAIN_SD_HIST | extreme_white_flag | 36 | 9270 | 0.003883495 | 0.003585271 |
| PLAIN_SD_HIST | high_frequency_flag | 0 | 9270 | 0 | 0 |
| PLAIN_SD_HIST | long_line_flag | 5814 | 9270 | 0.6271845 | 0.6291667 |
| PLAIN_SD_HIST | cross_proxy_flag | 2 | 9270 | 0.0002157497 | 0.0001937984 |

低对比/平滑/极端黑白/频谱/长直线/十字阈值事前固定；这类代理可能误检真实界面或采集UI，没有临床校准。未合成为医学真实度总分，未用其删图。ROI/病理认证：NOT_AVAILABLE。

## 11. 重复、节点多样性与统计单位

| branch | equal_to_branch | duplicate_images_beyond_first |
| --- | --- | --- |
| OP_CPFM_OFF | OP_CPFM_OFF | 18 |
| PLAIN_SD_HIST | PLAIN_SD_HIST | 18 |
| REAL_HISTORY_REFERENCE | REAL_HISTORY_REFERENCE | 65 |

全队列固定原始像素坐标数组找候选、全 native RGB ndarray比较证实：首张以外101张像素重复；没有执行SHA/MD5/CRC/文件校验和，也没有移除重复图。

| branch | metric | seed_pairs | nodes | patients | images | pair_mean | node_mean | patient_equal_mean | patient_ci_low | patient_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| OP_CPFM_OFF | rgb_rmse | 23175 | 1545 | 860 | 9270 | 79.66396 | 79.66396 | 79.73276 | 79.45649 | 80.02037 |
| OP_CPFM_OFF | gray_ssim | 23175 | 1545 | 860 | 9270 | 0.3233448 | 0.3233448 | 0.3227686 | 0.3209357 | 0.3245528 |
| OP_CPFM_OFF | lpips | 23175 | 1545 | 860 | 9270 | 0.5363177 | 0.5363177 | 0.5364107 | 0.5350568 | 0.5377195 |
| OP_CPFM_OFF | bit_exact | 23175 | 1545 | 860 | 9270 | 0 | 0 | 0 | 0 | 0 |
| OP_CPFM_ON | rgb_rmse | 23175 | 1545 | 860 | 9270 | 80.95276 | 80.95276 | 80.75752 | 80.43498 | 81.08124 |
| OP_CPFM_ON | gray_ssim | 23175 | 1545 | 860 | 9270 | 0.2128515 | 0.2128515 | 0.213292 | 0.2114299 | 0.2151243 |
| OP_CPFM_ON | lpips | 23175 | 1545 | 860 | 9270 | 0.5798503 | 0.5798503 | 0.5791634 | 0.5776509 | 0.5806815 |
| OP_CPFM_ON | bit_exact | 23175 | 1545 | 860 | 9270 | 0 | 0 | 0 | 0 | 0 |

每臂23,175技术种子配对，总46,350；每节点全6图/15对。image→node→patient等权；REAL为view→Exam→patient。技术重复不作为独立患者，2000患者bootstrap区间与图像分布/MC变化分开。

## 12. 固定图板及隐私

固定12等距节点：前6主图、后6补充，不再择优。四组医学图板共36个节点级PNG，所在绝对目录：`<private output root>/figures/private_medical_panels/`；清单 `PRIVATE_MEDICAL_PANEL_MANIFEST_V1.csv`。MedicalPlate01/02为ON/OFF全部6seed；03为统一0–255差值图及预定中心crop（非ROI）；04为Full/Plain辅助图。原图不改、不锐化、不造毫米尺度；目录700/PNG600权限。没有传出/下载/公开任何医学图片。REAL图板像素因缺乏已验证脱敏显示流程而WITHHELD；保留背景说明和未开展原因，不能写作完成了真实医学可视对照。

## 13. 非医学图件与源数据

8组真实聚合图：每组sourceCSV、RGB600dpi PNG、可编辑文字SVG、嵌字PDF、英汉caption。自动实际绘制文字越界/导出检查PASS，图件清单如下；实际agent/human视觉状态单列，不冒充人工检查。

| filename | source_rows | image_dimensions | dpi | auto_export_status | agent_visual_review_status | human_review_status |
| --- | --- | --- | --- | --- | --- | --- |
| Fig01_Quality_Study_Flow | 5 | 4320x2520 | 600x600 | AUTO_EXPORT_PASS | AGENT_VISUAL_PASS | NOT_RUN |
| Fig02_Real_vs_ON_OFF_FID_KID | 4 | 4320x2280 | 600x600 | AUTO_EXPORT_PASS | AGENT_VISUAL_PASS | NOT_RUN |
| Fig03_Paired_ON_OFF_Pixel_Changes | 48 | 4320x3300 | 600x600 | AUTO_EXPORT_PASS | AGENT_VISUAL_PASS | NOT_RUN |
| Fig04_Contrast_Edge_Frequency | 15 | 4320x2280 | 600x600 | AUTO_EXPORT_PASS | AGENT_VISUAL_PASS | NOT_RUN |
| Fig05_Node_Seed_Diversity | 8 | 4320x2040 | 600x600 | AUTO_EXPORT_PASS | AGENT_VISUAL_PASS | NOT_RUN |
| Fig06_Reference_Support_and_Matching | 5 | 4320x1980 | 600x600 | AUTO_EXPORT_PASS | AGENT_VISUAL_PASS | NOT_RUN |
| Fig07_Artifact_Flags_and_Low_Contrast | 35 | 4320x3060 | 600x600 | AUTO_EXPORT_PASS | AGENT_VISUAL_PASS | NOT_RUN |
| Fig08_Quality_Evidence_and_Limitations | 8 | 4320x2760 | 600x600 | AUTO_EXPORT_PASS | AGENT_VISUAL_PASS | NOT_RUN |

绝对目录：`<private output root>/figures/nonmedical/`、`<private output root>/figures/source_data/`、`<private output root>/figures/captions/`。聚合表12份位于 `<private output root>/tables/`，数值保留完整浮点精度，报告仅格式化显示。

## 14. 实际评阅状态

非医学agent视觉复核：`PASS`；human_review=NOT_RUN。医学agent复核/医师盲评均NOT_RUN。空白盲评模板300行，医生数0、实际评分0；仅服务器私有随机映射，不生成假分数。

| metric | status | reason |
| --- | --- | --- |
| clinical_expert_review | NOT_RUN | No expert review performed; score template remains empty |
| medical_agent_visual_review | NOT_RUN | No task-specific authorization to transfer medical images |
| thyroid_ROI_Dice_or_anatomy_score | NOT_AVAILABLE | No validated annotated thyroid ROI assets with applicable-domain provenance supplied |
| FID_patient_CI | NOT_RUN | Resource-bounded omission frozen before scores; paired KID cluster CI computed |
| target_year_reconstruction_accuracy | NOT_ESTABLISHED | REAL historical context is not query-year paired ground truth |
| new_independent_HT_validation | NOT_RUN | This task is offline image quality evaluation only |


## 15. 英汉 Methods/Results/Discussion

目录 `<private output root>/writing/` 中五份：IMAGE_QUALITY_METHODS_CN_EN_V1.md、IMAGE_QUALITY_RESULTS_CN_EN_V1.md、IMAGE_QUALITY_DISCUSSION_AND_LIMITATIONS_CN_EN_V1.md、FIGURES_AND_TABLES_CAPTIONS_CN_EN_V1.md、PAPER_EXPERIMENT_INTEGRATION_V1.md。每段英文后紧跟中文，术语ledger独立保存。采用图表与科学写作skill约束：评价证据与医学/下游主张分开，呈现全部混合/负面结果，不套用性能门槛。

## 16. GitHub 发布

公开子目录为 `B16v1_Progression/experiments/historical_six_branch_checkpoint_replay_v1/analysis/generation_quality_v1/`。包已通过严格白名单及实际私有值扫描，只有代码、匿名测试、聚合表/非医学图与双语材料。医学影像、患者身份/日期、逐患者特征/概率、权重及临床报告不在包中。为避免将自己的 commit 标识嵌入其本身造成循环自引用，公开包的真实版本以仓库提交历史为准；最终 push 与远端核验在发布后记录于单独执行结果报告，不能以包组装时的状态代替。

## 17. 资源、恢复与原资产保护

GPU冻结评价batch16/FP32，特征阶段自身峰值allocated=393.94873046875MiB；CPU解码6worker、LPIPS/SSIM4worker，未动其他作业/共享Torch/驱动。唯一主链PID锁防重启，逐批缓存核验同 frozen queue/encoder/runtime后可恢复；曾因汇总输出中重复pair_mean字段退出，修复后仅恢复聚合阶段，未重算已有FID/KID、配对或特征，历史失败日志保留且不是当前状态。原源和模型/manifest/已冻结split等元数据postflight `PASS`：58604个保护记录；比较path/resolved_path/size/mtime_ns/inode，非内容hash，不能把元数据核验夸为全部文件内容byte-proof。新生成训练0、新SD采样0、新CLIMATv2训练0、Official Test访问0。

## 18. 负面/混合结果与论文交接

全量与均衡/原支持敏感性矩阵及辅助路线结果完整保留；ON的特征距离较OFF小，但长直线候选患者等权比例 ON=0.9150194、OFF=0.6253876、REAL=0.5381151，不能给ON统一“临床更优”结论。有101张像素重复、1名患者无REAL支持、均衡FID秩限制、技术伪影代理未知特异性、无FID CI/医学ROI/专家认证、REAL显示缺少已验证脱敏。不是“所有指标越高越好”，不因方向或数值不理想换样本/权重/编码器。接入既有七分支论文时只新增离线质量小节，不修改封存结局表或checkpoint；可补合法专家阅片，但须实际资源、授权和盲评流程，不能从自动指标推断医师结论。

本轮生成图像的自动质量统计与视觉对照用于评价统计分布、伪影和结构可视性，不等价于获得医生认证的医学真实性，也不构成新的独立下游 HT 预测验证。
