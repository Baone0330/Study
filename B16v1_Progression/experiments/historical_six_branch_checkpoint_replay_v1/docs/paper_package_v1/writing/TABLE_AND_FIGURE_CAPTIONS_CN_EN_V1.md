## Fig. 1 | Historical architecture and stage separation

a Original F16 and query encoders provide the Full condition or the replacement Original Progression route. b Frozen B14 hooks modulate the configured diffusion UNet. c Existing synthetic training inputs and CLIMATv2 classification belong to separate historical stages. No new generation, training or forward was performed in this packaging stage. Later Clean/Strict systems are excluded. Source data are provided in the corresponding source CSV.

## 图 1 | 历史架构与阶段分离

a 原F16及查询编码器形成Full条件或替换式Original Progression条件。b 冻结B14通过hook调制扩散UNet。c 既有合成训练输入与CLIMATv2分类属于独立历史阶段。本封存阶段没有新生成、训练或前向，后来Clean/Strict系统不参与。对应源CSV提供结构证据。

## Fig. 2 | Original Progression layer structure

a Adjacent differences are constructed from frozen pre-query exam representations. b A shared delta MLP and valid-mask mean produce p16. c Four 16-dimensional inputs enter 64→128→64 fusion and a separate 64→16 tanh adapter; the adapter is not B14 CPFM. d The original historical losses and source roles are shown. Zero valid deltas imply p=0, not necessarily a zero final condition. Source data and original-source anchors accompany the figure.

## 图 2 | Original Progression层结构

a 冻结查询前检查表征构成相邻差分。b 共享差分MLP与有效mask均值得到p16。c 四项16维输入进入64→128→64融合及独立64→16 tanh适配头，适配头不是B14 CPFM。d 展示原历史损失与输入角色。没有有效差分意味着p=0，不意味着最终条件必为零。附源数据与源码锚点。

## Fig. 3 | Historical B14 modulation and effective UNet injection

a Magnitude, zero-referenced direction and patient residual sub-networks. b Directions combine with bounded magnitudes and residual activation modulation. c Pre-hooks operate at the first ResNet input of D0–D3, with channels320/320/640/1280. d Configured ResNet interfaces are shown; the exact historical installed-library revision is not recovered. No new UNet forward or medical image test was run. Source data include the strict-loaded module inventory.

## 图 3 | 历史B14调制及有效UNet注入

a 展示幅值、零参考方向和病人残差子网络。b 方向与有界幅值结合，形成激活残差调制。c pre-hook位于D0–D3首个ResNet输入，通道为320/320/640/1280。d 展示配置支持的ResNet接口；原安装库精确revision未恢复。未执行新UNet前向或医学影像测试。源数据包括严格加载模块盘点。

## Fig. 4 | Actual multi-view fixed-K6 CLIMATv2

a Valid images pass through ResNet18 and within-exam masked feature-map mean; the standard GAP/FC is inactive. b Spatial Radiologist and constant-zero-input GP FeaT produce exam tokens. c The original prognosis Transformer is reused for the masked six-slot sequence, followed by the five-logit decoder and two-logit HT projection. d Three-seed averaging occurs after model output. Report/time content is inactive, but masks and learned positions remain active. Source data are provided.

## 图 4 | 实际多图固定K6 CLIMATv2

a 有效图像经ResNet18并在检查内按mask平均特征图；标准GAP/FC不活动。b 空间Radiologist与零输入常量上下文GP FeaT形成检查token。c 原prognosis Transformer复用于六槽mask序列，接五logit解码和二logit HT投影。d 三seed平均位于模型输出之后。报告/时间内容不活动，mask与学习位置仍活动。附源数据。

## Fig. 5 | Six-branch historical discrimination scores

a ROC-AUC. b AP, defined as average_precision_score, not trapezoidal PR integration. Bars show equal-mean positive-probability ensembles from seeds42,3407,2026 on the same historical internal Val (n=189; HT0=131, HT1=58). No variability, CI or p-values are estimated here. Plain-SD above Full is retained. Source data are provided as Fig05_source_data.csv.

## 图 5 | 六分支历史区分指标

a ROC-AUC。b AP定义为average_precision_score，不是PR梯形积分。柱形来自42、3407、2026三个种子正类概率等权ensemble，同一历史内部Val共189人（HT0=131、HT1=58）。本轮不估计变异、CI或p值，完整保留Plain-SD高于Full的结果。源数据见Fig05_source_data.csv。

## Fig. 6 | Threshold-dependent historical metrics

a Accuracy. b Sensitivity. c Specificity. d F1. Values use each branch’s unchanged native historical threshold strategy, with positive probability greater than or equal to the chosen threshold; n=189, three-seed probability ensemble. No threshold was newly tuned for this figure. Different error trade-offs are retained. Source data are provided as Fig06_source_data.csv.

## 图 6 | 阈值相关历史指标

a Accuracy。b Sensitivity。c Specificity。d F1。各值沿用该分支原历史阈值策略，正类判定为概率大于或等于阈值；n=189、三seed概率ensemble。本图未重新调阈值，保留不同错误类型权衡。源数据见Fig06_source_data.csv。

## Fig. 7 | Descriptive historical ensemble differences

a ROC-AUC differences. b AP differences. Points are automatically subtracted from the full-precision six-branch table; positive values favour the first named condition. These are observed point differences, not isolated causal module effects. No CI or p-value is computed. The Plain-SD–Full comparison is included explicitly. Source data are provided as Fig07_source_data.csv.

## 图 7 | 描述性历史ensemble差值

a ROC-AUC差值。b AP差值。点由完整精度六分支表自动相减，正值有利于前一个条件。这些是观察到的点差值，不是独立因果模块效应。未计算CI或p值，明确包括Plain-SD−Full比较。源数据见Fig07_source_data.csv。

## Fig. 8 | Historical replay provenance and evidence

a Six branches each map to three original selected checkpoints and previously recomputed patient-matched probability outputs; only the counts, not patient records, are displayed. b Eighteen real forwards, six ensembles and42 matching display fields are tied to the existing public replay commit. New forwards in this paper stage are zero. Reproducibility is not independent clinical validation. Source data are provided as Fig08_source_data.csv.

## 图 8 | 历史重放来源与证据

a 六分支各对应三个原selected checkpoint与前阶段真实重新计算的配对概率输出；仅展示计数，不展示患者记录。b 18次真实前向、6个ensemble与42项匹配展示字段连接到既有公开重放commit。本论文封存阶段新增前向为零；计算可复现性不等于独立临床验证。源数据见Fig08_source_data.csv。

## Table 1 | Historical six-branch probability ensembles

The same historical internal Val contains189 patients (HT0=131, HT1=58); positive probabilities of seeds42,3407,2026 are equally averaged. AP is average_precision_score; other threshold metrics retain original branch-specific rules. No new independent Test is reported.

## 表 1 | 历史六分支概率ensemble

同一历史内部Val189人（HT0=131、HT1=58），三个种子正类概率等权平均。AP为average_precision_score，阈值指标沿用原分支策略，无新增独立Test。

## Table 2 | Descriptive ensemble comparisons

Differences are derived automatically from full-precision Table01; no CI/p-value or independent causal component estimate is added.

## 表 2 | 描述性ensemble比较

差值从完整精度Table01自动计算，未增加CI/p值或独立因果模块估计。

## Table 3 | Historical selected checkpoint inventory

Private absolute locators remain on the server; the public table contains only semantic branch, seed, original best epoch and existence/size metadata. These weights were replayed in the preceding stage, not forwarded again here.

## 表 3 | 历史selected checkpoint盘点

具体私有路径留服务器；公开表只含语义分支、seed、原best epoch与存在/大小信息。这些权重在前阶段重放，本轮没有再次前向。

## Table 4 | Source-verified original layer inventory

Module attributes and learned-token/scale parameters come from original source and strict CPU schema loads. Inactive and metadata-only paths are marked. Complete SD/LoRA weights and historical library revisions are not claimed verified.

## 表 4 | 源码核验原版层清单

模块属性及学习token/尺度参数来自原源码与CPU严格schema加载，明确标出不活动/元数据路径，不声称已验证完整SD/LoRA权重或历史库revision。

## Table 5 | Claim evidence and limitations

Each statement is marked supported, observed, unsupported or not evaluated. Computation reproducibility does not establish external clinical efficacy.

## 表 5 | 陈述证据与局限

逐条标记支持、观察、不支持或未评价；计算可复现性不建立外部临床有效性。
