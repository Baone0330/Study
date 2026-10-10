# 历史原版 B16v1 方法草稿

## 研究对象与阶段关系

本节描述历史六分支实验实际采用的纵向生成与分类流程，而非后来 CleanF16、Clean P16 或 Strict Pre-Target 系统。历史表中的 Full B16v1 特指“历史图像表征＋查询时间编码＋B14 条件调制”的纵向版本，不包含更早 Full-B16 系统的报告、病理、结构或纹理分支。生成器首先离线构建既定合成训练输入，下游 CLIMATv2 随后独立训练；生成与分类不属于一次联合端到端优化。本轮仅整理已存在资产，没有重新执行这些训练与生成步骤。

## 历史图像与查询表征

历史图像先转换为灰度并双线性缩放至 32×32，计算八项统计量：均值、标准差、10%/50%/90%分位数、横向和纵向平均差分的绝对值、灰度范围。这里的平均差分不是梯度绝对值的平均，不能改写成更复杂的纹理描述。八维图像统计与六维报告计数拼接为十四维输入；本历史生成入口将报告列表置空，因此报告六维为零，而非活动的医学语言编码。

BEFORE 与 AFTER 使用独立的 ExamSetEncoder，逐图执行 Linear(14,32)→LayerNorm(32)→GELU→Linear(32,16)，随后在图像集合维求均值。BEFORE 合并查询前、除最近检查外的历史图像；AFTER 对应最近一次合法历史检查。F16 因此是轻量统计量 MLP，不是 CNN 或 ResNet。空集合以十四维零行经投影处理，带偏置网络的输出不应被自动称为零向量。

LongitudinalQueryEncoder 接收九项数值：年序号、查询相对锚点时间、查询与 BEFORE/AFTER 的间隔、两端日期存在标志、查询月份的正余弦及日期字符串有效长度。数值经 Linear(9,24)→LayerNorm(24)→GELU→Linear(24,16) 得到查询表征。月份特征确实存在于原实现；本轮不把它改写成仅相对时间编码。

Full 的 ConditionBridge 将三项十六维表征拼接为四十八维，经过 Linear(48,32)→LayerNorm(32)→GELU，再由 Linear(32,16)＋tanh 输出条件向量。原网络还包含 scale、attribute、proxy 头。实际历史生成没有消费 attribute/proxy 输出；condition_scale 虽被计算和记录，但 generate_one 与 B14Hooks 没有读取它，因此不能把该标量画成有效调制控制路径。Original Progression 分支替换的是条件向量融合路径，保留原冻结 F16 和查询编码器，其具体结构见单独方法草稿。

## B14 CPFM 与扩散接口

B14 CPFM 是独立于 P16 的调制控制器。其幅值路径拼接 condition16、timestep8 和 layer-embedding8，采用 LayerNorm(32)→Linear(32,128)→SiLU→Linear(128,128)→SiLU，并由逐层幅值头产生 sigmoid 有界强度。方向路径再接入十六维历史摘要和当前激活摘要，采用六十四维输入的两层 128 维 MLP。病人残差路径包含 Linear(32,32)→SiLU→Linear(32,16) 表征和六十四维输入的 64 维方向网络；历史摘要在本生成路由固定为零，当前摘要来自扩散激活，而非额外临床检查图像。

方向按同一上下文下真实条件与零条件的差构造，并叠加系数 0.25 的病人方向残差，再进行 RMS 归一化。gamma、beta 分别由幅值与方向结合，beta 还受安全 sigmoid 限制。原 gamma_heads/beta_heads 的原始输出虽在 b12_raw_forward 中被计算，但 effective 最终丢弃这两个原始输出，不能将其作为最终方向来源。

调制在 UNet 四个 down block 的第一个 ResNet 之前通过 pre-hook 注入。四处输入通道为 320、320、640、1280；它们不同于各 down block 输出通道 320、640、1280、1280。记空间归一化激活为 u=(x−μ_HW)/max(σ_HW,10⁻⁵)，实际操作为 y=x+s_l[((1+γ_l)⊙u+β_l)−u]，其中 s_l 是学习层标量的 sigmoid。这是相对于原激活的残差修正，不是直接用标准 FiLM 输出替换 x。零条件的方向差和幅值掩码使调制为零，故源码代数上得到 y=x；本轮未重新运行影像恒等性实验。

扩散接口采用本地 Stable Diffusion 1.5、既有医学超声 LoRA、PNDM epsilon 预测、30 个推理步、guidance 7.0 和 512×512 输出。每节点原历史生成六张图，样本沿用冻结种子，未按质量择优。SD 主干、VAE、文本编码器、LoRA 及 F16/B14 在该生成入口均冻结。源配置和当前安装库支持记录 UNet 的卷积/ResNet/交叉注意力接口，但原生成时安装库的精确提交未独立恢复，完整 LoRA 逐层 rank 和完整 SD 权重 schema 未在本轮审计。此处不声称已完成预训练来源审计。

## 下游 CLIMATv2 的实际结构

原 FixedDataset 在六个固定年度槽提供 RGB256 图像及原始全部 views，使用 ImageNet 均值/标准差归一化。报告输入为 768 维零向量、报告缺失标志为真，两个时间输入为零；学习的位置编码仍活动。历史数据构建的 REAL／B16／LOCF 槽分配沿用旧协议，不能将本节包装为新的严格未来预测定义。

有效图像共享 torchvision ResNet18。主干包含 7×7 stride2 Conv→BN→ReLU→3×3 stride2 MaxPool，以及四级 [2,2,2,2] BasicBlock；BasicBlock 的主路径为 3×3 Conv→BN→ReLU→3×3 Conv→BN，并有恒等或 1×1 stride2 投影残差。256×256 图像输出 512×8×8 特征图。同一检查的 views 按有效掩码逐位置均值聚合，保持空间特征图；标准 ResNet 的全局平均池化及 1000 类 FC 存在于对象中，但实际 core.forward_img 不调用它们。

每检查的 64 个空间 token 输入 512 维 Radiologist FeaT。元信息接口由八维零报告投影、两维零时间及一维零存在标志形成十一维输入，经过 Linear(11,64)→ReLU→LayerNorm(64) 映射到 64 维 GP FeaT。GP 仍作为可学习的常量上下文活动，但不含患者报告或时间内容。Radiologist 的 512 维状态与 GP 的 64 维 CLS 拼接为 576 维多模态 token。原 prognosis FeaT 同时用于每检查空间预测及六槽序列预测；序列读取使用 wrapper 的 CLS 和学习位置表，而非跨检查均值池化。

Radiologist、GP 和 prognosis Transformer 各深度 1、注意力 4 heads；其 FFN 隐层分别为 256、64、256。注意力真实缩放使用 dim⁻¹/²，而非未经核验地改成 head_dim⁻¹/²。患者读取经过 LayerNorm(576)→Linear(576,288)→GELU→Linear(288,5)，再由 wrapper Linear(5,2) 得到 HT logits。Softmax 的正类概率用于患者级评价，三个训练种子的概率等权平均位于模型输出之外。

## 证据与待补输入

上述结构由原历史调用器、源码锚点及严格 CPU 权重结构核验支持。F16、Original P16、B14 和完整下游对象参数总数分别为 5,421、19,008、1,707,072、15,640,702；总数包含对象内未活动的头，不等于所有参数均影响该历史推理路径。原稿需要作者在投稿前补入经核验的 SD／LoRA／CLIMATv2 文献和资产许可信息；本草稿未杜撰参考文献、医学有效性或独立 Test 结论。

写作结构：阶段关系→输入编码→调制机制→扩散接口→下游结构→证据边界。若需修改，请指出具体段落或陈述，以便仅修订该处并保持已核验术语一致。
