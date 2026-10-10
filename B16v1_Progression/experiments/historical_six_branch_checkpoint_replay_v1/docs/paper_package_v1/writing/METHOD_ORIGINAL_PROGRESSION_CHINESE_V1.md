# Original Progression 方法草稿

## 输入角色与历史版本身份

Original Progression 对应原训练 seed 20260921、原 selected checkpoint epoch120，而不是下游三个分类器种子42、3407、2026。生成时使用冻结 BEFORE、AFTER 和 query 编码器获得 b,a,q∈R¹⁶。各合法历史检查还单独经过冻结 AFTER 编码器形成 e₁,…,e_L；历史条件只取查询前真实检查，未将之前生成图递归作为历史输入。本轮未重新读取训练患者影像或重新训练表征。

## 相邻变化编码与融合

相邻差分定义为 Δ_k=e_{k+1}−e_k。每个十六维差分经过 Linear(16,32)→LayerNorm(32)→GELU→Linear(32,16)，得到 d_k。有效差分由 mask m_k 控制，纵向输出 p=Σ_k m_k d_k / max(Σ_k m_k,1)。少于两次检查时，源码构造一条零差分并把 mask 置零，所以 p=0；但 b/a/q 仍进入后续融合，最终条件并不因此必然为零。

四项十六维输入 [b;a;q;p] 拼接为六十四维，经过 Linear(64,128)→LayerNorm(128)→GELU→Linear(128,64) 得到 h。最终 c=tanh(W_adapter h+b_adapter)∈R¹⁶，其中 cpfm_adapter 是单独的 Linear(64,16) 适配头，不是 B14 CPFM 控制器。生成入口以 c 替换旧 ConditionBridge 的十六维条件向量，再传入独立 B14 调制网络。原 ConditionBridge 仍被调用以产生被记录的 scale 信息，但该历史 hook 不消费 condition_scale；因此图示将这条元数据路径标为不活动。

## 原训练监督与损失

原训练样本由历史前缀预测下一次历史检查的冻结条件空间表示，而非官方临床目标检查。按原源码，teacher 条件把前缀 BEFORE 与下一检查 AFTER、查询编码组合；目标变化 δ_target 为下一检查 AFTER 表征减去当前 AFTER 表征，old 条件由原冻结 bridge 对前缀状态计算。患者级 inner train/dev 划分用于原模块训练选模；本轮仅说明原实现，不重新审计整个数据链。

原损失由五项组成：

L = MSE(c,c_teacher) + 0.20[1−cos(c,c_teacher)] + 0.50 MSE(p,δ_target) + 0.10[1−cos(p,δ_target)] + 0.10 MSE(c,c_old)。

cos 项在 batch 维取平均。原配置为 AdamW、lr0.002、weight_decay10⁻⁴、batch128、最多120 epochs、patience18、梯度裁剪1.0；这些是原训练记录，不是本轮新训练参数。原 F16 提取器冻结。对“teacher 是否直接代表病理进展”等解释，本轮只支持其为冻结特征/条件空间的下一历史检查监督，不支持把此监督自动等同于医学病程真值。

## 范围与解释边界

该模块引入的并非一条仅加法附着、其它结构不变的 P16 支路：它把 Full 的48→32→16条件路径替换为64→128→64→16融合，参数量、非线性与可训练融合均改变。因此 Original Progression−Full 的历史指标差不能直接称为 P16 独立因果贡献。MASK pooling 保持序列差分的聚合结构，但不等同于 Transformer 时间注意力，也不保证临床进展方向已被学得。历史上该分支内部 Val 指标最高是观察事实，后续解释应维持这种证据强度。

段落工作：版本和输入→差分与公式→实际传递→原损失→可归因性限制。缺项：原生成时安装库精确提交、独立病理进展真值和外部验证均未在此任务恢复或新增。
