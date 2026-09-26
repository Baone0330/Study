# ProgressionResidualAdapter v1

输入为既有 Delta Encoder 的 `P∈R16`，不改变 `ΔF_i=F_i−F_(i−1)`。新增映射为 `LayerNorm(16) → Linear(16,16) → tanh`，输出有界残差 `r_p`。标量 `g_c=sigmoid(a_c)` 初始化约 0.10；condition 分支按执行文档的字面定义计算 `c_tilde=tanh(c_base+g_c r_p)`。仅 C/CP arm 启用；R/P arm 使用原 `c_base`。

新增可训练参数 305 个：LayerNorm 32、Linear 272、gate 1。Stage 1 仅训练该模块；R backbone、ExamSet encoders、SD1.5/medical LoRA 全部冻结。Stage 2 未执行。

重要归因限制：由于外层 `tanh`，即使把 `P` 设为零，C/CP 也不严格回到 R 的 `c_base`；并且 Linear bias 可产生非零残差。因此 C−R/CP−R 不可单独归因于 progression 内容。`P_ZERO` tensor control 与真实 P 对照用于量化这部分结构性差异，最终结论必须保留该限制。
