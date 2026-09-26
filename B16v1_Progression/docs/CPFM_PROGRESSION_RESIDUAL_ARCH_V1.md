# B14CPFM Progression Residual v1

继承已封存 `B14CPFM`，不重写其 gamma/beta heads、RMS 归一化、magnitude heads、layer gates 或 U-Net hooks。既有 `P∈R16` 经 `LayerNorm(16) → Linear(16,16) → SiLU` 形成 `h_prog`，以 `h_patient_new=h_patient_base+g_p h_prog` 注入 patient-direction latent；`g_p=sigmoid(a_p)` 初始化约 0.10。仅 P/CP arm 启用；R/C arm 关闭。

新增可训练参数 305 个：LayerNorm 32、Linear 272、gate 1。Stage 1 只训练新增投影和 gate；原 B14CPFM、SD1.5、medical LoRA 均冻结，Stage 2 未执行。已有零条件参考分支保持 progression-free，以免 `effective()` 相减时消去 active residual。

归因限制：Linear bias 与 SiLU 路径使 `P=0` 时的 patient-direction residual 未必为零；因此必须对比 `P_ACTIVE`、`P_ZERO` 和 R 的 tensor/epsilon 差异，不能将 P−R 点估计直接解释为 progression 信息的独立贡献。
