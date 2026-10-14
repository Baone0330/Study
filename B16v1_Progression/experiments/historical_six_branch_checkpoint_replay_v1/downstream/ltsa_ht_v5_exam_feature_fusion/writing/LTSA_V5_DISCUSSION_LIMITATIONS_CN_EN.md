# V5 Discussion and limitations / 讨论与局限

The observed V5 OP−REAL effects were AUC0.011582 and AP-0.008114. Both patient-bootstrap confidence intervals included zero: the AUC point estimate increased while AP slightly decreased, without clear evidence of a stable advantage on this internal-validation set. These measurements evaluated the single pre-registered fusion strategy and did not trigger further tuning or selective retraining. Neither direction establishes a universal benefit or failure of synthetic imaging.

V5实测OP−REAL效应为AUC0.011582、AP-0.008114。两个患者bootstrap置信区间均包含0：AUC点估计提高而AP略降，尚无清晰证据支持本内部验证集上的稳定增益。这里只评价唯一预设融合方案，不据此调参或择优复训；无论正负都不能建立合成图像普遍有效或无效的结论。

The method retains REAL token counts and observed month indices, but it also introduces a learned residual gate and projection, different image encoding order, and potential BatchNorm changes. Cross-protocol V3/V4 versus V5 differences therefore cannot isolate token count as the unique causal explanation. Attaching query content to a later historical REAL Exam is retrospective within-window augmentation, not prediction at query time using an actually future REAL examination.

该方法保持真实token数量和观测月份，但也引入学习残差gate/projection、不同编码顺序及可能的BN变化。跨协议差异不能唯一归因于token数。把query信息附着后续历史真实检查是窗口内后验增强，不是在query时刻利用实际未来真实检查预测。

All evaluation concerns the same 189 internal-validation patients. Prior V3/V4 outcomes informed the choice of V5, and validation also selected checkpoints, stopping and thresholds. Patient bootstrap does not capture all adaptive design and selection uncertainty. No independent clinical generalization, Official Test performance, or superiority over all generators is claimed.

评价仅涉及相同189名内部验证患者；既有V3/V4结果影响V5设计，Val也参与checkpoint、早停和阈值选择。患者bootstrap不能覆盖全部适应性设计/选择不确定性，不能声称独立临床泛化、Official Test表现或优于所有生成器。
