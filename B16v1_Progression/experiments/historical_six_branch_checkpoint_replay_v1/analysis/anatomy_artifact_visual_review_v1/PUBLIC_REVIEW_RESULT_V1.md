# Public engineering result / 工程结果

Engineering completed with medical-review limitations. Publication provenance is the repository commit history, not a self-referential claim in this file.

## 2. 指令与严格边界

按附件执行；只读继承旧质量目录和历史 Original Progression 消融。新 Stable Diffusion 采样=0、新模型/CLIMATv2训练=0、新 FID/KID=0、新 AUC评价=0、Official Test 内容访问=0。没有按标签、诊断、临床报告或图像好看程度筛选。任务内没有引入新分割模型、改动驱动/CUDA、其他用户环境或 Git 全局配置。

## 3. 权威模型与配对身份

模型为历史 Original Progression 20260921 epoch120，原 B14 CPFM；不是 S2、Clean 或 Strict 变体。ON/OFF 以权威患者/节点、slot、图序、master-seed逐对验证。全部9,270对、每臂1,545节点/860患者、每节点6图序0–5。Full/Plain只为既有路线辅助对照，不是同一 CPFM 单开关因果效应。上轮仅36个固定 pilot ON 重采样有像素一致证据，本轮没有重采样全量。

## 4. 源资产与本轮读取计数

旧五组累计50,280图 QA 仅作为源证据引用。本轮定位实际读取/解码 ON9,270、OFF9,270、REAL13,200，合计31,740图，不能把旧50,280称为本轮新解码数。REAL 为3,154个原生 Exam、859患者；ON/OFF完整配对860患者分母不因REAL少1人而改变。Full/Plain各9,270图仅既有辅助；本轮候审核心各实际读取72图。

## 5. 冻结复核样本与覆盖

原固定12节点不替换，前6主图/后6补充，全部六图序组成72对/144图/12患者。四层各12对，实际48对/96图/48不同患者，与核心患者及配对重叠=0。总核心+分层120对/240图/60患者。医生候选只取核心图序0、2、5，为36对/72卡/12患者，是核心子集不能另加为新患者或独立图。

| stratum | all_queue_pairs | eligible_noncore_pairs | target_pairs | selected_pairs | selected_patients | shortfall |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | 671 | 668 | 12 | 12 | 12 | 0 |
| S2 | 2914 | 2893 | 12 | 12 | 12 | 0 |
| S3 | 4901 | 4858 | 12 | 12 | 12 | 0 |
| S4 | 364 | 361 | 12 | 12 | 12 | 0 |

四层覆盖不足：另有420对 OFF-only 未分层，完整保留全量统计；未强制归S4，未剔除。随机排序种子20261012，优先未用患者，抽样分数不是实际纳入概率，不将48对结果外推总体发生率。

## 6. 原flag复现与新增定位器

本轮31,740张旧长线/十字flag及原线段数逐图匹配，不一致=0。CPU6进程、OpenCV单线程；未调用GPU。

| branch | images_read_and_decoded_this_stage | old_long_line_hits | old_cross_hits | old_flag_mismatches | actual_line_candidates | actual_cross_candidates | patients |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OP_CPFM_ON | 9270 | 8486 | 671 | 0 | 237968 | 2319 | 860 |
| OP_CPFM_OFF | 9270 | 5777 | 2 | 0 | 11166 | 2 | 860 |
| REAL_HISTORY_REFERENCE | 13200 | 7069 | 472 | 0 | 72788 | 4181 | 859 |

原定义不变：RGB灰度0.299/0.587/0.114，全图512双线性，Canny50/150，Hough rho1/thetaπ/180/票数50/minLength128/maxGap10，水平/垂直容差10°，双线平均坐标与投影范围的近似交点。新 `LOCALIZATION_DIAGNOSTIC_V1` 追加位置，不替换旧flag、不覆盖旧比例。

## 7. 几何区域及实际候选分布

外圈26像素、中心[102:410)×[102:410)、其余为中间区域；不是甲状腺 ROI、病灶或毫米坐标。线段以≤1像素间隔取点，十字区域为双臂并集，交点位置单独私有记录。跨区候选保留各区并额外标 MULTIZONE；各区域比例可能重叠不相加为100%。

| branch | metric | images | patients | image_mean | patient_mean |
| --- | --- | --- | --- | --- | --- |
| OP_CPFM_ON | line_border_any | 9270 | 860 | 0.7727076591154262 | 0.773546511627907 |
| OP_CPFM_ON | line_central_any | 9270 | 860 | 0.8773462783171521 | 0.8760658914728683 |
| OP_CPFM_ON | line_intermediate_any | 9270 | 860 | 0.8725997842502697 | 0.8718992248062014 |
| OP_CPFM_ON | line_multizone_any | 9270 | 860 | 0.8539374325782093 | 0.852906976744186 |
| OP_CPFM_ON | cross_border_any | 9270 | 860 | 0.0425026968716289 | 0.0426356589147286 |
| OP_CPFM_ON | cross_central_any | 9270 | 860 | 0.0706580366774541 | 0.0718023255813953 |
| OP_CPFM_ON | cross_multizone_any | 9270 | 860 | 0.0689320388349514 | 0.0695736434108527 |
| OP_CPFM_OFF | line_border_any | 9270 | 860 | 0.4548004314994606 | 0.4564922480620155 |
| OP_CPFM_OFF | line_central_any | 9270 | 860 | 0.221143473570658 | 0.2226744186046511 |
| OP_CPFM_OFF | line_intermediate_any | 9270 | 860 | 0.354908306364617 | 0.3554263565891472 |
| OP_CPFM_OFF | line_multizone_any | 9270 | 860 | 0.2225458468176915 | 0.2220930232558139 |
| OP_CPFM_OFF | cross_border_any | 9270 | 860 | 0.0001078748651564 | 9.689922480620156e-05 |
| OP_CPFM_OFF | cross_central_any | 9270 | 860 | 0.0001078748651564 | 9.689922480620156e-05 |
| OP_CPFM_OFF | cross_multizone_any | 9270 | 860 | 0.0002157497303128 | 0.0001937984496124 |
| REAL_HISTORY_REFERENCE | line_border_any | 13200 | 859 | 0.4498484848484848 | 0.4540471062010667 |
| REAL_HISTORY_REFERENCE | line_central_any | 13200 | 859 | 0.243560606060606 | 0.2460986430630877 |
| REAL_HISTORY_REFERENCE | line_intermediate_any | 13200 | 859 | 0.3001515151515151 | 0.30538726693328 |
| REAL_HISTORY_REFERENCE | line_multizone_any | 13200 | 859 | 0.2417424242424242 | 0.2461196838814329 |
| REAL_HISTORY_REFERENCE | cross_border_any | 13200 | 859 | 0.0257575757575757 | 0.0262990324714921 |
| REAL_HISTORY_REFERENCE | cross_central_any | 13200 | 859 | 0.0323484848484848 | 0.0331877513943454 |
| REAL_HISTORY_REFERENCE | cross_multizone_any | 13200 | 859 | 0.0331060606060606 | 0.0336999611111951 |

## 8. 患者配对效应与不确定性

每节点六图平均，再患者内节点平均，配对 ON−OFF 同患者同权、860人、2,000次百分位 bootstrap、种子20261012。22项技术对比是探索性，未作多重性校正确证。

线段患者等权均值 ON=25.69293、OFF=1.21366，差值=24.47926，95%CI[23.80328, 25.15729]；十字差值=0.250678，95%CI[0.220250, 0.283527]。正值表示检测器响应更多，不代表质量改善或已证实更多医学伪影。

| metric | on_patient_mean | off_patient_mean | delta_on_minus_off | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| old_long_line_flag | 0.9150193798449612 | 0.6253875968992249 | 0.2896317829457365 | 0.2784883720930233 | 0.3012621124031008 |
| old_cross_proxy_flag | 0.0734496124031007 | 0.0001937984496124 | 0.0732558139534883 | 0.0677325581395348 | 0.0789728682170542 |
| line_candidates | 25.692926356589144 | 1.2136627906976745 | 24.479263565891475 | 23.803282461240308 | 25.157286821705423 |
| cross_candidates | 0.2508720930232558 | 0.0001937984496124 | 0.2506782945736434 | 0.2202495155038759 | 0.2835271317829457 |

## 9. 人工观察、结构可视性与未知项

自动技术定位已执行；NONCLINICAL_HUMAN、CODEX医学视觉和CLINICIAN均NOT_RUN，真实观察者/实际完整医学图查阅/实际评分=0。240行空白非临床表的可视性字段NOT_REVIEWED；实际未观察不写UNCERTAIN。TP/FP/TN/FN、敏感度、特异度、精确率、Kappa、ICC为空/NA，不能当0。可能UI、采集标记、真实组织界面或生成伪影只是待验证解释，没有逐病例已观察分类。

## 10. 私有候审图板与医生准备

实际新医疗图板84个PNG：A12、B12、C48、D12；另有72张无语义随机医生卡。A原12节点全部6图序；B严格匹配Full/Plain，标route-only；C原图与技术候选叠加并列，全部48对保留；D统一差值0–255和事前中心[192:320)裁片，非ROI。REAL面板E=WITHHELD，无REAL像素图板。

五项1–5分医生模板高分方向已写明，几何严重度高分更严重，其余高分较有利；允许UNSURE/NOT_ASSESSABLE/SKIPPED且不当0。实际无医生、无填写分数、无量表训练及非患者示例。72卡单图随机呈现，映射单独私有；去标签不保证感知盲法，当前界面未激活，没有启动HTTP服务。

## 11. 隐私、脱敏和显示审批

医学像素外传=0，聊天/外部视觉模型/API展示=0，本地医学/诊断图保存=0。新根及映射/评分/医学图目录700，私有图片600。生成图也按潜在敏感医学影像处理；未自动遮挡原图。PHI审核PENDING、REAL显示WITHHELD、公开医学图许可NOT_APPROVED。存服务器不等于脱敏；所有图板是候审，不是可直接发表。

## 12. 七表、六图与视觉QA

7张汇总表、6组非医学图各PNG/SVG/PDF（18导出）、6份源数据CSV、6份英汉图注、5份英汉材料及术语表。PNG为RGB600dpi，SVG可编辑文字无栅格嵌入，PDF嵌入TrueType；实际draw文字边界与≥8pt检查PASS。6组非医学图真实会话目检状态：PASS_AFTER_ACTUAL_VIEW；这不是医学影像复核，未本地保存图片。

| figure | dimensions | dpi | auto_export | actual_nonmedical_agent_visual_review |
| --- | --- | --- | --- | --- |
| Fig01_Review_Design_and_Cohorts | 4320x2580 | 600 | PASS | PASS_AFTER_ACTUAL_VIEW |
| Fig02_Line_Cross_Proxy_and_Regions | 4320x2340 | 600 | PASS | PASS_AFTER_ACTUAL_VIEW |
| Fig03_Flag_Audit_And_Unknowns | 4320x2220 | 600 | PASS | PASS_AFTER_ACTUAL_VIEW |
| Fig04_Paired_Artifact_Difference | 4320x2100 | 600 | PASS | PASS_AFTER_ACTUAL_VIEW |
| Fig05_Visibility_Observation_Distribution | 4320x2220 | 600 | PASS | PASS_AFTER_ACTUAL_VIEW |
| Fig06_Claim_Evidence_Limitations | 4320x2700 | 600 | PASS | PASS_AFTER_ACTUAL_VIEW |

## 13. 论文表述与负面结果

上轮自然图像特征距离ON较小；本轮ON线段/十字候选更多。两者测量不同问题，不构成医学结构质量已更好的证据。本轮不能说OFF更自然或甲状腺更清晰，因为没有真实看图。讨论明确域偏差、重采样/采集混杂、近似交点、多区重复、抽样权重缺失、医生/ROI/目标年真值缺位。旧Fig01–08保持，新增用Qual-Fig01–06、Qual-Table01–07避免覆盖。

## 14. 验证、失败与恢复

匿名合同自检7项PASS；代码语法检查PASS。首轮启动安全查询因命令行自身匹配而阻止启动；随后实际进程核查无现存任务，使用单实例flock启动，未重复生成。定位全部成功，无源解码或flag失败。绘图初次Fig02标题超出画布被自动阻止；只修标题换行，原数据/阈值/样本不变，并保存旧脚本时间戳备份。Markdown初次因可选tabulate缺失退出，改用原生Markdown表格写入，不安装或修改共享环境。发布逐值扫描较慢，改为相同字面值/边界规则的一次编译前缀树；实际私有值检查不减少。仅停止自身prepare-only父进程后，发现网络clone子进程仍在下载、重试工作区缺少HEAD；随后只停止本任务clone及其index-pack，保留不完整目录，改为从已核实基线的本机上一轮Git只读副本复制新的独立checkout，不用硬链接，仍核验remote/main。候选定位有256图分块账本，同冻结配置resume；不自动改阈值填缺失。

## 15. 源保护与禁止项复核

实际对59,179个唯一源文件逐个比较path/resolved_path/size/mtime_ns/inode，状态PASS，变化数0。涵盖原图、权重、manifest、split/locks、旧质量材料及历史源依赖；这是元数据保护检查，不冒充内容校验认证。没有运行或报告文件哈希。当前任务脚本不打开Official Test内容、不加载Torch/训练代码、不改上游。禁止项依据实际调用与保护范围核对，不作系统全盘监控保证。

## 18. 未完成项及后续条件

必须另行提供真实机构授权/PHI脱敏确认、本地完整图查阅记录、实际非临床或具备甲状腺超声经验医生、量表培训/非患者示例及适用ROI参考，才可开展医学观察和评分。两名真实医生才可评分者间一致性；只有一名只报告单人主观描述。此工程结束不需要虚构医学结论，也不自动启动下一训练、采样或新Test评价。
