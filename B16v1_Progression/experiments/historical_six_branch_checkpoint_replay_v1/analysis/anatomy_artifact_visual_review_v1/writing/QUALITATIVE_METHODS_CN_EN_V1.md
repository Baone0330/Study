# QUALITATIVE METHODS CN EN V1.md

We performed a frozen, offline positional audit of historical Original Progression with original CPFM injection ON versus OFF. All9,270 matched pairs (1,545 nodes,860 patients) shared authoritative patient/node, image-index and master-seed identity. No image was sampled anew, modified at source, quality-filtered or evaluated on Test.

对历史 Original Progression 及原 CPFM 注入 ON/OFF 开展固定离线位置复核。全部9,270对来自1,545节点、860患者，依据权威患者/节点、图序及 master-seed 配对；无新采样、源图修改、质量筛选或 Test 评价。

The prior fixed12 nodes were inherited unchanged; all six indices supplied72 core pairs. A seeded permutation selected12 distinct-patient pairs from each of four prespecified flag strata, excluding core nodes and previously selected patients. This yielded48 pairs with no core-patient overlap. The nonuniform selection has no calibrated inclusion probabilities. The420 OFF-only residual pairs remain in full-queue statistics rather than being assigned to a false stratum. Expert candidates were the precommitted core indices0,2,5 (36pairs/72images/12patients).

原12个固定节点保持不变，全部六图序组成72核心配对。按预声明 flag 四层各选择12名互不重复患者，排除核心节点及此前入选患者，得到48对且与核心患者无重叠。该非均匀抽样缺少校准纳入概率；420个未被四层覆盖的 OFF-only 配对仍保留于全量统计。医生候选为核心预设图序0、2、5，共36对/72图/12患者。

All saved ON/OFF and13,200 native REAL views were read on CPU. Prior whole512 bilinear grayscale/Canny50-150/Hough votes50/minlength128/gap10 flags were reused unchanged. LOCALIZATION_DIAGNOSTIC_V1 retained candidate segments, approximate cross intersections and coordinates privately. Zones were outer26pixels, central[102:410), and the remainder. Candidates spanning zones were retained in all intersected zones; cross support was the union of both arms. These are positional zones, not thyroid ROIs.

CPU 读取全部既有 ON/OFF 及13,200个 REAL 原生 view；不改变原全图512双线性灰度、Canny50–150、Hough票数50/最短128/间隔10定义。新诊断定位器私有保存线段、近似十字交点和坐标。区域为外圈26像素、中心[102:410)及其余区域；跨区候选保留各区，十字按双臂并集。区域不是甲状腺 ROI。

Image values were averaged within six-image nodes, then within patients; REAL views were averaged within exams then patients. Paired ON-minus-OFF contrasts used2,000 percentile-bootstrap resamples of860 patients with identical paired patient weights (seed20261012). No confirmatory multiplicity-adjusted claim was made. Case-level human fields stayed NOT_REVIEWED; observer roles were not inferred from machine calculations.

生成图先在每节点六图内平均，再在患者内平均；REAL 先在 Exam 内再在患者内平均。ON−OFF 对比使用860患者同权配对的2,000次百分位 bootstrap，种子20261012；不作未经多重性校正的确证性推断。人工字段保持 NOT_REVIEWED，机器计算不自动获得人工角色。

