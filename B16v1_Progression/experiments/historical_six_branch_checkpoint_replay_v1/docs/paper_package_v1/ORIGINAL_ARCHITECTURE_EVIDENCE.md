# 历史实际模块 DAG 与源锚点

身份：HISTORICAL_SIX_BRANCH / ORIGINAL_PROGRESSION_20260921。没有本轮训练、生成或临床forward。

## 生成路径

REAL查询前历史图像 → image_feature[8] + report zeros[6] → BEFORE/AFTER独立ExamSetEncoder(14→32 LN GELU→16) → view/set mean。
query numeric[9] → Linear9→24 LN GELU→16。
Full：concat[b16,a16,q16] →48→32 LN GELU→16 tanh。
Original P16：每次历史检查AFTER e16→相邻delta16→16→32 LN GELU→16→masked mean p16；concat[b,a,q,p]64→128 LN GELU→64→Linear64→16 tanh。
P16 adapter ≠ B14 CPFM。原bridge condition_projection在P16路由被替换；scale被计算但未被hook消费；attribute/proxy未被生成消费。
c16 →B14：magnitude32→128→128；direction64→128→128；patient projection32→32→16与patient direction64→64；c与zero-c方向差 + .25 patient residual→RMS normalize→bounded gamma/beta。
history summary=zero16；current summary=四项UNet激活统计重复四次。方向raw gamma/beta heads的旧输出丢弃，不是effective最终方向。
UNet.down_blocks[i].resnets[0]输入pre-hook，i0–3通道320/320/640/1280；空间归一化u后 y=x+sigmoid(layer_scale)*(((1+gamma)*u+beta)-u)。无up-path B14 hook。
冻结SD1.5 + 医学LoRA + 通用text prompt，PNDM epsilon/30step/guidance7，VAE decoder→既有离线合成图。以上是原历史阶段关系，不在本轮运行。

## 下游路径

RGB256图像 [B,6,V,3,256,256]→有效view共享ResNet18 stem+BasicBlock[2,2,2,2]→每图512x8x8→masked view mean→exam512x8x8→64 patches512。
Radiologist FeaT512(depth1,heads4,FFN256)，元数据zero11→Linear11,64 ReLU LN64→GP FeaT64(depth1,heads4,FFN64)。常量GP网络活动，不读取report/time内容。
Radiologist states[65,512] + repeated GP CLS[65,64]→[65,576]→原exam prognosis FeaT576。
每exam的Radiologist CLS512+GP CLS64组成visit token576；6 slots+wrapper CLS+learned position→masked原prognosis Transformer(depth1,heads4,FFN256)，CLS→LN576→Linear576,288 GELU→Linear288,5→wrapper Linear5,2→softmax HT概率。
序列prognosis复用原模块权重，不是第二个独立训练Transformer。ResNet标准GAP/1000类FC不活动；image projection Identity。序列位置表容量7（+CLS为8），实际T=6。
三个原训练seed概率→患者键配对等权平均→旧Val原阈值策略/ROC-AUC/AP。该聚合在model输出之后。

## 证据与未知

逐节点file/class/function/line见SOURCE_LINE_ANCHORS和Table04。F16/P16/B14/一个完整下游对象CPU strict load通过，只用于schema确认，forward=0。18映射均有前阶段独立新forward QA。
原生成时安装diffusers精确revision UNVERIFIED_FROM_SOURCE；目前库仅支持结构接口说明。医学LoRA逐层rank未审计；没有完整SD/LoRA权重加载。未重新评估临床生成图质量、医学真实性或外部Test。未创建权重第二备份。
