"""Always write a separately saved Chinese result with observed completion only."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import pandas as pd
from common import setup,config,jread,jout,RESULT,ARMS,SEEDS,now


def table(frame):
    rows=[list(frame.columns)]+frame.values.tolist()
    def cell(v):return 'NA' if pd.isna(v) else str(v).replace('|','\\|').replace('\n',' ')
    lines=['| '+' | '.join(cell(x) for x in r)+' |' for r in rows];lines.insert(1,'| '+' | '.join('---' for _ in frame.columns)+' |');return '\n'.join(lines)


def main(path):
    c=config(path);root=setup(c['project_private']);counts=pd.read_csv(root/'tables/REAL_COHORT_NATIVE_SUPPORT_V3.csv');current=jread(root/'releases/EXECUTION_STATE_V3.json')
    complete=[];progress=[]
    for arm in ARMS:
        for seed in SEEDS:
            run=root/'checkpoints'/arm/f'seed_{seed}';done=run/'RUN_COMPLETED.json';log=run/'train_log.csv'
            if done.exists():
                d=jread(done);complete.append(d);progress.append(dict(arm=arm,seed=seed,status=d['status'],best_epoch=d['best_epoch'],actual_epochs=d['actual_epochs'],ROC_AUC=d['metrics']['ROC_AUC'],AP=d['metrics']['AP'],reload=d['reload']))
            elif log.exists():
                d=pd.read_csv(log);best=d[d.selected_best.astype(bool)].iloc[-1];last=d.iloc[-1]
                progress.append(dict(arm=arm,seed=seed,status='RUNNING_OR_TRAINED_PENDING_RELOAD',best_epoch=int(best.epoch),actual_epochs=int(last.epoch),ROC_AUC=float(best.ROC_AUC),AP=float(best.AP),reload='NOT_RUN'))
            else:progress.append(dict(arm=arm,seed=seed,status='NOT_RUN_OR_STARTING',best_epoch=None,actual_epochs=None,ROC_AUC=None,AP=None,reload='NOT_RUN'))
    finalfile=root/'releases/FINAL_LTSA_TWO_ARM_STATUS_V3.json';final=jread(finalfile) if finalfile.exists() else None
    status=final['status'] if final else 'REAL_BASELINE_COMPLETED_OP_PENDING' if (root/'releases/REAL_ONLY_STAGE_COMPLETE_V3.json').exists() else 'PARTIAL_REAL_ONLY_TRAINING_RUNNING'
    if (root/'releases/ENGINEERING_INCOMPLETE_V3.json').exists() and not final:status='PARTIAL_ENGINEERING_INCOMPLETE'
    sections=[]
    def sec(title,text):sections.append('## '+title+'\n\n'+text+'\n')
    environment=jread(root/'audit/ENVIRONMENT_AND_WEIGHTS_V3.json');source=jread(root/'audit/LOCAL_SOURCE_SNAPSHOT_METADATA_V3.json');mapping=jread(root/'releases/ORIGINAL_PROGRESSION_LINEAGE_STATUS_V3.json')
    sec('1. 实际执行状态',f'状态：`{status}`；已独立重载完成 {len(complete)}/6 runs。当前阶段 `{current["phase"]}`。更新时间（北京时间）：{datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")}。\n\n尚未完成的训练、重载、集成、差值、统计图及Git必须保持待完成；不能把启动或烟测当正式结果。')
    sec('2. 本地源码与模型身份',f'权威源码 `LOCAL_PRIVATE_INPUT_REQUIRED_0 `upstream/LTSA_local_snapshot`，文件数{len(source["files"])}；来源无.git：`{source["version"]["revision"]}`，没有联网查询版本、clone/pull/download LTSA。\n\n原源码确有全局args、Tensor错误拆包/need_weights、cuda:0固定及原train.py删除输出/Test路径问题；原trainer未运行。timm和pytest不在当前环境，新wrapper从原文件AST加载精确PositionalEncoding类，torchvision离线同结构ResNet18严格载入官方缓存，标准库unittest执行匿名回归；没有伪装安装成功或修改共享依赖。称LTSA-based HT adapted predictor，不称未修改官方LTSA。')
    sec('3. 环境、权重与参数',f'Python{environment["python"].split()[0]}；{environment["versions"]}；CUDA runtime{environment["cuda_runtime"]}，GPU RTX5090/sm_120，驱动580.126.09。驱动报告CUDA13.0是能力上限，本次使用现有cu128 Torch并已实际forward/backward通过，不改系统CUDA/驱动。\n\nResNet18官方IMAGENET1K_V1本机权重，严格state_dict载入通过，无网络权重请求/随机初始化回退。端到端可训练参数12,755,009；每Exam所有view在同共享encoder后均值512，原固定sin/cos月索引TPE长度79，1层4头Transformer，FFN512/ReLU/postnorm/dropout0.25，因果+padding mask，最后真实历史token→dropout/Linear512→1 logit/BCE。无survival/censor/step-ahead/历史报告encoder。')
    sec('4. 冻结患者及全部原生支持',table(counts)+'\n\n同历史Train860(617/243)、Val189(131/58)患者/标签，不随机重分，不删除患者。Train4,585真实Exams/18,413native views，Val991/3,861，原生最长7次；按日期严格排序，全部保留。原CLIMAT固定K6/LOCF只选槽位，LTSA不把LOCF算新Exam，所以实际原生支持量不同。不同source/seed不会被计为新增患者。')
    sec('5. 目标年与Test隔离','只枚举Train/Val已确认患者的日期目录元数据；只有date.year<target_year的目录才读像素/缓存，目标年文件不打开，报告文本不读取。最终标签来自历史冻结target_label，不重新用影像诊断。模型字典仅images、Exam view counts、lengths、observed relative months；numeric sample_index只用于结果对齐，patient码/target_year/date/path/report不进入forward。\n\n没有读取Official Test影像、预测概率或标签清单，不运行原官方train.py。Train/Val本轮重新确认不重叠；与Official Test隔离依赖既有受保护拆分证据和只接收Train/Val的数据管道，并非新读取Test内容核对；匿名三集合与Test拒绝测试已执行。')
    sec('6. 唯一合成token协议',f'候选1,545节点/9,270图的patient/node/sixindices/seeds/path及原progression CPFM ON条件已逐条核验。历史条件严格prequery且pretarget、无报告/target标签/future truth，首锚点与原生首Exam相同。全部query为首个真实历史日期的周年+冻结annual index，不由目标真实日期回推。\n\n正式Val前冻结一条映射：仅{mapping["eligible_synthetic_nodes"]}个节点/{mapping["eligible_synthetic_images"]}图，时间严格在首末真实历史之间，6图→1个明确SYNTH_BOUNDED_ANNUAL_QUERY虚拟token。没有query与实际Exam同日期，故不把它冒充真实就诊。全部真实骨架保留，最终读出仍末真实Exam；597患者有合成、263无合成但都参与同一860人训练。758个目标年预测节点及101个末真实之后未知将来节点排除，不凑9,270张，不新增生成或其他臂。\n\n这是文档允许的明确来源/合法时点synthetic-token方案，不是同次真实Exam关联增强；与原CLIMAT全部9,270图固定K6协议不同。')
    sec('7. 训练、选模及恢复合同',f'臂仅{ARMS}；种子{SEEDS}，6个fresh process独立初始化，不用REAL已训权重warm-start OP。AdamW lr1e-4/wd1e-4；max60、warmup5+cosine、patience12/min15；有效患者batch8，microbatch{c["patient_microbatch"]}/累积{c["gradient_accumulation"]}，FP32，无TF32/Flash/memory-efficientSDPA。view分块16但全部进入可微图，不丢历史或views、不用旧预测器features。\n\n每患者每epoch只一个监督loss，末不足8人的optimizer组按实际人数归一化。Val AUC→AP→较低loss→较早epoch固定选模；没有性能门槛。仅完整epoch边界resume，保存model/optimizer/scheduler/early-stop和Python/NumPy/Torch CPU/CUDA/shuffle RNG；中断epoch回播。每best另一个进程strictload，与原189人同序概率容差1e-6；三seed按正类概率平均后算指标。')
    scheduling=root/'releases/GPU_SCHEDULING_POLICY_V3.json'
    if scheduling.exists():
        policy=jread(scheduling)
        sec('7.1 本次授权的共享GPU调度',f'用户明确同意并行接续后，调度模式改为`{policy["status"]}`：每次训练或独立重载之前连续2次确认GPU可查询、剩余显存至少{policy["min_free_mib"]/1024:g}GiB；不再要求其他计算进程数为零。只调整工程排队门槛，冻结模型、microbatch4/累积2、FP32、学习率、种子及数据合同未改。原生最大窗口烟测峰值5.621GiB，已完成OP seed42实际峰值4.526GiB；12GiB启动门槛留有额外空间，但不保证其他作业后续显存增长不会造成冲突。\n\n仅停止了本项目无训练子进程的等待调度器后重新接续，没有中断其他GPU任务；完成run跳过，OP seed42从独立重载开始，其他seed从已有合法epoch边界或独立初始化接续。CUDA/显存错误仍停止并报告，不无限重试。共享算力可能增加单epoch耗时，不把本次调整称为训练提速或结果完成。')
    sec('8. 实际run进度与结果',table(pd.DataFrame(progress))+'\n\n运行中行只是截至最近完整epoch的临时best，尚未重载/完成，不作为最终seed结果。以RUN_COMPLETED独立重载和ensemble产物作为完成证据。')
    for arm in ARMS:
        f=root/'tables'/f'{arm}_ENSEMBLE_METRICS_V3.csv'
        sec('9.'+('1' if arm==ARMS[0] else '2')+' '+arm+'三种子概率集成',table(pd.read_csv(f)) if f.exists() else 'NOT_RUN：3个种子/独立重载尚未全部完成，不能填推定ensemble分数。')
    diff=root/'tables/LTSA_OP_MINUS_REAL_PAIRED_BOOTSTRAP_V3.csv'
    sec('10. 两臂配对差值及评价边界',table(pd.read_csv(diff)) if diff.exists() else 'NOT_RUN：待两臂同189人概率齐全后执行2,000次患者配对bootstrap，种子20261013在训练前固定，报告OP−REAL的ROC-AUC/AP差值与95%CI。区间跨0不宣称显著改善。')
    sec('11. 阈值和旧模型参考','主表Youden J在各自Val上确定，另报0.5阈值。Accuracy/Precision/Sensitivity/Specificity/F1/TN/FP/FN/TP为历史内部Val指标，不是独立Test泛化。旧CLIMAT在文档中给出REAL0.763885、OP0.871808、ΔAUC约0.107923，仅历史参考，未新重算。主要比较同预测器内部OP−REAL；输入池化/时间骨架/合成过滤不同，不绝对排名或称完全同构。')
    sec('12. 工程验收与图文产物','15项匿名CPU测试PASS（含日期、pool、mask/padding、设备、接口、无step-ahead/无Test、有限梯度、合法合成、统计）；FP32真实最大窗口GPU micro1/2/4均forward/backward与2次AdamW更新PASS，micro4峰值5.621GiB。正式独立重载状态见每run，未做前不能填PASS。\n\n非医学结构/映射图及真实ROC/PR/seed/差值图由后续汇总程序生成；没有数据不画虚构曲线，自动导出不冒充实际视觉QA。论文方法/结果/局限均英段后紧跟中文，术语固定LTSA-HT adapted predictor、REAL Exam、bounded synthetic query token、AP、internal Val。')
    pf=jread(root/'audit/SOURCE_PROTECTION_POSTFLIGHT_V3.json')
    sec('13. 源保护、隐私与失败历史',f'实际源文件元数据保护{pf["unique_files_checked"]}个，状态{pf["status"]}；比较path/resolved/size/mtime/inode及LTSA目录清单，不做文件内容校验和。本地LTSA和旧B16/CLIMAT/原图不写入，训练只新根；新根权限700，患者/日期/标签/概率/权重/cache均私有。无医学图显示/下载。\n\n首次schema探测曾误将30个患者字典键编码返回到内部工具输出；已记录privacy incident，不能声称完全零身份编码工具输出。之后只输出字段/汇总，公开包不得带这些编码、私有路径/表或医学像素。其他错误/恢复记录在新日志与ENGINEERING_INCOMPLETE；未中断他人GPU作业。')
    gitfile=root/'releases/GIT_PUBLICATION_STATUS_V3.json';git=jread(gitfile) if gitfile.exists() else dict(status='PUSH_NOT_DONE',commit=None)
    sec('14. Git及私有公开分区',f'仓库Baone0330/Study，新子树 `B16v1_Progression/experiments/historical_six_branch_checkpoint_replay_v1/downstream/ltsa_ht_two_arm_v3_local_source/`。实际状态 `{git["status"]}`；commit `{git.get("commit")}`。\n\n仅安全代码/配置模板/匿名tests/汇总图表/英汉材料可归档；患者ID/日期/报告/像素/逐人标签概率/权重/private manifest不发布。不改旧分支；没有实际push+remote核验前不写PUSH_VERIFIED。')
    sec('15. 服务端绝对路径',f'实验：`{root}`\n\n日志：`{root}/logs`；模型：`{root}/checkpoints`；私有结果：`{root}/metrics/private`；汇总：`{root}/tables`；图：`{root}/figures/nonmedical`；源码证据：`{root}/audit`。最终代码/参数和身份锚点均在新根，不下载来源版本。')
    sec('运行结果 Markdown 文档',f'服务端实际路径：`{root/RESULT}`。\n\n中文摘要：完成本地源码/离线权重/原生历史/合成来源审核、匿名和GPU烟测；正式结果仅按实际run/ensemble记录，未完成保持待完成。客户端下载状态：`NOT_VERIFIED_FOR_THIS_SNAPSHOT`（仅实际传输核验后更新交付证据）。后台后续生成的最终版本需再次交付，不能把当前进行中本地副本当最终。')
    (root/RESULT).write_text('# LTSA-HT V3 本地源码双分支实验结果\n\n'+'\n'.join(sections),encoding='utf-8')
    jout(root/'releases/RESULT_SNAPSHOT_STATUS_V3.json',dict(status=status,completed_verified_runs=len(complete),target_runs=6,current_phase=current['phase'],server_result_private=str(root/RESULT),client_delivery='NOT_VERIFIED_FOR_THIS_SNAPSHOT',updated_utc=now()))
    print({'result_status':status,'completed_verified_runs':len(complete),'sections':len(sections)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();main(a.config)
