"""Standalone Chinese status/result Markdown, only measured aggregates, no patient content."""
import argparse
from pathlib import Path
import pandas as pd
from common import config,setup,jread,jout,ARMS,SEEDS,RESULT,now
def table(d,cols=None):
    if cols:d=d[cols]
    def fmt(v):return f'{v:.8g}' if isinstance(v,float) else str(v)
    return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in d.itertuples(index=False,name=None))+'\n'
def readtable(root,name,cols=None):
    p=root/'tables'/name;return table(pd.read_csv(p),cols) if p.exists() else '尚未完成，不预填结果。\n'
def ensemble_table(root,fixed=False):
    p=root/'tables/V5_THREE_SEED_ENSEMBLE.csv'
    if not p.exists():return '尚未完成，不预填结果。\n'
    d=pd.read_csv(p).set_index('arm')
    names=['ROC_AUC','AP','Accuracy','Precision','Sensitivity','Specificity','F1','threshold','TN','FP','FN','TP']
    fields=[('fixed05_'+n) if fixed else n for n in names]
    d=d[fields].T.reset_index().rename(columns={'index':'指标',ARMS[0]:'REAL-only',ARMS[1]:'OP特征融合'})
    d['指标']=names
    return table(d)
def main(path):
    c=config(path);root=setup(c['project_private']);final=jread(root/'releases/FINAL_V5_STATUS.json') if (root/'releases/FINAL_V5_STATUS.json').exists() else {}
    runs=[];completed=0
    for arm in ARMS:
        for seed in SEEDS:
            run=root/'checkpoints'/arm/f'seed_{seed}';done=run/'RUN_COMPLETED.json';init=run/'RUN_INITIALIZATION_V5.json';log=run/'train_log.csv'
            row=dict(arm=arm,seed=seed,status='未启动',last_epoch='—',best_epoch='—',ROC_AUC='—',AP='—')
            if done.exists():
                d=jread(done);completed+=1;row.update(status='独立重载PASS',last_epoch=d['actual_epochs'],best_epoch=d['best_epoch'],ROC_AUC=d['metrics']['ROC_AUC'],AP=d['metrics']['AP'])
            elif init.exists():
                row['status']='运行中/以实时进程为准'
                if log.exists():
                    hist=pd.read_csv(log);last=hist.iloc[-1];row.update(last_epoch=int(last.epoch),ROC_AUC=float(last.ROC_AUC),AP=float(last.AP))
            runs.append(row)
    smoke=jread(root/'audit/GPU_PREFLIGHT_V5.json') if (root/'audit/GPU_PREFLIGHT_V5.json').exists() else {}
    anon=jread(root/'tests/ANONYMOUS_CPU_TEST_RESULTS_V5.json') if (root/'tests/ANONYMOUS_CPU_TEST_RESULTS_V5.json').exists() else {}
    env=jread(root/'audit/ENVIRONMENT_V5.json') if (root/'audit/ENVIRONMENT_V5.json').exists() else {}
    protect=jread(root/'audit/SOURCE_PROTECTION_POSTFLIGHT_V5.json') if (root/'audit/SOURCE_PROTECTION_POSTFLIGHT_V5.json').exists() else {}
    auto=jread(root/'figures/export_qa/V5_EXPORT_QA.json') if (root/'figures/export_qa/V5_EXPORT_QA.json').exists() else {}
    review=jread(root/'figures/export_qa/V5_ACTUAL_VISUAL_REVIEW.json') if (root/'figures/export_qa/V5_ACTUAL_VISUAL_REVIEW.json').exists() else {}
    git=jread(root/'releases/GITHUB_STATUS_V5.json') if (root/'releases/GITHUB_STATUS_V5.json').exists() else {}
    failure=root/'releases/ENGINEERING_INCOMPLETE_V5.json'
    conclusion='训练未结束，Q1增益、Q2跨协议趋势、Q3 REAL数值一致性均待实测。'
    if final.get('RUNS_AND_METRICS_COMPLETE'):
        boot=pd.read_csv(root/'tables/V5_PAIR_BOOTSTRAP.csv');a=boot[boot.metric.eq('ROC_AUC')].iloc[0];ap=boot[boot.metric.eq('AP')].iloc[0]
        parity=pd.read_csv(root/'tables/V5_REAL_NUMERICAL_PARITY_DIAGNOSTIC.csv')
        conclusion=f"Q1：OP−REAL ROC-AUC={a.delta_OP_minus_REAL:.6f}，95%CI [{a.ci_low:.6f}, {a.ci_high:.6f}]；AP={ap.delta_OP_minus_REAL:.6f}，95%CI [{ap.ci_low:.6f}, {ap.ci_high:.6f}]。AUC点估计提高、AP略降，两者区间均跨0，尚不能确认稳定增益；不调参择优。Q2：相对V3/V4的−0.068834 AUC、−0.112441 AP，增强效应分别改善约0.080416/0.104327，但只能作跨协议描述，不能唯一归因于token数。Q3：三个seed及集成对齐V3/V4的八项REAL概率比较，最大绝对差为{parity.max_abs_probability_difference.max():.8g}，属于工程数值一致性证据，不是性能Gate。"
    recovery=jread(root/'releases/POSTPROCESS_RECOVERY_V5.json') if (root/'releases/POSTPROCESS_RECOVERY_V5.json').exists() else {}
    sections=[f'# LTSA-HT V5 检查级特征融合双臂实验结果\n\n更新时间（UTC）：{now()}；正式完成并重载：{completed}/6。\n\n本文件记录实际执行结果；训练/统计、交付和Git远端核验状态分别列示。\n',
       '## 1. 任务、环境与三项独立状态\n\n'+f"RUNS_AND_METRICS_COMPLETE：{final.get('RUNS_AND_METRICS_COMPLETE',False)}；DELIVERABLES_COMPLETE：{final.get('DELIVERABLES_COMPLETE',False)}；GITHUB_PUSH_VERIFIED：{git.get('status')=='GITHUB_PUSH_VERIFIED'}。\n\n任务：`{c['task']}`。服务器5090-lin；新输出根：`{root}`。V3/V4和原LTSA只读，源码Git修订不可得时标记UNKNOWN。\n\n"+json_safe(env),
       '## 2. 固定真实/合成支持统计\n\n'+readtable(root,'FEATURE_FUSION_INPUT_COUNTS_V5.csv',['split','patients','HT0','HT1','REAL_exams','REAL_views','synthetic_nodes','synthetic_views'])+'\n'+readtable(root,'FEATURE_FUSION_INPUT_COUNTS_V5.csv',['split','patients_with_synthetic','patients_without_synthetic','attached_REAL_exams','duplicate_bindings','missing_nodes']),
       '## 3. next-real映射与不变的真实时间轴\n\n686节点在读取像素前冻结到第一个严格晚于query的真实Exam；没有新增时序token。618个真实Exam附着节点：550个有一个，68个有两个；其余3967个Train真实Exam没有。重复绑定/遗漏均0。真实长度、月份、padding、causal和最后REAL索引在两臂相同。\n\n'+readtable(root,'V5_NODE_TO_REAL_EXAM_AGGREGATE.csv'),
       '## 4. 同构模型、梯度及最大窗口测试\n\n两臂均为共享ResNet18→真实Exam全view均值→可选残差融合→原LTSA月TPE→单层四头Transformer→最后REAL→二分类头。两个独立LayerNorm512、门控1024→1、投影512→512，固定残差系数0.1；节点先六图均值再节点等权均值。无节点严格恒等旁路。新模块初始化保存/恢复Python、NumPy、Torch CPU和已初始化CUDA RNG；标准训练BN不冻结。\n\n'+f"匿名测试：{anon.get('status','待执行')}，{anon.get('tests',0)}项。实际GPU：{smoke.get('status','待执行')}，参数{smoke.get('trainable_parameters','待测')}，额外fusion参数{smoke.get('new_fusion_parameters','待测')}；2步AdamW峰值{smoke.get('peak_allocated_GiB','待测')}GiB；真实views {smoke.get('real_views','待测')}、合成views {smoke.get('synthetic_views','待测')}，smoke权重不用于训练。\n",
       '## 5. 六次fresh训练及每seed结果\n\n'+readtable(root,'V5_SEED_LEVEL_METRICS.csv',['arm','seed','best_epoch','actual_epochs','ROC_AUC','AP','F1','reload','reload_max_abs'])+'\n所有seed完整Accuracy/Precision/Sensitivity/Specificity和混淆矩阵亦保留在同根tables/V5_SEED_LEVEL_METRICS.csv，未省略负向seed。六个初始化、训练和strict重载进程独立。\n',
       '## 6. 三seed概率均值集成\n\n内部Val Youden阈值：\n\n'+ensemble_table(root)+'\n固定0.5敏感性：\n\n'+ensemble_table(root,True)+'\n同189人、58阳性/131阴性；AP为average_precision_score，不是梯形PR面积。Youden阈值来自内部Val。\n',
       '## 7. 患者配对bootstrap\n\n'+readtable(root,'V5_PAIR_BOOTSTRAP.csv')+'\n固定seed20261013、2000次重采样；单类样本跳过并报告。\n',
       '## 8. V3/V4/V5六行描述性比较\n\n'+readtable(root,'V3_V4_V5_CROSS_PROTOCOL_COMPARISON.csv',['version','input_protocol','arm','ROC_AUC','AP'])+'\n旧协议增加虚拟query token；新协议只在后续真实Exam做残差融合。gate、编码顺序与BN也可能影响差异，不能唯一归因于时间token。\n\n'+readtable(root,'V5_REAL_NUMERICAL_PARITY_DIAGNOSTIC.csv'),
       '## 9. 图件、文字与实际复核\n\n'+f"四组非医学图：Fig01_V5_ROC_PR、Fig02_V3_V4_V5_Descriptive_Comparison、Fig03_V5_Seeds_and_Ensemble、Fig04_V5_Exam_Fusion_Internal_Architecture。PNG600dpi及可编辑SVG保存于同根figures/nonmedical；统计预览不保存本地。导出自动QA：{auto.get('status','尚未生成')}；实际视觉复核：{review.get('status','NOT_RUN')}。已实际复核才填写PASS，未做医生复核/作者批准或期刊投稿认证。五份writing/LTSA_V5_*_CN_EN.md为Methods、Results、Discussion/Limitations、Table/Figure Captions、Integration with CLIMAT，每英文段后对应中文。作者审阅仍待完成。\n",
       '## 10. 工程中断/等待/恢复\n\n首次启动在第一批数据进入前遇到AF_UNIX通信路径过长，无epoch/Val/优化器更新，非CUDA故障；FAILED_IPC_ATTEMPT日志与初始化保留。修复仅设V5内短相对IPC路径，原2workers/batch/随机数不变，实际完整双worker Train-only测试通过。\n\n六次训练和统计完成后，Fig02额外1.2刻度标签触发画布边界检查，后处理暂停；修复固定0–1刻度并错开数值标签，同时补充结构图raw REAL skip、图注与双语损失说明，未降低校验标准、未重训。\n\n'+f"后处理恢复状态：{recovery.get('status','IN_PROGRESS')}；原失败标记/诊断/日志保留为历史，不冒充无失败。完成权重、逐人预测、统计、冻结配置及科学代码元数据核对：{recovery.get('numerical_artifacts_protection','待最终核对')}。"+('\n\n仍有活动工程未完成标记，请检查同根日志。\n' if failure.exists() else '\n\n无当前活动工程未完成标记。\n'),
       '## 11. 源保护与禁项复核\n\n'+f"元数据postflight：{protect.get('status','待检查')}，检查{protect.get('unique_files_checked',0)}个文件；路径、解析路径、size、mtime_ns、inode，不是内容相等证明。新SD采样/医学合成图0（只导出非医学图）；新上游训练0；Official Test内容访问0；目标年度真实影像/报告不进入输入。私有逐人数据只留服务器，不下载、不发布；未执行文件内容哈希。Python open-event审计不是OS级全面证明。\n",
       '## 12. GitHub真实状态\n\n'+f"状态：{git.get('status','NOT_ATTEMPTED')}；commit：{git.get('commit','未提交')}；远端核验：{git.get('remote_verified',False)}。公开目录为B16v1_Progression/experiments/historical_six_branch_checkpoint_replay_v1/downstream/ltsa_ht_v5_exam_feature_fusion，仅新增匿名代码/模板/聚合结果和非医学图，不覆盖V3/V4；无逐人数据、医学像素、权重和凭据。\n",
       '## 13. 结论与边界\n\n'+conclusion+'\n\n同一189患者的探索性内部Val，已参与选模和新方案设计；bootstrap不覆盖适应性方案、早停和阈值选择的全部不确定性。结果不能代表Official Test或独立临床泛化，也不能推广到所有生成模型。CLIMATv2 fixedK6只作历史跨预测器描述，不假定输入协议等价。\n']
    (root/RESULT).write_text('\n'.join(sections),encoding='utf-8')
    print(dict(result_file=str(root/RESULT),completed_runs=completed,metrics_complete=final.get('RUNS_AND_METRICS_COMPLETE',False)),flush=True)
def json_safe(value):
    import json
    return '```json\n'+json.dumps(value,ensure_ascii=False,indent=2)+'\n```\n'
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);main(p.parse_args().config)
