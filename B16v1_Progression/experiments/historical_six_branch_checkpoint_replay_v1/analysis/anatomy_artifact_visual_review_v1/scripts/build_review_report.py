"""Standalone truthful Chinese report, plus separate path-free public release summary."""
import argparse
from pathlib import Path
import pandas as pd
from review_common import setup,read_json,json_out,RESULT_NAME,now


def markdown(frame,index=False):
    f=frame.reset_index() if index else frame
    def cell(v):
        if pd.isna(v):return 'NA'
        return str(v).replace('|','\\|').replace('\n',' ')
    rows=[list(f.columns)]+f.values.tolist()
    lines=['| '+' | '.join(cell(v) for v in row)+' |' for row in rows]
    lines.insert(1,'| '+' | '.join('---' for _ in f.columns)+' |')
    return '\n'.join(lines)


def main(output_root,delivery=False):
    root=setup(output_root);c=read_json(root/'protocol/FROZEN_VISUAL_AND_ARTIFACT_REVIEW_CONTRACT_V1.json')
    loc=read_json(root/'release/LOCALIZATION_COMPLETE_V1.json');prep=read_json(root/'release/PRIVATE_PANEL_PREPARATION_COMPLETE_V1.json')
    pf=read_json(root/'source_audit/ORIGINAL_SOURCE_POSTFLIGHT_V1.json');ci=pd.read_csv(root/'metrics/PAIR_CLUSTER_BOOTSTRAP_V1.csv').set_index('metric')
    reproduction=pd.read_csv(root/'metrics/ORIGINAL_FLAG_FULL_QUEUE_REPRODUCTION_V1.csv');cov=pd.read_csv(root/'metrics/REVIEW_SAMPLE_COVERAGE_AND_SELECTION_V1.csv')
    region=pd.read_csv(root/'metrics/FULL_QUEUE_REGION_LEVEL_AGGREGATES_V1.csv');qa=pd.read_csv(root/'figures/export_qa/NONMEDICAL_FIGURE_EXPORT_QA_V1.csv')
    tests=read_json(root/'tests/ANONYMOUS_TEST_RESULTS_V1.json');gitfile=root/'release/GIT_PUBLICATION_STATUS_V1.json'
    git=read_json(gitfile) if gitfile.exists() else dict(status='PUSH_NOT_DONE',commit=None,files=0,remote_verified=False)
    visual=qa.actual_nonmedical_agent_visual_review.eq('PASS_AFTER_ACTUAL_VIEW').all();assert len(qa)==6 and tests['status']=='PASS'
    final='ANATOMY_ARTIFACT_REVIEW_ENGINEERING_COMPLETED_WITH_LIMITATIONS' if visual and pf['status']=='PASS' and git['status']=='PUSH_VERIFIED' else 'ENGINEERING_READY_WITH_LIMITATIONS_GIT_OR_VISUAL_PENDING'
    l=ci.loc['line_candidates'];x=ci.loc['cross_candidates']
    sections=[]
    def sec(title,text):sections.append('## '+title+'\n\n'+text+'\n')
    sec('1. 实际执行状态',f'状态：`{final}`。更新时间：{now()}。\n\n本轮完成离线技术复核工程与候审材料，不是医学结构认证。医学图像未在会话查看，真实人工/医生观察均为0，医学生物学结论仍未建立。Git 状态独立记录为 `{git["status"]}`。')
    sec('2. 指令与严格边界','按附件执行；只读继承旧质量目录和历史 Original Progression 消融。新 Stable Diffusion 采样=0、新模型/CLIMATv2训练=0、新 FID/KID=0、新 AUC评价=0、Official Test 内容访问=0。没有按标签、诊断、临床报告或图像好看程度筛选。任务内没有引入新分割模型、改动驱动/CUDA、其他用户环境或 Git 全局配置。')
    sec('3. 权威模型与配对身份','模型为历史 Original Progression 20260921 epoch120，原 B14 CPFM；不是 S2、Clean 或 Strict 变体。ON/OFF 以权威患者/节点、slot、图序、master-seed逐对验证。全部9,270对、每臂1,545节点/860患者、每节点6图序0–5。Full/Plain只为既有路线辅助对照，不是同一 CPFM 单开关因果效应。上轮仅36个固定 pilot ON 重采样有像素一致证据，本轮没有重采样全量。')
    sec('4. 源资产与本轮读取计数','旧五组累计50,280图 QA 仅作为源证据引用。本轮定位实际读取/解码 ON9,270、OFF9,270、REAL13,200，合计31,740图，不能把旧50,280称为本轮新解码数。REAL 为3,154个原生 Exam、859患者；ON/OFF完整配对860患者分母不因REAL少1人而改变。Full/Plain各9,270图仅既有辅助；本轮候审核心各实际读取72图。')
    sec('5. 冻结复核样本与覆盖','原固定12节点不替换，前6主图/后6补充，全部六图序组成72对/144图/12患者。四层各12对，实际48对/96图/48不同患者，与核心患者及配对重叠=0。总核心+分层120对/240图/60患者。医生候选只取核心图序0、2、5，为36对/72卡/12患者，是核心子集不能另加为新患者或独立图。\n\n'+markdown(cov[['stratum','all_queue_pairs','eligible_noncore_pairs','target_pairs','selected_pairs','selected_patients','shortfall']])+'\n\n四层覆盖不足：另有420对 OFF-only 未分层，完整保留全量统计；未强制归S4，未剔除。随机排序种子20261012，优先未用患者，抽样分数不是实际纳入概率，不将48对结果外推总体发生率。')
    sec('6. 原flag复现与新增定位器',f'本轮31,740张旧长线/十字flag及原线段数逐图匹配，不一致={loc["old_flag_mismatches"]}。CPU6进程、OpenCV单线程；未调用GPU。\n\n'+markdown(reproduction)+'\n\n原定义不变：RGB灰度0.299/0.587/0.114，全图512双线性，Canny50/150，Hough rho1/thetaπ/180/票数50/minLength128/maxGap10，水平/垂直容差10°，双线平均坐标与投影范围的近似交点。新 `LOCALIZATION_DIAGNOSTIC_V1` 追加位置，不替换旧flag、不覆盖旧比例。')
    selected=region[region.metric.isin(['line_border_any','line_central_any','line_intermediate_any','line_multizone_any','cross_border_any','cross_central_any','cross_multizone_any'])][['branch','metric','images','patients','image_mean','patient_mean']]
    sec('7. 几何区域及实际候选分布','外圈26像素、中心[102:410)×[102:410)、其余为中间区域；不是甲状腺 ROI、病灶或毫米坐标。线段以≤1像素间隔取点，十字区域为双臂并集，交点位置单独私有记录。跨区候选保留各区并额外标 MULTIZONE；各区域比例可能重叠不相加为100%。\n\n'+markdown(selected))
    sec('8. 患者配对效应与不确定性',f'每节点六图平均，再患者内节点平均，配对 ON−OFF 同患者同权、860人、2,000次百分位 bootstrap、种子20261012。22项技术对比是探索性，未作多重性校正确证。\n\n线段患者等权均值 ON={l.on_patient_mean:.5f}、OFF={l.off_patient_mean:.5f}，差值={l.delta_on_minus_off:.5f}，95%CI[{l.ci_low:.5f}, {l.ci_high:.5f}]；十字差值={x.delta_on_minus_off:.6f}，95%CI[{x.ci_low:.6f}, {x.ci_high:.6f}]。正值表示检测器响应更多，不代表质量改善或已证实更多医学伪影。\n\n'+markdown(ci.loc[['old_long_line_flag','old_cross_proxy_flag','line_candidates','cross_candidates'],['on_patient_mean','off_patient_mean','delta_on_minus_off','ci_low','ci_high']],index=True))
    sec('9. 人工观察、结构可视性与未知项','自动技术定位已执行；NONCLINICAL_HUMAN、CODEX医学视觉和CLINICIAN均NOT_RUN，真实观察者/实际完整医学图查阅/实际评分=0。240行空白非临床表的可视性字段NOT_REVIEWED；实际未观察不写UNCERTAIN。TP/FP/TN/FN、敏感度、特异度、精确率、Kappa、ICC为空/NA，不能当0。可能UI、采集标记、真实组织界面或生成伪影只是待验证解释，没有逐病例已观察分类。')
    sec('10. 私有候审图板与医生准备',f'实际新医疗图板{prep["medical_panel_PNGs"]}个PNG：A12、B12、C48、D12；另有{prep["blind_card_PNGs"]}张无语义随机医生卡。A原12节点全部6图序；B严格匹配Full/Plain，标route-only；C原图与技术候选叠加并列，全部48对保留；D统一差值0–255和事前中心[192:320)裁片，非ROI。REAL面板E=WITHHELD，无REAL像素图板。\n\n五项1–5分医生模板高分方向已写明，几何严重度高分更严重，其余高分较有利；允许UNSURE/NOT_ASSESSABLE/SKIPPED且不当0。实际无医生、无填写分数、无量表训练及非患者示例。72卡单图随机呈现，映射单独私有；去标签不保证感知盲法，当前界面未激活，没有启动HTTP服务。')
    sec('11. 隐私、脱敏和显示审批','医学像素外传=0，聊天/外部视觉模型/API展示=0，本地医学/诊断图保存=0。新根及映射/评分/医学图目录700，私有图片600。生成图也按潜在敏感医学影像处理；未自动遮挡原图。PHI审核PENDING、REAL显示WITHHELD、公开医学图许可NOT_APPROVED。存服务器不等于脱敏；所有图板是候审，不是可直接发表。')
    sec('12. 七表、六图与视觉QA',f'7张汇总表、6组非医学图各PNG/SVG/PDF（18导出）、6份源数据CSV、6份英汉图注、5份英汉材料及术语表。PNG为RGB600dpi，SVG可编辑文字无栅格嵌入，PDF嵌入TrueType；实际draw文字边界与≥8pt检查PASS。6组非医学图真实会话目检状态：{"PASS_AFTER_ACTUAL_VIEW" if visual else "NOT_RUN"}；这不是医学影像复核，未本地保存图片。\n\n'+markdown(qa[['figure','dimensions','dpi','auto_export','actual_nonmedical_agent_visual_review']]))
    sec('13. 论文表述与负面结果','上轮自然图像特征距离ON较小；本轮ON线段/十字候选更多。两者测量不同问题，不构成医学结构质量已更好的证据。本轮不能说OFF更自然或甲状腺更清晰，因为没有真实看图。讨论明确域偏差、重采样/采集混杂、近似交点、多区重复、抽样权重缺失、医生/ROI/目标年真值缺位。旧Fig01–08保持，新增用Qual-Fig01–06、Qual-Table01–07避免覆盖。')
    sec('14. 验证、失败与恢复',f'匿名合同自检{tests["tests_run"]}项PASS；代码语法检查PASS。首轮启动安全查询因命令行自身匹配而阻止启动；随后实际进程核查无现存任务，使用单实例flock启动，未重复生成。定位全部成功，无源解码或flag失败。绘图初次Fig02标题超出画布被自动阻止；只修标题换行，原数据/阈值/样本不变，并保存旧脚本时间戳备份。Markdown初次因可选tabulate缺失退出，改用原生Markdown表格写入，不安装或修改共享环境。发布逐值扫描较慢，改为相同字面值/边界规则的一次编译前缀树；实际私有值检查不减少。仅停止自身prepare-only父进程后，发现网络clone子进程仍在下载、重试工作区缺少HEAD；随后只停止本任务clone及其index-pack，保留不完整目录，改为从已核实基线的本机上一轮Git只读副本复制新的独立checkout，不用硬链接，仍核验remote/main。候选定位有256图分块账本，同冻结配置resume；不自动改阈值填缺失。')
    sec('15. 源保护与禁止项复核',f'实际对{pf["unique_files_actually_checked"]:,}个唯一源文件逐个比较path/resolved_path/size/mtime_ns/inode，状态{pf["status"]}，变化数{len(pf["changed_paths_private"])}。涵盖原图、权重、manifest、split/locks、旧质量材料及历史源依赖；这是元数据保护检查，不冒充内容校验认证。没有运行或报告文件哈希。当前任务脚本不打开Official Test内容、不加载Torch/训练代码、不改上游。禁止项依据实际调用与保护范围核对，不作系统全盘监控保证。')
    sec('16. GitHub归档真实状态',f'仓库 Baone0330/Study；新子树 `B16v1_Progression/experiments/historical_six_branch_checkpoint_replay_v1/analysis/anatomy_artifact_visual_review_v1/`。\n\n状态 `{git["status"]}`；commit `{git.get("commit")}`；新文件数 {git.get("files",0)}；push returncode `{git.get("push_returncode","NOT_RUN")}`；remote/main核验={git.get("remote_verified",False)}；原前缀字节未变文件数={git.get("old_prefix_files_byte_unchanged","NOT_RUN")}。\n\n仅公开安全代码、匿名toy、通用协议、汇总表、非医学图、英汉材料与无病例编码空表头。无医学PNG、PHI、患者/病例映射、真实评分、权重或训练日志。使用Git协议验证，不冒充GitHub API。作者仅checkout本地配置study <l3026714004@outlook.com>，常规main push无force。')
    paths=[('结果文档',root/RESULT_NAME),('候选位置私有表',root/'metrics/FULL_QUEUE_FLAG_LOCATION_DIAGNOSTICS_V1.parquet'),('患者配对区间',root/'metrics/PAIR_CLUSTER_BOOTSTRAP_V1.csv'),('图板',root/'figures/private_medical_panels'),('医生候审卡',root/'figures/private_expert_review_cards'),('汇总表',root/'tables'),('非医学图',root/'figures/public_nonmedical'),('英汉材料',root/'writing'),('私有表',root/'reviews_private'),('安全发布包',root/'github_public_stage'),('发布状态',root/'release')]
    sec('17. 服务端产物绝对路径','\n'.join(f'- {name}：`{path}`' for name,path in paths))
    sec('18. 未完成项及后续条件','必须另行提供真实机构授权/PHI脱敏确认、本地完整图查阅记录、实际非临床或具备甲状腺超声经验医生、量表培训/非患者示例及适用ROI参考，才可开展医学观察和评分。两名真实医生才可评分者间一致性；只有一名只报告单人主观描述。此工程结束不需要虚构医学结论，也不自动启动下一训练、采样或新Test评价。')
    sec('运行结果 Markdown 文档',f'实际服务端路径：`{root/RESULT_NAME}`。\n\n服务器Git：`{git["status"]}`，隔离checkout工作区状态 `{git.get("working_tree_clean","NOT_VERIFIED")}`。客户端中文Markdown交付状态：`{"VERIFIED_LOCAL_FILE_RECEIVED" if delivery else "NOT_VERIFIED"}`。仅实际文件已收到后才更新为已交付；未下载任何医学图片。')
    text='# B16v1 历史 Original Progression 结构可视性与伪影专项复核结果\n\n'+'\n'.join(sections)
    (root/RESULT_NAME).write_text(text,encoding='utf-8')
    public='# Public engineering result / 工程结果\n\nEngineering completed with medical-review limitations. Publication provenance is the repository commit history, not a self-referential claim in this file.\n\n'+('\n'.join(s for s in sections if not any(s.startswith('## '+n) for n in ['1.','16.','17.','运行结果'])))
    # Separate public provenance avoids self-referential commit content and excludes private absolute paths.
    (root/'writing/PUBLIC_REVIEW_RESULT_V1.md').write_text(public,encoding='utf-8')
    json_out(root/'release/FINAL_ANATOMY_ARTIFACT_REVIEW_STATUS_V1.json',dict(FINAL_STATUS=final,actual_source_images_decoded=31740,old_flag_mismatches=0,
       medical_panels=84,blind_cards=72,actual_medical_observers=0,CLINICIAN_REVIEW='NOT_RUN',REAL_VISUAL_REFERENCE='WITHHELD',PHI_REVIEW='PENDING',
       nonmedical_visual_QA='PASS_AFTER_ACTUAL_VIEW' if visual else 'NOT_RUN',patient_bootstrap_repeats=2000,primary_patients=860,
       test_access=0,new_training=0,new_SD_sampling=0,new_FID_KID=0,medical_pixels_transferred=0,source_protection=pf['status'],
       git_status=git['status'],git_commit=git.get('commit'),client_delivery='VERIFIED_LOCAL_FILE_RECEIVED' if delivery else 'NOT_VERIFIED',updated_utc=now()))
    print({'status':final,'report_sections':len(sections),'medical_review':'NOT_RUN','git_status':git['status'],'delivery':delivery})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',required=True);p.add_argument('--delivery-verified',action='store_true');a=p.parse_args();main(a.output_root,a.delivery_verified)
