"""Eight nonmedical aggregate figures from measured CSVs, editable vectors, export QA."""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from b16_quality_common import setup, read_json, atomic_json, write_csv, now

BRANCHES=['REAL_HISTORY_REFERENCE','OP_CPFM_ON','OP_CPFM_OFF','FULL_B16V1_HIST','PLAIN_SD_HIST']
LABELS=dict(zip(BRANCHES,['Real history','OP ON','CPFM OFF','Full (aux.)','Plain SD (aux.)']))
COLORS=dict(zip(BRANCHES,['#626D7A','#197F86','#D88030','#8874A7','#5594BE']))


def make_tables(root):
    tables=root/'tables';tables.mkdir(exist_ok=True)
    src=root/'figures/source_data'
    mapping={
        'Table01_Branch_Support_V1.csv':'manifests/COHORT_COUNTS_AND_PATIENT_OVERLAP_V1.csv',
        'Table03_Distribution_Distances_V1.csv':'metrics/QUALITY_DISTRIBUTION_METRICS_V1.csv',
        'Table04_Paired_Changes_V1.csv':'metrics/PAIRED_CHANGE_AGGREGATES_V1.csv',
        'Table05_Technical_Proxies_V1.csv':'metrics/TEXTURE_AND_ARTIFACT_DESCRIPTIVES_V1.csv',
        'Table06_Node_Seed_Diversity_V1.csv':'metrics/NODE_SEED_DIVERSITY_AGGREGATES_V1.csv',
        'Table07_Artifact_Proxy_Review_Scope_V1.csv':'metrics/STRUCTURE_EXPLORATORY_V1.csv',
        'Table09_Paired_KID_Patient_CI_V1.csv':'metrics/QUALITY_PATIENT_CLUSTER_BOOTSTRAP_V1.csv',
        'Table10_Unavailable_Metrics_V1.csv':'metrics/METRIC_NOT_AVAILABLE_OR_EXCLUDED_V1.csv'}
    for dest,source in mapping.items():shutil.copyfile(root/source,tables/dest)
    c=read_json(root/'protocol/FROZEN_QUALITY_EVALUATION_CONTRACT_V1.json');e=read_json(root/'image_features/encoder_contract.json')
    write_csv(tables/'Table02_Reference_Preprocessing_V1.csv',[
        dict(item='Native REAL unique views',value=c['real_unique_views'],meaning='original native views; repeated LOCF slot copies not re-counted'),
        dict(item='Native REAL unique exams',value=c['real_unique_exams'],meaning='historical, not target-year ground truth'),
        dict(item='Native REAL patients',value=c['real_unique_patients'],meaning='one generator patient has no native REAL slot reference'),
        dict(item='Common patients',value=c['common_patients'],meaning='primary distribution support'),
        dict(item='Balanced images per patient',value=c['balanced_policy']['m'],meaning='objective minimum native reference view count, capped at6; seed20261012'),
        dict(item='Texture scale',value='whole RGB512 PIL bilinear',meaning='resizing/aspect ratio may confound technical proxies; no anatomy ROI'),
        dict(item='Feature encoder',value=e['encoder']+' '+e['torch_fidelity_version'],meaning='2048 standard feature; native RGB grouped by dimensions; TF1-compatible internal299'),
        dict(item='Feature value range',value='uint8 RGB; internal (x-128)/128',meaning='no histogram equalization or quality selection'),
        dict(item='SSIM',value='BT.601 gray255; gaussian1.5; win11; population covariance',meaning='paired change, not real-target reconstruction'),
        dict(item='LPIPS',value='AlexNet v0.1; package0.1.4; RGB[-1,1]',meaning='offline calibrated perceptual change, not a medical authenticity score'),
        dict(item='KID',value='degree3 gamma1/2048 coef1;50 subsets512',meaning='MC SD is separate from2000 patient-cluster bootstrap CI')])
    write_csv(tables/'Table08_Claims_Evidence_Limits_V1.csv',[
        dict(claim='Controlled CPFM injection changes saved pixels',evidence='Table04 + exact node/index/seed pairing',limit='Path dependence; neither image is ground truth'),
        dict(claim='Natural-image feature distribution distance is measured',evidence='Table03 + Table09',limit='Inception has natural-image domain bias; no clinical certification'),
        dict(claim='Low-level texture and candidate overlays are quantified',evidence='Table05 + Table07',limit='Whole-image statistical proxies; resize/acquisition nuisance and unknown clinical false-positive rates'),
        dict(claim='Six-seed within-node variation is quantified',evidence='Table06',limit='Technical seed pairs are not independent clinical patients'),
        dict(claim='Full and Plain routes are image/seed aligned',evidence='Table01 + frozen auxiliary source audit',limit='Different routes/conditions; not isolated CPFM effects'),
        dict(claim='Thyroid anatomy or pathology authenticity',evidence='NOT_ESTABLISHED',limit='No validated ROI model or completed expert review'),
        dict(claim='Future target-year reconstruction',evidence='NOT_ESTABLISHED',limit='No exact future-year real-image truth in this study'),
        dict(claim='Independent HT prediction utility',evidence='NOT_RUN',limit='No new CLIMATv2 training, replay, Test access or validation')])
    qa=pd.read_parquet(root/'metrics/IMAGE_DECODE_AND_TECHNICAL_QA_V1.parquet')
    native=qa.groupby(['branch','native_width','native_height','native_mode'],dropna=False).size().rename('images').reset_index()
    write_csv(tables/'Table11_Native_Dimensions_V1.csv',native)
    dup=pd.read_parquet(root/'metrics/EXACT_PIXEL_DUPLICATE_PROOFS_V1.parquet')
    # Pairwise source groups are aggregate only; no IDs, paths or pixel fingerprints.
    if len(dup):duplicate=dup.groupby(['branch','equal_to_branch']).size().rename('duplicate_images_beyond_first').reset_index()
    else:duplicate=pd.DataFrame([dict(branch='ALL',equal_to_branch='NONE',duplicate_images_beyond_first=0)])
    write_csv(tables/'Table12_Exact_Pixel_Duplicates_V1.csv',duplicate)
    return tables


def main(output_root):
    root=setup(output_root);tables=make_tables(root)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,
                         'ytick.labelsize':8,'legend.fontsize':8,'svg.fonttype':'none','pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
    rows=[]
    def export(fig,name,source,capt_en,capt_cn):
        # Capture only actual Text.draw calls (not undrawn, stale tick objects).
        drawn=[];original=Text.draw
        def traced(text,renderer):
            result=original(text,renderer)
            if text.get_visible() and text.get_text() and text.get_figure() is fig:
                bb=text.get_window_extent(renderer);drawn.append((text.get_text(),tuple(bb.extents)))
            return result
        Text.draw=traced
        try:fig.canvas.draw()
        finally:Text.draw=original
        width,height=fig.canvas.get_width_height();overflow=[t for t,b in drawn if b[0]<-2 or b[1]<-2 or b[2]>width+2 or b[3]>height+2]
        if overflow:raise RuntimeError(f'Actual drawn text clips in {name}: {overflow}')
        dest=root/'figures/nonmedical';png=dest/(name+'.png');svg=dest/(name+'.svg');pdf=dest/(name+'.pdf')
        fig.savefig(png,dpi=600);fig.savefig(svg);fig.savefig(pdf)
        with Image.open(png) as im:
            im.convert('RGB').save(png,dpi=(600,600))
        with Image.open(png) as im:dims=im.size;dpi=im.info.get('dpi');mode=im.mode
        assert mode=='RGB' and abs(dpi[0]-600)<.1 and abs(dpi[1]-600)<.1
        svgtext=svg.read_text();assert '<text' in svgtext and '<image' not in svgtext
        assert b'/FontFile2' in pdf.read_bytes()
        caption=root/'figures/captions'/(name+'_CN_EN.md');caption.write_text(capt_en+'\n\n'+capt_cn+'\n',encoding='utf-8')
        source_path=root/'figures/source_data'/(name+'.csv');write_csv(source_path,source)
        rows.append(dict(filename=name,source_data=str(source_path.relative_to(root)),source_rows=len(source),
                         rendered_png=str(png.relative_to(root)),rendered_svg=str(svg.relative_to(root)),pdf_if_present=str(pdf.relative_to(root)),
                         image_dimensions=f'{dims[0]}x{dims[1]}',dpi='600x600',caption_path=str(caption.relative_to(root)),
                         auto_export_status='AUTO_EXPORT_PASS',agent_visual_review_status='NOT_RUN',human_review_status='NOT_RUN',
                         issues='none in automatic drawn-text/export checks',fixed_actions='RGB PNG conversion; editable text; actual tick draw bounds checked'))
        plt.close(fig)
    def labeled(ax,title):ax.set_title(title,loc='left',fontweight='bold',pad=10)
    def box(ax,xy,width,height,text,color):
        p=FancyBboxPatch(xy,width,height,boxstyle='round,pad=0.015',fc=color,ec='#556675',lw=.8);ax.add_patch(p)
        ax.text(xy[0]+width/2,xy[1]+height/2,text,ha='center',va='center',fontsize=9,color='#172C3A')
    counts=pd.read_csv(tables/'Table01_Branch_Support_V1.csv')
    dist=pd.read_csv(tables/'Table03_Distribution_Distances_V1.csv')
    paired=pd.read_csv(tables/'Table04_Paired_Changes_V1.csv')
    tech=pd.read_csv(tables/'Table05_Technical_Proxies_V1.csv');div=pd.read_csv(tables/'Table06_Node_Seed_Diversity_V1.csv')
    flags=pd.read_csv(tables/'Table07_Artifact_Proxy_Review_Scope_V1.csv')
    fig,ax=plt.subplots(figsize=(7.2,4.2),layout='constrained');ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    box(ax,(.04,.71),.41,.22,'Real historical reference\n859 patients / 3,154 exams\n13,200 native views','#E9EDF1')
    box(ax,(.55,.71),.41,.22,'Exact CPFM ON / OFF pairs\n860 patients / 1,545 nodes\n9,270 images per arm','#E6F3F3')
    box(ax,(.23,.38),.54,.21,'Fixed offline evaluation\nCommon support: 859 patients\nBalanced: 2 images / patient / arm','#F1ECF6')
    box(ax,(.04,.05),.27,.21,'Distribution\nStandard Inception\nFID / KID','#E9F0F8')
    box(ax,(.365,.05),.27,.21,'Pair change\nMAE / RMSE\nSSIM / LPIPS','#FBEEE2')
    box(ax,(.69,.05),.27,.21,'Technical proxies\nTexture / overlays\nSix-seed diversity','#EAF2EC')
    for x in [.245,.755]:ax.annotate('',xy=(.5,.59),xytext=(x,.71),arrowprops=dict(arrowstyle='->',color='#526677',lw=1))
    for x in [.175,.5,.825]:ax.annotate('',xy=(x,.26),xytext=(.5,.38),arrowprops=dict(arrowstyle='->',color='#526677',lw=1))
    export(fig,'Fig01_Quality_Study_Flow',counts,'Offline quality evaluation uses frozen saved images; auxiliary Full and Plain routes are not isolated CPFM interventions. No new training, sampling or Test access.','固定已保存图像的离线质量评价；Full 和 Plain 仅为辅助路线，不是 CPFM 单变量干预。无新训练、采样或 Test 访问。')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.8),layout='constrained')
    selected=dist[dist.scope.isin(['all_available_common_support','patient_balanced_common_support']) & dist.branch.isin(['OP_CPFM_ON','OP_CPFM_OFF'])]
    for ax,metric,title in zip(axes,['fid','kid_mc_mean'],['a  Natural-image FID','b  Unbiased cubic KID']):
        for j,b in enumerate(['OP_CPFM_ON','OP_CPFM_OFF']):
            g=selected[selected.branch==b].set_index('scope').loc[['all_available_common_support','patient_balanced_common_support']]
            pos=np.arange(2)+(j-.5)*.28;ax.bar(pos,g[metric],width=.25,color=COLORS[b],label=LABELS[b])
            if metric=='kid_mc_mean':ax.errorbar(pos,g[metric],yerr=g.kid_mc_sd,fmt='none',color='#26343C',lw=.8,capsize=2)
        ax.set_xticks([0,1],['All available\ncommon patients','2 per patient\ncommon patients']);ax.set_ylabel('Feature distance' if metric=='fid' else 'KID (raw MMD squared)');labeled(ax,title);ax.axhline(0,color='#777',lw=.6)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=2,frameon=False)
    export(fig,'Fig02_Real_vs_ON_OFF_FID_KID',selected,'Distances use 859 common patients. KID error bars are Monte Carlo SD over50 subsets512, not patient confidence intervals. Balanced covariance is rank-limited (1,718 images vs2,048 features); FID CI was not run. Neither distance certifies thyroid anatomy.','距离使用 859 名共同患者。KID 误差线是 50 次、每次 512 图的蒙特卡洛标准差，不是患者置信区间。均衡集 1,718 图小于 2,048 特征维数，协方差秩受限；未计算 FID 区间。两指标均不认证甲状腺解剖真实性。')
    pv=pd.read_csv(root/'metrics/ON_OFF_PATIENT_AGGREGATE_V1.csv');pv=pv[pv.comparison=='ON_OFF'];histrows=[]
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.5),layout='constrained')
    for ax,metric,bounds,title in zip(axes.flat,['rgb_mae','rgb_rmse','gray_ssim','lpips'],[(0,255),(0,255),(-1,1),(0,2)],['a  RGB MAE (0-255)','b  RGB RMSE (0-255)','c  Gray SSIM (pair similarity)','d  LPIPS (pair difference)']):
        edges=np.linspace(*bounds,13);v=pv[metric].to_numpy();hist,_=np.histogram(v,edges)
        ax.bar(edges[:-1],hist,width=np.diff(edges),align='edge',color=COLORS['OP_CPFM_ON'],ec='white',lw=.5)
        labeled(ax,title+f'\nn={len(v)}; outside range={int((v<bounds[0]).sum()+(v>bounds[1]).sum())}')
        ax.set_ylabel('Patients');ax.set_xlabel(metric.replace('_',' '));ax.set_xticks(np.linspace(bounds[0],bounds[1],5))
        histrows.extend(dict(metric=metric,bin_left=float(a),bin_right=float(b),patients=int(n),underflow=int((v<bounds[0]).sum()),overflow=int((v>bounds[1]).sum())) for a,b,n in zip(edges[:-1],edges[1:],hist))
    export(fig,'Fig03_Paired_ON_OFF_Pixel_Changes',pd.DataFrame(histrows),'Distributions aggregate six images within node, then nodes within patient; n=860. ON/OFF shares node, image index and exact seed. SSIM/LPIPS describe structural/perceptual change, not reconstruction against target truth. All responses, including zero and extremes, are retained.','先对每节点六图求均值，再对每患者节点求均值；n=860。ON/OFF 共享节点、图序和精确种子。SSIM/LPIPS 表示结构/感知变化，不是相对目标真值的重建准确性。保留零响应及极端值。')
    fig,axes=plt.subplots(1,3,sharey=True,figsize=(7.2,3.8),layout='constrained');tsel=tech[tech.metric.isin(['gray_std','sobel_gradient_mean','high_frequency_fraction'])]
    for ax,metric,title in zip(axes,['gray_std','sobel_gradient_mean','high_frequency_fraction'],['a  Gray contrast (SD)','b  Sobel gradient','c  High-frequency\nfraction']):
        g=tsel[tsel.metric==metric].set_index('branch').loc[BRANCHES];pos=np.arange(5)
        ax.barh(pos,g.patient_equal_mean,color=[COLORS[b] for b in BRANCHES],height=.62)
        ax.errorbar(g.patient_equal_mean,pos,xerr=np.vstack([g.patient_equal_mean-g.patient_ci_low,g.patient_ci_high-g.patient_equal_mean]),fmt='none',color='#26343C',lw=.8,capsize=2)
        ax.set_yticks(pos,[LABELS[b] for b in BRANCHES]);ax.invert_yaxis();labeled(ax,title);ax.set_xlabel('Equal-patient mean')
    export(fig,'Fig04_Contrast_Edge_Frequency',tsel,'Whole-image512 proxies show equal-patient means with2000 patient-bootstrap95% percentile intervals. Native view counts and dimensions differ; resampling and acquisition may affect values. More contrast or high-frequency power is not necessarily more pathological detail.','全图 512 尺度代理显示患者等权均值与 2,000 次患者 bootstrap 的 95% 百分位区间。原始 view 数和尺寸不同，重采样及采集可能影响数值。更多对比度或高频功率不必然表示更好的病理细节。')
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.4),layout='constrained')
    for ax,metric,title in zip(axes,['rgb_rmse','gray_ssim','lpips'],['a  Seed-pair RMSE','b  Seed-pair SSIM','c  Seed-pair LPIPS']):
        g=div[div.metric==metric].set_index('branch').loc[['OP_CPFM_ON','OP_CPFM_OFF']]
        ax.bar([0,1],g.patient_equal_mean,color=[COLORS[b] for b in g.index],width=.5)
        ax.errorbar([0,1],g.patient_equal_mean,yerr=np.vstack([g.patient_equal_mean-g.patient_ci_low,g.patient_ci_high-g.patient_equal_mean]),fmt='none',color='#26343C',capsize=3,lw=.8)
        ax.set_xticks([0,1],['OP ON','CPFM OFF']);labeled(ax,title);ax.set_ylabel('Equal-patient mean')
    export(fig,'Fig05_Node_Seed_Diversity',div,'All15 unordered pairs of six seeds per node were evaluated:23,175 seed pairs per arm,1,545 nodes,860 patients. Error bars use2000 patient-cluster bootstrap; these technical pairs are not independent clinical subjects.','每节点六种子的全部 15 个无序配对均被评价：每臂 23,175 对、1,545 节点、860 患者。误差线来自 2,000 次患者聚类 bootstrap；技术配对不是独立临床受试者。')
    fig,axes=plt.subplots(1,3,sharey=True,figsize=(7.2,3.3),layout='constrained')
    for ax,col,title in zip(axes,['patients','images','nodes'],['a  Patient support','b  Image / view count','c  Generated nodes']):
        values=counts.set_index('branch').loc[BRANCHES,col]
        ax.barh(np.arange(5),values,color=[COLORS[b] for b in BRANCHES],height=.6)
        ax.set_xlim(0,float(values.max())*1.25)
        for j,v in enumerate(values):ax.text(float(v)+float(values.max())*.025,j,f'{int(v):,}',va='center',fontsize=8)
        ax.set_yticks(np.arange(5),[LABELS[b] for b in BRANCHES]);ax.invert_yaxis();labeled(ax,title);ax.set_xlabel('Count')
    export(fig,'Fig06_Reference_Support_and_Matching',counts,'Each generated arm has9270 images/1545 nodes/860 patients; REAL has13200 unique native views/3154 exams/859 patients. One generated patient lacks a native REAL slot reference. Primary distribution comparisons use common859, not an silently altered pair denominator.','各生成臂为 9,270 图/1,545 节点/860 患者；REAL 为 13,200 个原始 view/3,154 Exam/859 患者。有 1 名生成患者缺少原始 REAL slot 参考。主分布比较用共同 859 人，不暗改配对分母。')
    fig,axes=plt.subplots(2,4,figsize=(7.2,5.1),layout='constrained')
    for i,(ax,proxy) in enumerate(zip(axes.flat,flags.proxy.drop_duplicates())):
        g=flags[flags.proxy==proxy].set_index('branch').loc[BRANCHES]
        ax.bar(np.arange(5),g.patient_equal_fraction,color=[COLORS[b] for b in BRANCHES],width=.7)
        ax.set_xticks(np.arange(5),['Real','ON','OFF','Full','Plain'],rotation=50,ha='right');ax.set_ylim(0,1);labeled(ax,chr(97+i)+'  '+proxy.replace('_flag','').replace('_proxy','').replace('_','\n'));ax.set_ylabel('Flag fraction')
    axes.flat[-1].axis('off');axes.flat[-1].text(.03,.96,'Fixed technical flags\n\nNo expert labels\n\nUnknown medical\nfalse-positive rate\n\nNo image excluded',va='top',fontsize=9)
    export(fig,'Fig07_Artifact_Flags_and_Low_Contrast',flags,'Flags are prespecified whole-image statistical/line candidates, not thyroid diagnoses. Bars are equal-patient mean fractions. Hough line/cross candidates can detect genuine interfaces or acquisition UI; clinical specificity is unknown. None were removed.','标记是预声明的全图统计/直线候选，不是甲状腺诊断。柱高为患者等权标记比例。Hough 直线/十字可能检出真实界面或采集 UI，临床特异性未知；没有据此剔图。')
    evidence=pd.read_csv(tables/'Table08_Claims_Evidence_Limits_V1.csv')
    fig,ax=plt.subplots(figsize=(7.2,4.6),layout='constrained');ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    good=['Saved-pixel intervention response','Natural-image feature distances','Low-level technical texture proxies','Within-node seed variation']
    missing=['Certified thyroid anatomy / pathology','Future target-year reconstruction','Independent HT prediction validation']
    ax.text(.03,.97,'Measured offline evidence',fontweight='bold',va='top',fontsize=11)
    for i,text in enumerate(good):box(ax,(.035,.77-i*.145),.45,.11,text,'#E6F3F3')
    ax.text(.55,.97,'Not established by this study',fontweight='bold',va='top',fontsize=11)
    for i,text in enumerate(missing):box(ax,(.55,.77-i*.18),.41,.13,text,'#F4EBE1')
    ax.text(.04,.065,'Expert review: NOT_RUN   |   Validated thyroid ROI: NOT_AVAILABLE\nNo authenticity composite score; no quality-based exclusion.',va='bottom',fontsize=9)
    export(fig,'Fig08_Quality_Evidence_and_Limitations',evidence,'Measured statistical and technical evidence is separated from absent clinical or target-truth evidence. Offline quality, structural visibility and downstream HT utility are different questions.','已测量的统计/技术证据与缺失的临床/目标真值证据分开。离线质量、结构可视性与下游 HT 效用是不同问题。')
    write_csv(root/'figures/export_qa/NONMEDICAL_FIGURE_EXPORT_QA_V1.csv',rows)
    atomic_json(root/'figures/export_qa/AUTO_EXPORT_STATUS_V1.json',dict(status='AUTO_EXPORT_PASS',groups=len(rows),actual_visual_review='NOT_RUN',updated_utc=now()))
    print({'aggregate_tables':len(list(tables.glob('*.csv'))),'figure_groups':len(rows),'auto_export':'PASS','agent_visual':'NOT_RUN'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',required=True);a=p.parse_args();main(a.output_root)
