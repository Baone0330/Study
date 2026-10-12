"""Six source-grounded, nonmedical figures; actual render QA and no image examples."""
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from review_common import setup,read_json,csv_out,json_out,now


def main(output_root):
    root=setup(output_root);assert len(read_json(root/'protocol/PRE_DRAW_FIGURE_EVIDENCE_PLAN_V1.json')['figures'])==6
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
                         'legend.fontsize':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
    qa=[];captions=[]
    def export(fig,name,source,en,cn):
        original=Text.draw;drawn=[]
        def traced(t,renderer):
            result=original(t,renderer)
            if t.get_visible() and t.get_text() and t.get_figure() is fig:drawn.append((t.get_text(),tuple(t.get_window_extent(renderer).extents),t.get_fontsize()))
            return result
        Text.draw=traced
        try:fig.canvas.draw()
        finally:Text.draw=original
        w,h=fig.canvas.get_width_height();bad=[t for t,b,s in drawn if b[0]<-2 or b[1]<-2 or b[2]>w+2 or b[3]>h+2 or s<8]
        if bad:raise RuntimeError('Actual drawn text/font QA failed '+name+': '+repr(bad))
        dest=root/'figures/public_nonmedical';png=dest/(name+'.png');svg=dest/(name+'.svg');pdf=dest/(name+'.pdf')
        fig.savefig(png,dpi=600);fig.savefig(svg);fig.savefig(pdf)
        with Image.open(png) as im:im.convert('RGB').save(png,dpi=(600,600))
        with Image.open(png) as im:assert im.mode=='RGB' and abs(im.info['dpi'][0]-600)<.1;dims=im.size
        text=svg.read_text();assert '<text' in text and '<image' not in text;assert b'/FontFile2' in pdf.read_bytes()
        caption=en+'\n\n'+cn+'\n';(root/'figures/captions_cn_en'/(name+'_CN_EN.md')).write_text(caption,encoding='utf-8');captions.append('## '+name+'\n\n'+caption)
        csv_out(root/'figures/source_data_public'/(name+'.csv'),source)
        qa.append(dict(figure=name,source_data='figures/source_data_public/'+name+'.csv',rows=len(source),png='figures/public_nonmedical/'+name+'.png',
                       svg='figures/public_nonmedical/'+name+'.svg',pdf='figures/public_nonmedical/'+name+'.pdf',dimensions=f'{dims[0]}x{dims[1]}',dpi=600,mode='RGB',
                       auto_export='PASS',actual_nonmedical_agent_visual_review='NOT_RUN',human_review='NOT_RUN',medical_pixels=False))
        plt.close(fig)
    def box(ax,x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',facecolor=color,edgecolor='#506576',lw=.8))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9,color='#203340')
    colors=['#177E89','#D48B42','#75818B'];labels=['ON','OFF','Real history']
    cohort=pd.read_csv(root/'tables/Table01_Review_Cohort_and_Coverage.csv');region=pd.read_csv(root/'tables/Table03_Region_Artifact_Distribution.csv')
    contrast=pd.read_csv(root/'tables/Table04_CPFM_ON_OFF_Qualitative_Comparison.csv');joint=pd.read_csv(root/'tables/Table02_Flag_vs_Visual_Category.csv')
    fig,ax=plt.subplots(figsize=(7.2,4.3),layout='constrained');ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    box(ax,.04,.72,.43,.20,'Full ON/OFF primary\n9,270 pairs / 1,545 nodes\n860 patients','#E7F2F3')
    box(ax,.56,.72,.4,.20,'Real historical reference\n13,200 views / 3,154 exams\n859 patients; NOT future truth','#EDF0F2')
    box(ax,.05,.40,.41,.19,'Fixed core\n12 nodes / 72 pairs\n12 patients; all six indices','#EAE7F2')
    box(ax,.55,.40,.41,.19,'Flag-stratified sample\n4 strata x12 pairs\n48 distinct patients','#FBEEE1')
    box(ax,.07,.07,.37,.18,'Expert subset of core\n36 pairs / 72 cards\n0 actual doctors / ratings','#EAEFF5')
    box(ax,.55,.07,.41,.18,'Full statistics retain\n420 OFF-only residual pairs\n48-case prevalence: unknown','#F4E8E5')
    for xy,source in [((.25,.59),(.25,.72)),((.75,.59),(.25,.72)),((.25,.25),(.25,.40))]:ax.annotate('',xy=xy,xytext=source,arrowprops=dict(arrowstyle='->',lw=.8,color='#4E6674'))
    export(fig,'Fig01_Review_Design_and_Cohorts',cohort,'Frozen paired cohort and review subsets. Core and stratified samples have zero patient overlap. Expert cards are preparation only; the four strata do not exhaust the full queue.','固定配对队列及复核子集。核心和分层样本患者无重叠；医生卡片只是准备，四层并未穷尽全队列。')
    branches=['OP_CPFM_ON','OP_CPFM_OFF','REAL_HISTORY_REFERENCE']
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.9),layout='constrained')
    for ax,m,title in zip(axes[:2],['old_long_line_flag','old_cross_proxy_flag'],['a  Original line-hit\nfraction','b  Original cross-hit\nfraction']):
        g=region[region.metric==m].set_index('branch').loc[branches]
        ax.bar(np.arange(3),g.patient_mean,color=colors);ax.set_xticks(range(3),labels,rotation=35,ha='right');ax.set_ylim(0,1);ax.set_ylabel('Equal-patient fraction');ax.set_title(title,loc='left',fontsize=9)
    m=['line_border_any','line_central_any','line_intermediate_any','line_multizone_any'];source=region[region.metric.isin(['old_long_line_flag','old_cross_proxy_flag']+m)]
    ax=axes[2]
    for j,b in enumerate(branches):
        g=region[region.branch.eq(b)].set_index('metric').loc[m];ax.bar(np.arange(4)+(j-1)*.25,g.patient_mean,width=.22,color=colors[j],label=labels[j])
    ax.set_xticks(range(4),['Border','Center','Middle','Multi-zone'],rotation=40,ha='right');ax.set_ylim(0,1);ax.set_ylabel('Equal-patient candidate-hit fraction');ax.set_title('c  Geometric positions\n(overlap)',loc='left',fontsize=9)
    fig.legend(loc='outside upper center',ncol=3,frameon=False)
    export(fig,'Fig02_Line_Cross_Proxy_and_Regions',source,'Original flags were reproduced exactly. Bars are equal-patient detector-hit fractions, not clinical artifact prevalence. Geometric positions overlap and are not thyroid ROI; REAL acquisition support differs.','原 flag 精确复现。柱高为患者等权检测器命中比例，不是临床伪影发生率。几何区域可重叠且不是甲状腺 ROI；REAL 采集支持不同。')
    fig,ax=plt.subplots(figsize=(7.2,3.7),layout='constrained');ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    box(ax,.04,.64,.43,.25,'31,740 current source reads\nOriginal flag / count mismatches:0\nReproducibility: PASS','#E7F2F3')
    box(ax,.55,.64,.41,.25,'Human reference annotations:0\nTP / FP / TN / FN: NA\nSensitivity / specificity: NOT_RUN','#F4E8E5')
    box(ax,.04,.20,.43,.27,'Full-pair machine joint counts\nBoth true / ON-only / OFF-only\nBoth false; NOT a confusion matrix','#EAEFF5')
    box(ax,.55,.20,.41,.27,'Candidate origin: NOT_REVIEWED\nUI / interface / generated geometry\nare competing explanations','#FBEEE1')
    ax.text(.04,.04,'Unknown false-positive rate is not zero; no candidate was removed.',fontsize=9)
    export(fig,'Fig03_Flag_Audit_And_Unknowns',joint,'Flag reproducibility and actual annotation missingness are separate. Machine joint counts are not TP/FP against anatomical truth. No actual image observations were available.','flag 复现和真实标注缺位分开。机器联合计数不是相对解剖真值的 TP/FP；没有真实图像观察。')
    groups=[['old_long_line_flag','old_cross_proxy_flag'],['line_candidates','cross_candidates']]
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.5),layout='constrained');used=contrast[contrast.metric.isin(sum(groups,[]))]
    for ax,ms,title in zip(axes,groups,['a  Proxy-hit fraction difference','b  Candidates per image difference']):
        g=contrast.set_index('metric').loc[ms];pos=np.arange(2);v=g.delta_on_minus_off.to_numpy();err=np.vstack([v-g.ci_low.to_numpy(),g.ci_high.to_numpy()-v])
        ax.errorbar(v,pos,xerr=err,fmt='o',color=colors[0],capsize=4,lw=1.3);ax.axvline(0,color='#64717B',lw=.8)
        ax.set_yticks(pos,['Long line','Cross']);ax.invert_yaxis();ax.set_xlabel('ON minus OFF; equal-patient mean');ax.set_title(title,loc='left',fontsize=9)
    export(fig,'Fig04_Paired_Artifact_Difference',used,'Paired mean contrasts with95% percentile intervals from2,000 patient-cluster resamples, n=860. Images average within node, nodes within patient. Positive means more detector response with ON, not improved medical quality. Exploratory contrasts; no multiplicity-adjusted confirmation.','860患者、2,000次患者聚类重抽样的配对均值与95%百分位区间。图像在节点内、节点在患者内平均。正值表示 ON 检测器响应更多，不是医学质量改善；为探索性对比，非多重性校正确证。')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.7),layout='constrained')
    ax=axes[0];ax.barh([0,1],[144,96],color=['#75818B','#D4A575']);ax.set_yticks([0,1],['Core144 images','Stratified96 images']);ax.invert_yaxis();ax.set_xlim(0,170);ax.set_xlabel('Images with NOT_REVIEWED visibility')
    for i,v in enumerate([144,96]):ax.text(v+3,i,str(v),va='center')
    ax.set_title('a  Visibility observation\nmissingness',loc='left',fontsize=9)
    ax=axes[1];ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    box(ax,.08,.72,.84,.20,'Machine: technical candidates only','#E7F2F3')
    box(ax,.08,.42,.84,.20,'Nonclinical / Codex medical viewing\nNOT_RUN','#EDF0F2')
    box(ax,.08,.10,.84,.20,'Doctors:0; actual scores:0\n72 private cards pending approval','#F4E8E5')
    source=pd.DataFrame([dict(cohort='core',unreviewed_images=144,actual_observations=0),dict(cohort='stratified',unreviewed_images=96,actual_observations=0),dict(cohort='expert_subset_not_added_to240',unreviewed_images=72,actual_observations=0)])
    export(fig,'Fig05_Visibility_Observation_Distribution',source,'Missingness replaces fabricated visibility grades. NOT_REVIEWED is not observed UNCERTAIN. The72 expert candidates are a subset of core144, not72 additional independent images; no completed ratings or inter-rater agreement.','以缺位代替虚构可视性分级。NOT_REVIEWED 不同于实际观察后的 UNCERTAIN；72医生候选是144核心图的子集，不能另外相加为独立图。无已完成评分或评分者一致性。')
    evidence=pd.read_csv(root/'tables/Table07_Claim_Evidence_Boundary.csv');fig,ax=plt.subplots(figsize=(7.2,4.5),layout='constrained');ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(0,1)
    ax.text(.04,.97,'Measured / prepared',va='top',fontweight='bold',fontsize=11);ax.text(.55,.97,'Not established / not performed',va='top',fontweight='bold',fontsize=11)
    for i,t in enumerate(['Prior FID / KID\nreferenced; NOT recomputed','Unchanged detector flags\nand candidate positions','Paired technical geometry\n860-patient bootstrap','Fixed private panels / cards\nprepared only']):box(ax,.04,.74-i*.19,.43,.14,t,'#E7F2F3')
    for i,t in enumerate(['Human anatomy / artifact labels','Thyroid ROI / future-year truth','Clinician medical authenticity','New prediction benefit / Test']):box(ax,.55,.74-i*.19,.41,.14,t,'#F4E8E5')
    ax.text(.04,.025,'Feature proximity, structure visibility and prediction utility are distinct endpoints.',fontsize=9)
    export(fig,'Fig06_Claim_Evidence_Limitations',evidence,'Technical evidence does not close the medical validity gap. Prior lower feature distances for ON coexist with more line/cross candidates; their clinical origins remain unknown. No authenticity score or downstream inference was created.','技术证据不填补医学有效性缺口。上轮 ON 特征距离较小与本轮线/十字候选较多并存，临床来源仍未知；未构造真实性评分或下游推断。')
    csv_out(root/'figures/export_qa/NONMEDICAL_FIGURE_EXPORT_QA_V1.csv',qa)
    (root/'writing/CAPTIONS_ALL_CN_EN_V1.md').write_text('# Bilingual figure captions\n\n'+'\n'.join(captions),encoding='utf-8')
    json_out(root/'figures/export_qa/AUTO_EXPORT_STATUS_V1.json',dict(status='AUTO_EXPORT_PASS',figures=len(qa),actual_agent_visual_review='NOT_RUN',updated_utc=now()))
    print({'nonmedical_figures':len(qa),'RGB600_PNG_SVG_PDF':'PASS','actual_visual_QA':'NOT_RUN'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',required=True);a=p.parse_args();main(a.output_root)
