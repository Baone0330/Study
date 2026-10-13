"""Real model detail and source-grounded aggregate figures; never draw medical image pixels."""
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve,precision_recall_curve
from common import setup,config,jread,jout,cout,ARMS,SEEDS,now


def main(path):
    c=config(path);root=setup(c['project_private'])
    plan=[dict(figure='Fig01_LTSA_HT_Detailed_Architecture',claim='local LTSA backbone/time/causal body adapted to one patient HT logit',type='schematic-led composite',evidence='local source+GPU parameter/gradient smoke'),
          dict(figure='Fig02_Bounded_Synthetic_Token_Mapping',claim='same real skeleton; explicitly virtual bounded queries only, Train only',type='schematic-led composite',evidence='pre-Val frozen686-node ledger'),
          dict(figure='Fig03_Two_Arm_Internal_Val_ROC_PR',claim='actually measured two patient-ensemble internal-Val curves',type='quantitative grid',evidence='completed189-patient ensembles only'),
          dict(figure='Fig04_Seed_And_Ensemble_Performance',claim='three independent seed results and probability-average ensemble',type='quantitative grid',evidence='six strict reloads'),
          dict(figure='Fig05_OP_Minus_REAL_Paired_Intervals',claim='patient-paired augmentation contrasts, not absolute predictor rank',type='quantitative grid',evidence='2000patient bootstrap+historical reference')]
    jout(root/'figures/FIGURE_EVIDENCE_CONTRACT_V3.json',dict(backend='Python',figures=plan,exports='RGB600dpiPNG/editableSVG/embeddedTrueTypePDF',min_font_pt=8,medical_pixels=False,qualitative_timeline='schematic not actual patient dates'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.text import Text
    from matplotlib.patches import FancyBboxPatch,Rectangle
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
                         'legend.fontsize':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white','figure.facecolor':'white'})
    qa=[];captions=[];colors=['#55758D','#A2754C']
    def box(ax,x,y,w,h,text,color='#EAF0F4',size=8):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008',fc=color,ec='#5A6B7A',lw=.8));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size)
    def arrow(ax,xy,xytext):ax.annotate('',xy=xy,xytext=xytext,arrowprops=dict(arrowstyle='->',color='#526B7C',lw=1))
    def export(fig,name,source,en,cn):
        drawn=[];old=Text.draw
        def trace(t,r):
            result=old(t,r)
            if t.get_visible() and t.get_text() and t.get_figure() is fig:drawn.append((t.get_text(),tuple(t.get_window_extent(r).extents),t.get_fontsize()))
            return result
        Text.draw=trace
        try:fig.canvas.draw()
        finally:Text.draw=old
        w,h=fig.canvas.get_width_height();bad=[t for t,b,s in drawn if b[0]<-2 or b[1]<-2 or b[2]>w+2 or b[3]>h+2 or s<8]
        if bad:raise RuntimeError('Actual drawn text/font bounds fail '+name+repr(bad))
        prefix=root/'figures/nonmedical'/name
        fig.savefig(str(prefix)+'.png',dpi=600);fig.savefig(str(prefix)+'.svg');fig.savefig(str(prefix)+'.pdf')
        png=str(prefix)+'.png'
        with Image.open(png) as im:im.convert('RGB').save(png,dpi=(600,600))
        with Image.open(png) as im:assert im.mode=='RGB' and abs(im.info['dpi'][0]-600)<.1;size=im.size
        svg=(root/'figures/nonmedical'/(name+'.svg')).read_text();assert '<text' in svg and '<image' not in svg
        assert b'/FontFile2' in (root/'figures/nonmedical'/(name+'.pdf')).read_bytes()
        cout(root/'figures/source_data'/(name+'.csv'),source)
        (root/'figures/captions'/(name+'_CN_EN.md')).write_text(en+'\n\n'+cn+'\n',encoding='utf-8')
        captions.append('## '+name+'\n\n'+en+'\n\n'+cn+'\n')
        qa.append(dict(figure=name,dimensions=f'{size[0]}x{size[1]}',dpi=600,mode='RGB',auto_export='PASS',actual_agent_visual_review='NOT_RUN',medical_pixels=False));plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,6.2),layout='constrained');ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    ax.text(.02,.98,'a  Shared image encoder: end-to-end ResNet18',fontsize=10,fontweight='bold',va='top')
    box(ax,.02,.82,.13,.10,'All valid views\nRGB224\nImageNet norm')
    box(ax,.18,.80,.15,.14,'Stem\nConv7x7 s2\nBN / ReLU\nMaxPool3x3')
    for x,width,nch,stride in [(.36,.13,64,1),(.52,.13,128,2),(.68,.13,256,2),(.84,.13,512,2)]:
        box(ax,x,.80,width,.14,f'Stage {nch}\n2 basic blocks\nfirst stride{stride}',color='#E4EEF4')
    for end,start in [((.18,.87),(.15,.87)),((.36,.87),(.33,.87)),((.52,.87),(.49,.87)),((.68,.87),(.65,.87)),((.84,.87),(.81,.87))]:arrow(ax,end,start)
    box(ax,.03,.62,.58,.12,'Basic block (each stage)\nConv3x3 -> BN -> ReLU -> Conv3x3 -> BN\n+ identity / projected skip -> ReLU',color='#F1F4F6')
    box(ax,.68,.63,.29,.10,'Global average pool\nOne512D feature / image')
    arrow(ax,(.82,.73),(.90,.80))
    ax.text(.02,.58,'b  Multi-view grouping and original temporal encoding',fontsize=10,fontweight='bold',va='top')
    box(ax,.03,.38,.29,.14,'Per real Exam (or synth node)\nMasked mean of all valid views\ne = sum(f_view) / M\nOne512D time token',color='#EAF1E9')
    box(ax,.37,.38,.29,.14,'Fixed original LTSA TPE\nm = floor(observed days /30.4375)\nsin / cos at512 dimensions\nNo target-date gap',color='#EEE9F3')
    box(ax,.72,.38,.25,.14,'Variable sequence\nAll real Exams retained\nSynth nodes marked virtual\nPad mask from valid lengths')
    arrow(ax,(.37,.45),(.32,.45));arrow(ax,(.72,.45),(.66,.45))
    ax.plot([.82,.99,.99,.175],[.63,.63,.545,.545],color='#526B7C',lw=1);arrow(ax,(.175,.52),(.175,.545))
    ax.text(.02,.34,'c  Causal Transformer and last-real-token classification',fontsize=10,fontweight='bold',va='top')
    box(ax,.03,.095,.37,.20,'One encoder layer\n4-head self-attention (d512)\nCausal + padding masks\nResidual + LayerNorm\nLinear512 -> ReLU / Dropout0.25\n-> Linear512; Residual + LayerNorm',color='#E6EFF3')
    box(ax,.46,.095,.24,.20,'Readout\nLast eligible REAL token\n(not padded final slot)\nDropout0.25\nLinear512 ->1 logit',color='#F2EADF')
    box(ax,.76,.095,.21,.20,'Training: BCE logits\nOne loss / patient\nInference: sigmoid\nP(final-year HT=1)\nNo step-ahead / survival',color='#EBEDEA')
    ax.plot([.84,.99,.99,.215],[.38,.38,.315,.315],color='#526B7C',lw=1);arrow(ax,(.215,.295),(.215,.315))
    arrow(ax,(.46,.195),(.40,.195));arrow(ax,(.76,.195),(.70,.195))
    ax.text(.02,.025,'12,755,009 trainable parameters; identical architecture in both data arms.',fontsize=8)
    source=pd.DataFrame([dict(component=k,description=v) for k,v in [('ResNet18','all native views224RGB; stages64/128/256/512x2;12,755,009 total model parameters'),('pool','allvalid views; one token per actualExam or virtualsource node'),('TPE','original sinusoidal512; observed or frozenboundedquery relative month; maximum78'),('Transformer','1layer/4heads/FFN512ReLU/postnorm/.25dropout/causal+padding'),('head','lastREALtoken/Dropout.25/Linear512->1/BCElogits')]])
    export(fig,'Fig01_LTSA_HT_Detailed_Architecture',source,'Fig.1 | LTSA-HT adapted architecture. ResNet18 internal stages and residual basic-block flow, all-view grouping, fixed local LTSA temporal encoding, masked causal Transformer and last-real-token logit head are shown. Components were source-checked and GPU-smoke tested; neither original survival head nor step-ahead is used.','图1 | LTSA-HT适配结构。显示ResNet18内部stage/残差basic block、全部view聚合、原本地LTSA固定时间编码、掩码因果Transformer和末真实token分类头。结构经源码与GPU烟测核实；不使用原生存头或step-ahead。')
    fig,ax=plt.subplots(figsize=(7.2,4.1),layout='constrained');ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    box(ax,.04,.71,.92,.17,'Shared real-history skeleton: first REAL ... intermediate REAL ... last REAL\nOnly observed date intervals; all native views; same860Train/189Val patients',color='#E8EEF3')
    box(ax,.04,.38,.43,.18,'REAL-only\n4,585 real exams /18,413train views\nNo synthetic content')
    box(ax,.55,.38,.41,.18,'Original Progression arm\n+686 bounded virtual query nodes\n+4,116views;597patients augmented',color='#F2E9DF')
    box(ax,.04,.05,.43,.18,'Virtual query is NOT a real Exam\nFirst-REAL anniversary -> fixed query\nStrictly before last REAL;6views ->1token',color='#EEE9F3')
    box(ax,.55,.05,.41,.18,'Excluded (not remapped)\n758target-year +101after-last-REAL nodes\nVal always REAL only; last token REAL',color='#EFE7E4')
    arrow(ax,(.255,.56),(.255,.71));arrow(ax,(.755,.56),(.755,.71))
    mapping=pd.read_csv(root/'tables/OP_CANDIDATE_MAPPING_COUNTS_V3.csv')
    export(fig,'Fig02_Bounded_Synthetic_Token_Mapping',mapping,'Fig.2 | Frozen augmentation eligibility. Synthetic annual queries were anchored to the first real-history date, not the actual target date. Only686 nodes lying strictly inside the observed history span were included;859 nodes were excluded. The597 augmented and263 nonaugmented Train patients remain the same860-person cohort.','图2 | 冻结增强资格。合成年查询由首真实历史日期锚定，不由目标实际日期回推。仅686节点严格位于已观察历史跨度内，排除859节点。有合成597人、无合成263人仍组成同860训练人群。')
    final=root/'releases/FINAL_LTSA_TWO_ARM_STATUS_V3.json'
    if final.exists() and jread(final)['status']=='TWO_ARM_COMPLETED':
        ensembles=[pd.read_csv(root/'metrics/private'/f'{a}_THREE_SEED_ENSEMBLE.csv') for a in ARMS]
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.4),layout='constrained');source=[]
        for d,label,color in zip(ensembles,['REAL-only','Original Progression'],colors):
            fpr,tpr,_=roc_curve(d.y_true,d.p_positive);precision,recall,_=precision_recall_curve(d.y_true,d.p_positive)
            axes[0].plot(fpr,tpr,label=label,color=color,lw=1.6);axes[1].plot(recall,precision,label=label,color=color,lw=1.6)
            source.extend(dict(arm=label,curve='ROC',x=float(x),y=float(y)) for x,y in zip(fpr,tpr));source.extend(dict(arm=label,curve='PR',x=float(x),y=float(y)) for x,y in zip(recall,precision))
        axes[0].plot([0,1],[0,1],ls='--',color='#9B9B9B',lw=.7);axes[1].axhline(58/189,ls='--',color='#9B9B9B',lw=.7)
        for ax,x,y,title in zip(axes,['False-positive rate','Recall'],['True-positive rate','Precision'],['a  Internal-Val ROC','b  Internal-Val precision-recall']):
            ax.set(xlabel=x,ylabel=y,xlim=(0,1),ylim=(0,1),title=title);ax.legend(loc='lower right' if ax is axes[0] else 'lower left',frameon=False)
        export(fig,'Fig03_Two_Arm_Internal_Val_ROC_PR',pd.DataFrame(source),'Fig.3 | Two-arm internal-validation curves. Three independently reloaded seed probabilities were averaged for each of the same189patients (58positive/131negative). Dashed references show chance ROC and positive prevalence; AP is sklearn average precision, not trapezoidal area. This is not Official Test.','图3 | 双臂内部验证曲线。同189患者（58阳性/131阴性）的三个独立重载种子正类概率平均；虚线为随机ROC/阳性比例。AP为sklearn average precision，非梯形面积；不是官方Test。')
        perseed=pd.read_csv(root/'tables/LTSA_TWO_ARM_ALL_SIX_RUNS_V3.csv');ens=pd.read_csv(root/'tables/LTSA_TWO_ARM_ENSEMBLE_COMPARISON_V3.csv');fig,axes=plt.subplots(1,2,figsize=(7.2,3.5),layout='constrained')
        for ax,metric in zip(axes,['ROC_AUC','AP']):
            for i,arm in enumerate(ARMS):
                v=perseed[perseed.arm.eq(arm)][metric].to_numpy();ax.scatter(np.full(3,i)+np.array([-.10,0,.10]),v,color=colors[i],label='3 seeds' if i==0 else None,s=24)
                ax.scatter([i],[float(ens[ens.arm.eq(arm)][metric].iloc[0])],marker='D',facecolor='white',edgecolor=colors[i],s=50,label='Probability ensemble' if i==0 else None,zorder=4)
            ax.set_xticks([0,1],['REAL-only','Original Progression']);ax.set_ylim(0,1);ax.set_ylabel(metric);ax.set_title('a  ROC-AUC' if metric=='ROC_AUC' else 'b  Average precision',loc='left')
        handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=2,frameon=False)
        export(fig,'Fig04_Seed_And_Ensemble_Performance',pd.concat([perseed[['arm','seed','ROC_AUC','AP']],ens[['arm','ROC_AUC','AP']].assign(seed='PROBABILITY_ENSEMBLE')]),'Fig.4 | Seed and probability-ensemble performance. Filled points are the three independent downstream seeds42/3407/2026; open diamonds are average-probability ensembles, not average AUC. Seed variability is not a patient confidence interval. All use the same189 internal-Val patients.','图4 | 种子和概率集成表现。实心点为42/3407/2026三个独立下游种子，空心菱形为概率平均集成而非平均AUC。种子变异不是患者置信区间；均为同189内部Val患者。')
        ci=pd.read_csv(root/'tables/LTSA_OP_MINUS_REAL_PAIRED_BOOTSTRAP_V3.csv');fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
        for ax,row in zip(axes,ci.itertuples(index=False)):
            ax.errorbar([row.delta_OP_minus_REAL],[0],xerr=np.array([[row.delta_OP_minus_REAL-row.ci_low],[row.ci_high-row.delta_OP_minus_REAL]]),fmt='o',capsize=4,color=colors[1]);ax.axvline(0,color='#8C8C8C',ls='--',lw=.7)
            ax.set_yticks([0],['LTSA-HT']);ax.set_xlabel('OP minus REAL-only');ax.set_title(row.metric,loc='left');ax.set_ylim(-.8,.8)
        export(fig,'Fig05_OP_Minus_REAL_Paired_Intervals',ci,'Fig.5 | Paired augmentation contrasts. Intervals are95% percentile intervals from2,000 paired resamples of189 patients, seed20261013, conditional on the fixed selected ensembles. They do not include training/model-selection uncertainty; crossing0 does not demonstrate a significant gain. Two endpoints are exploratory.','图5 | 配对增强效应。固定已选集成条件下，同189患者、种子20261013、2,000次配对重抽样95%百分位区间；不包含训练/选模不确定性。跨0不证明显著增益；两终点为探索性。')
    cout(root/'figures/export_qa/FIGURE_EXPORT_QA_V3.csv',qa);jout(root/'figures/export_qa/AUTO_FIGURE_STATUS_V3.json',dict(status='AUTO_EXPORT_PASS',figures=len(qa),actual_visual_QA='NOT_RUN',medical_pixels=False,updated_utc=now()))
    (root/'writing/FIGURES_CAPTIONS_CN_EN_V3.md').write_text('# Figure legends\n\n'+'\n'.join(captions),encoding='utf-8');print({'figure_groups':len(qa),'auto_export':'PASS','actual_visual_review':'NOT_RUN'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();main(a.config)
