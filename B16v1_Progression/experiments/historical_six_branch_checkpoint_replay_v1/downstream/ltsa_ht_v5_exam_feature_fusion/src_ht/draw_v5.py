"""Four safe Python figure groups; export QA is not actual visual review."""
import os
from common import ROOT,config,setup,jread,jout,cout,ARMS
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache/matplotlib'))
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.text import Text
from PIL import Image
from sklearn.metrics import roc_curve,precision_recall_curve
from v5_safety import enable_history_reference
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
COLORS=['#4C78A8','#D39757']
def main(path):
    c=config(path);root=setup(c['project_private']);enable_history_reference(root);assert jread(root/'releases/FINAL_V5_STATUS.json')['RUNS_AND_METRICS_COMPLETE']
    receipts=[]
    def save(fig,name):
        fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox;issues=[];fonts=[]
        for t in fig.findobj(Text):
            if not t.get_visible() or not t.get_text():continue
            fonts.append(t.get_fontsize());bb=t.get_window_extent(renderer)
            if bb.x0<bounds.x0-2 or bb.x1>bounds.x1+2 or bb.y0<bounds.y0-2 or bb.y1>bounds.y1+2:
                issues.append(dict(reason='text_outside_canvas',text=t.get_text(),font_pt=t.get_fontsize(),bounds=[bb.x0,bb.y0,bb.x1,bb.y1]))
        if not fonts or min(fonts)<8 or issues:
            failure=dict(status='FAIL',figure=name,min_font_pt=min(fonts) if fonts else None,canvas=[bounds.x0,bounds.y0,bounds.x1,bounds.y1],issues=issues)
            jout(root/'figures/export_qa'/f'{name}_EXPORT_DIAGNOSTIC.json',failure)
            raise RuntimeError(f'Figure export check failed: {failure}')
        stem=root/'figures/nonmedical'/name
        fig.savefig(stem.with_suffix('.svg'),bbox_inches='tight');fig.savefig(stem.with_suffix('.png'),dpi=600,bbox_inches='tight',facecolor='white')
        svg=stem.with_suffix('.svg').read_text();assert '<text' in svg and '<image' not in svg
        with Image.open(stem.with_suffix('.png')) as im:
            dpi=im.info.get('dpi');assert dpi and min(dpi)>599;shape=im.size
        receipts.append(dict(figure=name,PNG_600dpi=True,SVG_editable_text=True,embedded_raster=False,min_font_pt=min(fonts),text_bounds='PASS',pixel_shape=shape,automatic_status='PASS',actual_visual_review='NOT_RUN'))
        plt.close(fig)
    ensemble=pd.read_csv(root/'tables/V5_THREE_SEED_ENSEMBLE.csv');seeds=pd.read_csv(root/'tables/V5_SEED_LEVEL_METRICS.csv');cross=pd.read_csv(root/'tables/V3_V4_V5_CROSS_PROTOCOL_COMPARISON.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.4))
    for i,arm in enumerate(ARMS):
        d=pd.read_csv(root/'metrics/private'/f'{arm}_THREE_SEED_ENSEMBLE.csv');fpr,tpr,_=roc_curve(d.y_true,d.p_positive);precision,recall,_=precision_recall_curve(d.y_true,d.p_positive)
        label='REAL-only' if i==0 else 'OP fusion';row=ensemble[ensemble.arm.eq(arm)].iloc[0]
        axes[0].plot(fpr,tpr,color=COLORS[i],ls='-' if i==0 else '--',label=f'{label}: AUC {row.ROC_AUC:.3f}',lw=1.6)
        axes[1].plot(recall,precision,color=COLORS[i],ls='-' if i==0 else '--',label=f'{label}: AP {row.AP:.3f}',lw=1.6)
        cout(root/'figures/source_data'/f'V5_ROC_{i}.csv',pd.DataFrame(dict(false_positive_rate=fpr,true_positive_rate=tpr)))
        cout(root/'figures/source_data'/f'V5_PR_{i}.csv',pd.DataFrame(dict(recall=recall,precision=precision)))
    axes[0].plot([0,1],[0,1],'--',color='#999999',lw=.8);axes[1].axhline(58/189,ls='--',color='#999999',lw=.8)
    for ax,title,xlabel,ylabel in zip(axes,['a  V5 ensemble ROC','b  V5 ensemble precision–recall'],['False positive rate','Recall'],['True positive rate','Precision']):
        ax.set(title=title,xlabel=xlabel,ylabel=ylabel,xlim=(0,1),ylim=(0,1.02));ax.legend(loc='lower right' if ax is axes[0] else 'lower left')
    fig.suptitle('Internal Val: 189 patients (58 positive); 3-seed probability mean',fontsize=10);fig.tight_layout();save(fig,'Fig01_V5_ROC_PR')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.4));x=np.arange(3)
    for i,metric in enumerate(['ROC_AUC','AP']):
        for j,arm in enumerate(['REAL_ONLY','ORIGINAL_PROGRESSION']):
            values=[float(cross[cross.version.eq(v)&cross.arm.eq(arm)][metric].iloc[0]) for v in ['V3','V4','V5']]
            axes[i].bar(x+(j-.5)*.32,values,width=.32,color=COLORS[j],hatch=None if j==0 else '//',label='REAL-only' if j==0 else 'Original Progression')
            for k,value in enumerate(values):axes[i].text(x[k]+(j-.5)*.32,value+.02+j*.035,f'{value:.3f}',ha='center',fontsize=8)
        axes[i].set(xticks=x,xticklabels=['V3\nvirtual token','V4\nvirtual token','V5\nExam fusion'],yticks=np.linspace(0,1,6),ylim=(0,1.03),ylabel=metric,title=('a  ' if i==0 else 'b  ')+'Cross-protocol '+metric)
    axes[0].legend(loc='lower left');fig.tight_layout();save(fig,'Fig02_V3_V4_V5_Descriptive_Comparison')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.4))
    for i,metric in enumerate(['ROC_AUC','AP']):
        for j,arm in enumerate(ARMS):
            d=seeds[seeds.arm.eq(arm)].set_index('seed');values=[float(d.loc[s,metric]) for s in [42,3407,2026]]+[float(ensemble[ensemble.arm.eq(arm)][metric].iloc[0])]
            axes[i].plot(np.arange(4)+(j-.5)*.1,values,marker='o' if j==0 else 's',color=COLORS[j],lw=.8,label='REAL-only' if j==0 else 'OP fusion')
        axes[i].set(xticks=np.arange(4),xticklabels=['42','3407','2026','Ensemble'],yticks=np.linspace(0,1,6),ylabel=metric,ylim=(0,1),title=('a  ' if i==0 else 'b  ')+'All seeds and ensemble')
    axes[0].legend(loc='lower left');fig.tight_layout();save(fig,'Fig03_V5_Seeds_and_Ensemble')
    # Anonymous internal structure: input grouping, equal-node aggregation and gate internals.
    fig,ax=plt.subplots(figsize=(10.8,5.4));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def box(x,y,w,h,text,color='#EFF3F7'):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.006',facecolor=color,edgecolor='#687A8A',lw=.8));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,lw=1,color='#687A8A'))
    ax.text(.02,.95,'a  Frozen queries attach only to the first later REAL Exam',fontsize=11,weight='bold')
    for x,label in [(.035,'REAL 1'),(.28,'REAL 2'),(.525,'REAL 3')]:box(x,.78,.13,.075,label)
    arrow((.165,.818),(.28,.818));arrow((.41,.818),(.525,.818))
    box(.17,.64,.15,.075,'query q₁ / q₂','#F9EDD9');arrow((.29,.715),(.345,.78))
    ax.text(.37,.665,'q after REAL 1 and before REAL 2\nNo added time token; no target-year input',fontsize=9,va='center')
    box(.755,.76,.21,.09,'REAL timeline, months,\ncausal / padding unchanged','#E8F0E7')
    ax.text(.02,.56,'b  Shared image encoder and masked residual fusion (512-D)',fontsize=11,weight='bold')
    box(.025,.39,.17,.105,'All REAL views\nShared ResNet18\nExam mean → rᵢ')
    box(.025,.20,.17,.12,'Each node: 6 images\nSame ResNet18\nnode mean → u_q','#F9EDD9')
    box(.24,.20,.16,.12,'Equal-node mean\nfor attached Exam\nsᵢ = mean_q u_q','#F9EDD9');arrow((.195,.26),(.24,.26))
    box(.445,.40,.16,.095,'REAL LayerNorm\nh_r = LN(rᵢ)');arrow((.195,.445),(.445,.445))
    box(.445,.21,.16,.095,'Synth LayerNorm\nh_s = LN(sᵢ)','#F9EDD9');arrow((.40,.26),(.445,.26))
    box(.65,.40,.16,.095,'[h_r,h_s] 1024→1\nSigmoid gate gᵢ','#EEE9F4')
    box(.65,.21,.16,.095,'h_s − h_r\n512→512 projection dᵢ','#EEE9F4')
    arrow((.605,.445),(.65,.445));arrow((.605,.26),(.65,.26));arrow((.57,.40),(.68,.305));arrow((.57,.305),(.68,.40))
    box(.85,.29,.12,.115,'fᵢ = rᵢ\n+ 0.1 gᵢ dᵢ\nif has_synth','#E8F0E7');arrow((.81,.445),(.85,.37));arrow((.81,.26),(.85,.32))
    # Explicit raw REAL-feature skip: the residual adds to r_i, not normalized h_r.
    ax.plot([.11,.11,.91],[.495,.525,.525],color='#687A8A',lw=1)
    arrow((.91,.525),(.91,.405));ax.text(.28,.531,'Raw REAL feature rᵢ skip (not h_r)',fontsize=8,va='bottom')
    box(.24,.025,.36,.10,'No attached nodes: exact fᵢ = rᵢ\nBypass both norms, gate and projection')
    box(.66,.025,.31,.13,'REAL tokens only → original month TPE\n1 causal Transformer layer (4 heads)\nLast REAL → Dropout / Linear → HT','#E8F0E7');arrow((.91,.29),(.91,.155));arrow((.60,.075),(.66,.075))
    fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.98);save(fig,'Fig04_V5_Exam_Fusion_Internal_Architecture')
    jout(root/'figures/export_qa/V5_EXPORT_QA.json',dict(status='PASS',backend='Python matplotlib',figures=receipts,clinical_image_pixels=0,actual_visual_review='NOT_RUN'))
    print(dict(figure_groups=4,automatic_export_QA='PASS',actual_visual_review='NOT_RUN'),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);main(p.parse_args().config)
