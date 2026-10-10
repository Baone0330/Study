"""Draw eight historical paper figures from verified aggregate and architecture source data."""
from __future__ import annotations
import argparse
import csv
import json
import math
import textwrap
import time
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'pdf.fonttype':42,'font.size':7,'axes.spines.right':False,'axes.spines.top':False,
                    'axes.linewidth':.65,'legend.frameon':False,'savefig.facecolor':'white'})
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from PIL import Image
import numpy as np

COLORS=['#7D8795','#759AC3','#B1C7DF','#76ABA8','#B1A0C9','#7461A6']
SHORT=['REAL-only','Full B16v1','Plain-SD','history_only','temporal_only','Original\nProgression']
INK='#263846'; BLUE='#E3EDF7'; TEAL='#E1F0EE'; PURPLE='#EAE5F3'; GOLD='#F5EADB'; GREY='#F1F3F5'
NAMES=['Fig01_Historical_Architecture','Fig02_Original_Progression_Detail','Fig03_B14_CPFM_UNet_Injection',
       'Fig04_CLIMATv2_Multiview_K6','Fig05_Six_Branch_ROC_AP','Fig06_Threshold_Metric_Comparison',
       'Fig07_Descriptive_Differences','Fig08_Replay_Evidence_Workflow']


def csvrows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def diagram(ax,label,title):
    ax.set_xlim(0,100); ax.set_ylim(0,100);ax.axis('off')
    ax.text(0,100,label,fontweight='bold',fontsize=8,va='top',color=INK)
    ax.text(4,100,title,fontweight='bold',fontsize=7.6,va='top',color=INK)


def box(ax,x,y,w,h,label,color=BLUE,fs=6.2,edge='#8497A8',dashed=False):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.15,rounding_size=1',linewidth=.65,
                    facecolor=color,edgecolor=edge,linestyle='--' if dashed else '-')
    ax.add_patch(p)
    def luminance(c):
        z=matplotlib.colors.to_rgb(c);v=[q/12.92 if q<=.04045 else ((q+.055)/1.055)**2.4 for q in z]
        return .2126*v[0]+.7152*v[1]+.0722*v[2]
    bg=luminance(color)
    def contrast(c):
        fg=luminance(c);return (max(fg,bg)+.05)/(min(fg,bg)+.05)
    textcolor=INK if contrast(INK)>=4.5 else max(['#000000','#FFFFFF'],key=contrast)
    t=ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fs,color=textcolor,linespacing=1.15)
    t._paper_box=(ax,x,y,w,h)
    return p


def arrow(ax,x1,y1,x2,y2,color='#617A8F',dashed=False):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=6,
               linewidth=.75,color=color,linestyle='--' if dashed else '-',shrinkA=1,shrinkB=1))


def line(ax,x1,y1,x2,y2,color='#617A8F',dashed=False):
    ax.plot([x1,x2],[y1,y2],color=color,lw=.7,ls='--' if dashed else '-')


def note(ax,x,y,text,fs=5.9,ha='left',color='#5A6977'):
    return ax.text(x,y,text,fontsize=fs,ha=ha,va='center',color=color,linespacing=1.2)


def strip(ax,x,y,w,h,labels,color=BLUE,fs=6.1,gap=2):
    n=len(labels); bw=(w-gap*(n-1))/n
    for i,s in enumerate(labels):
        bx=x+i*(bw+gap); box(ax,bx,y,bw,h,s,color,fs)
        if i<n-1:arrow(ax,bx+bw,y+h/2,bx+bw+gap,y+h/2)


def tensor(ax,x,y,cols=4,rows=3,cell=1.8,color='#C2D3E4'):
    for i in range(rows):
        for j in range(cols):
            ax.add_patch(Rectangle((x+j*cell,y+i*cell),cell*.9,cell*.9,fc=color,ec='#8BA0B3',lw=.3))


def page(height,ratios):
    fig=plt.figure(figsize=(183/25.4,height/25.4))
    gs=fig.add_gridspec(len(ratios),1,height_ratios=ratios,left=.045,right=.985,top=.967,bottom=.04,hspace=.12)
    return fig,[fig.add_subplot(gs[i,0]) for i in range(len(ratios))]


def footer(fig,text='Historical internal Val, n=189 | three-seed probability ensemble | no new experiment'):
    fig.text(.5,.012,text,ha='center',va='bottom',fontsize=5.7,color='#576878')


def fig01(facts):
    fig,axs=page(180,[1.25,1.05,1.1]);a,b,c=axs
    diagram(a,'a','Historical inputs and condition construction')
    box(a,1,53,16,29,'Pre-query REAL\nimage sets\nBEFORE / AFTER',GREY)
    tensor(a,4,28);tensor(a,11,28);note(a,8.8,19,'symbolic tensors, not patient images',5.5,ha='center')
    arrow(a,17,67,21,67)
    box(a,21,45,28,43,'8 grayscale statistics + 6 report zeros\nIndependent BEFORE / AFTER MLP\nLinear 14→32 · LN · GELU\nLinear 32→16 · set mean',BLUE)
    box(a,21,8,28,25,'Query numeric 9→24\nLN · GELU · Linear 24→16',GOLD)
    arrow(a,49,65,55,65);arrow(a,49,21,53,21);line(a,53,21,53,54);arrow(a,53,54,55,54)
    box(a,55,45,20,43,'Full condition bridge\nconcat48 · Linear48→32\nLN · GELU\nLinear32→16 · tanh',TEAL)
    box(a,55,8,20,25,'Original P16 alternative\ndelta16 → masked mean\nfusion64→128→64',PURPLE,5.9)
    arrow(a,42,45,42,38);line(a,42,38,61,38);arrow(a,61,38,61,33)
    arrow(a,49,14,55,14)
    arrow(a,75,20,78,20);line(a,78,20,78,52);arrow(a,75,65,81,65)
    box(a,81,46,18,40,'Condition c16\nFull or Original P16\n(not added together)',PURPLE)
    arrow(a,78,52,81,52)
    note(a,81,20,'P16 adapter: 64→16+tanh\nSeparate from B14 CPFM',5.6)
    diagram(b,'b','Frozen historical diffusion stage')
    strip(b,1,48,98,37,['c16 + t8 + layer8\nzero history16\ncurrent activation16',
           'B14 CPFM\nbounded magnitudes\nzero-referenced directions',
           'SD1.5 UNet\nD0–D3 ResNet pre-hooks\nMedical LoRA + text',
           'PNDM epsilon\n30 steps · CFG 7\nVAE decoder 512²'],PURPLE,6.0,3)
    note(b,1,32,'Hook input channels: 320 / 320 / 640 / 1280; no up-path B14 hook.',6.1)
    note(b,1,18,'Existing offline generation: six images/node; no report, pathology, structure or texture input.',5.9)
    note(b,1,5,'No joint optimization with CLIMATv2; no diffusion generation was run in this packaging stage.',5.8)
    diagram(c,'c','Independent historical downstream training and internal-Val replay')
    strip(c,1,51,98,34,['REAL + existing synthetic\nFrozen CAP2 slot allocation\nREAL > B16 > LOCF',
           'Fixed K6 CLIMATv2\nResNet18 + view mean\nspatial / sequence FeaT',
           'Three selected checkpoints\nseeds 42 / 3407 / 2026\nPreviously replayed',
           'Val189 HT probability\nmatch patient keys\nmean of three probabilities'],BLUE,6.0,3)
    box(c,15,9,70,26,'Table01: six branches, same historical internal Val\nAP = average_precision_score; native thresholds unchanged\nNot independent clinical Test or a single-factor P16 causal estimate',GREY,6.1)
    arrow(c,87,51,87,41);line(c,87,41,50,41);arrow(c,50,41,50,35)
    footer(fig,'Historical architecture | source-verified original modules | later Clean / Strict excluded')
    return fig


def fig02(facts):
    fig,axs=page(170,[.8,1.,1.15,.95]);a,b,c,d=axs
    diagram(a,'a','Frozen per-exam features and adjacent changes')
    for i,(x,s) in enumerate(zip([1,24,47,70],['e₁ ∈ R16','e₂ ∈ R16','…','eL ∈ R16'])):
        box(a,x,51,19,31,s,BLUE,6.5)
        if i<3:arrow(a,x+19,66,x+23,66)
    note(a,1,31,'Each e: AFTER ExamSetEncoder, Linear14→32 / LN / GELU / Linear32→16 / set mean.',6)
    note(a,1,12,'Only pre-query REAL exams enter inference; Δk = e(k+1) − ek.',6.1)
    diagram(b,'b','Shared delta encoder and valid-mask pooling')
    strip(b,1,44,98,39,['Δ ∈ R(L−1)×16','Linear16→32','LayerNorm32\nGELU','Linear32→16',
            'Mask-weighted mean\np ∈ R16'],PURPLE,6.2,2)
    box(b,12,5,77,22,'p = Σ mk encoder(Δk) / max(Σ mk, 1)\nFewer than two exams: one zero delta with mask0 → p=0',GREY,6.1)
    arrow(b,91,44,91,16);arrow(b,91,16,89,16)
    diagram(c,'c','Four-input fusion and the separate output adapter')
    for i,s in enumerate(['BEFORE b16','AFTER a16','Query q16','Progression p16']):
        x=3+i*24;box(c,x,71,21,17,s,BLUE if i<2 else GOLD if i==2 else PURPLE)
        arrow(c,x+10.5,71,x+10.5,65);line(c,x+10.5,65,50,65)
    line(c,50,65,8.333,65);arrow(c,8.333,65,8.333,58)
    strip(c,1,23,98,35,['Concat64','Linear64→128','LN128\nGELU','Linear128→64','Linear64→16','tanh → c16'],PURPLE,5.9,2)
    note(c,1,9,'Fusion hidden64; cpfm_adapter is Linear64→16, NOT the B14 controller. Parameters: 19,008.',5.9)
    diagram(d,'d','Original historical supervision, not new training')
    box(d,1,20,30,64,'Frozen teacher condition\nnext historical exam\nFrozen old bridge condition\nnext AFTER feature delta',TEAL,6.1)
    box(d,36,20,63,64,'L = MSE(c, teacher) + 0.20(1−cos(c, teacher))\n + 0.50 MSE(p, target delta) + 0.10(1−cos(p, target delta))\n + 0.10 MSE(c, old)\nOriginal upstream seed20260921; selected epoch120\nP16 route replaces the Full fusion, not a pure additive ablation',GREY,5.9)
    arrow(d,31,52,36,52,dashed=True)
    footer(fig,'Original Progression only | source: original trainer / generator | no S2 / Clean P16')
    return fig


def fig03(facts):
    fig,axs=page(190,[1.65,.85,1.2,.95]);a,b,c,d=axs
    diagram(a,'a','B14 controller: magnitude, direction and patient residual branches')
    labels=[['c16 + timestep8 + layer8\nconcat32','LN32 · Linear32→128 · SiLU\nLinear128→128 · SiLU',
             'Per-layer magnitude heads128→1\nsigmoid aγ and aβ'],
            ['c16 + projected history16\n+ projected current16 + t8 + layer8\nconcat64',
             'LN64 · Linear64→128 · SiLU\nLinear128→128 · SiLU',
             'Direction heads128→Cl\nTwo passes: c and zero-c'],
            ['c16 + zero history16 → concat32\nLN32 · Linear32→32 · SiLU\nLinear32→16 → z16',
             'concat[c16,z16,current16,t8,layer8]64\nLN64 · Linear64→64 · SiLU',
             'Patient heads64→Cl\nTwo passes: c and zero-c']]
    for i,col in enumerate(labels):
        x=1+i*33
        for y,h,s in zip([66,36,7],[23,23,22],col):box(a,x,y,31,h,s,[BLUE,TEAL,PURPLE][i],5.9)
        arrow(a,x+15.5,66,x+15.5,59);arrow(a,x+15.5,36,x+15.5,29)
    note(a,1,1,'Cl = 320 / 320 / 640 / 1280. Old raw γ/β heads are computed but discarded by effective().',5.5)
    diagram(b,'b','Zero-referenced direction, bounded strength and residual modulation')
    strip(b,1,45,98,41,['[D(c)−D(0)]\n+0.25[P(c)−P(0)]',
            'RMS normalize\ngamma / beta directions',
            '× magnitude sigmoid\nbeta × safety sigmoid\nc=0 → magnitude0',
            'u=(x−μHW)/σHW\ny=x+sl[((1+γ)u+β)−u]'],PURPLE,5.9,2)
    note(b,1,24,'sl = sigmoid(learned layer scale). History16=0; current16 = four activation statistics repeated ×4.',5.7)
    note(b,1,8,'Zero condition preserves x algebraically; condition_scale metadata is not consumed by this hook.',5.7)
    diagram(c,'c','Effective injection points within the configured SD1.5 UNet')
    for i,(x,inc,outc) in enumerate(zip([5,29,53,77],[320,320,640,1280],[320,640,1280,1280])):
        box(c,x,47,19,32,f'D{i}: in{inc} → out{outc}\nResNet ×2\n'+('cross-attention ×2\ndownsample' if i<3 else 'no cross-attention\nno downsample'),BLUE,5.6)
        box(c,x,82,19,9,'B14 pre-hook',PURPLE,5.5);arrow(c,x+9.5,82,x+9.5,79)
        if i<3:arrow(c,x+19,65,x+24,65)
    note(c,1,36,'Latent4 → Conv3×3 → D0. B14 modifies input to the first ResNet only; no up-path hook.',5.8)
    box(c,5,5,41,22,'Middle: ResNet · cross-attention · ResNet\nText context768; frozen medical LoRA',TEAL,5.7)
    box(c,53,5,43,22,'Up path: skip concatenation · ResNet blocks\nConfigured up attention → noise epsilon4',TEAL,5.7)
    arrow(c,87,47,87,31);arrow(c,87,31,25,31);arrow(c,25,31,25,27);arrow(c,46,16,53,16)
    diagram(d,'d','ResNet microstructure and frozen-library provenance boundary')
    strip(d,1,48,98,37,['GroupNorm32\nSiLU','Conv3×3\n+timestep Linear','GroupNorm32\nSiLU · Dropout','Conv3×3\n+ skip / Conv1×1'],TEAL,5.9,2)
    note(d,1,31,'UNet config: channels320/640/1280/1280, two down ResNets/block, cross-attention dim768.',5.9)
    note(d,1,16,'Current installed ResNet sources explain these interfaces; historical library revision is NOT_RECOVERED.',5.7)
    note(d,1,3,'B14 checkpoint: 93 tensors, 1,707,072 parameters; CPU strict schema load, no UNet or clinical forward.',5.7)
    footer(fig,'Historical B14CPFM + B14Hooks | exact effective() algebra | no later own-fit controller')
    return fig


def fig04(facts):
    fig,axs=page(190,[1.05,1.4,1.35,.55]);a,b,c,d=axs
    diagram(a,'a','Shared ResNet18 backbone and masked view aggregation')
    strip(a,1,56,98,32,['RGB256²\nvalid views','Conv7×7 s2\nBN · ReLU\nMaxPool3 s2',
            'Stage1 64\nBasicBlock×2\n64² map','Stage2 128\nBasicBlock×2\n32² map',
            'Stage3 256\nBasicBlock×2\n16² map','Stage4 512\nBasicBlock×2\n8² map'],BLUE,5.8,2)
    box(a,1,5,60,35,'BasicBlock: Conv3×3 → BN → ReLU → Conv3×3 → BN\n + identity / Conv1×1 s2 skip → ReLU\nStage2–4 first block downsamples; no final spatial GAP/FC call',TEAL,5.8)
    box(a,68,5,31,35,'Mask-mean over views\nExam512×8×8\n→ 64 patches ×512',BLUE,6.0)
    arrow(a,91,56,91,40)
    diagram(b,'b','Spatial Radiologist and constant-context GP FeaT')
    box(b,1,51,25,37,'64 image patches512\n+ learned CLS512\n+ positions',BLUE,6.0)
    box(b,32,51,30,37,'Radiologist FeaT512\ndepth1 · heads4\nFFN512→256→512',BLUE,6.0)
    box(b,1,7,25,33,'Report zeros8 + time zeros2\n+ presence0 → metadata11\nLinear11→64 · ReLU · LN64',GREY,5.8)
    box(b,32,7,30,33,'GP FeaT64, depth1\nheads4 · FFN64→64→64\nconstant context, still active',TEAL,5.8)
    box(b,70,51,29,37,'Concat image512 + GP64\n65 spatial tokens576\nFirst token = exam token576',PURPLE,6)
    box(b,70,7,29,33,'Per-exam prognosis FeaT576\nauxiliary five-logit output\nnot the sequence input',GREY,5.8)
    arrow(b,26,70,32,70);arrow(b,26,23,32,23);arrow(b,62,70,70,70)
    line(b,62,23,65,23);line(b,65,23,65,60);arrow(b,65,60,70,60);arrow(b,85,51,85,40)
    note(b,1,1,'The sequence token is Radiologist CLS + GP CLS, not a ResNet average or per-exam five logits.',5.7)
    diagram(c,'c','Six-slot masked sequence and the actual HT readout')
    strip(c,1,61,98,28,['Six exam tokens576\n+ wrapper CLS / positions',
                'Same prognosis FeaT576\nmasked depth1 · heads4',
                'CLS576 → LN576\nLinear576→288 · GELU',
                'Linear288→5\nwrapper Linear5→2\nSoftmax HT probability'],PURPLE,5.8,2)
    box(c,1,12,98,35,'Transformer zoom: LN → QKV Linear(D,3D) → 4-head masked attention → Linear(D,D) + residual\nLN → Linear(D,hidden) → GELU → Linear(hidden,D) + residual\nActual attention scaling D⁻¹/². Sequence D576, hidden256; ordered positions, no inter-visit mean pooling.',BLUE,5.9)
    arrow(c,39,61,39,47,dashed=True)
    note(c,1,3,'Actual T=6; wrapper learned position capacity7 (+CLS=8). Report/time content inactive; masks and positions active.',5.7)
    diagram(d,'d','Post-model aggregation (outside the learned network)')
    strip(d,1,10,98,69,['Three historical seed checkpoints\n42 / 3407 / 2026',
          '189 patient-matched probabilities\nequal arithmetic mean',
          'ROC-AUC / AP / native threshold metrics\nHistorical internal Val, not Official Test'],GREY,5.9,3)
    footer(fig,'Actual CLIMATv2HTWrapper + original FeaT | not a plain ResNet18 classifier')
    return fig


def style_metric(ax,title):
    ax.set_ylim(0,1);ax.set_yticks([0,.25,.5,.75,1]);ax.set_title(title,loc='left',fontweight='bold',fontsize=8)
    ax.set_axisbelow(True);ax.grid(axis='y',color='#E6E9ED',lw=.5);ax.set_ylabel('Score (0–1)',fontsize=6.8)
    ax.tick_params(labelsize=6.2,length=2);ax.set_xticks(np.arange(6));ax.set_xticklabels(SHORT,rotation=28,ha='right',fontsize=6)


def metric_bars(ax,data,key):
    v=np.array([float(r[key]) for r in data]); bars=ax.bar(np.arange(6),v,width=.7,color=COLORS,edgecolor=INK,lw=.35)
    for i,(bar,x) in enumerate(zip(bars,v)):ax.text(bar.get_x()+bar.get_width()/2,x+.012,f'{x:.6f}',ha='center',va='bottom',fontsize=5.65,rotation=0)
    assert np.array_equal(np.array([b.get_height() for b in bars]),v)


def fig05(data):
    fig,axs=plt.subplots(1,2,figsize=(183/25.4,112/25.4));fig.subplots_adjust(left=.07,right=.99,bottom=.25,top=.9,wspace=.21)
    for ax,key,title in zip(axs,['ROC_AUC','AP'],['a  ROC-AUC','b  AP (average_precision_score)']):style_metric(ax,title);metric_bars(ax,data,key)
    footer(fig,'Historical internal Val, n=189 (HT0=131, HT1=58) | three-seed probability ensemble | no CI / p-values')
    return fig


def fig06(data):
    fig,axs=plt.subplots(2,2,figsize=(183/25.4,168/25.4));fig.subplots_adjust(left=.07,right=.99,bottom=.14,top=.94,wspace=.21,hspace=.72)
    for ax,key,label in zip(axs.flat,['Accuracy','Sensitivity','Specificity','F1'],['a','b','c','d']):style_metric(ax,label+'  '+key);metric_bars(ax,data,key)
    footer(fig,'Thresholds unchanged from each native historical evaluation | n=189 | no new threshold optimization')
    return fig


def fig07(data):
    labels=['Original P16 − Full','Original P16 − REAL','Original P16 − Plain','Full − REAL','Plain − Full']
    fig,axs=plt.subplots(1,2,figsize=(183/25.4,112/25.4));fig.subplots_adjust(left=.25,right=.98,bottom=.19,top=.9,wspace=.22)
    y=np.arange(len(data))[::-1]
    for i,(ax,key,title) in enumerate(zip(axs,['delta_ROC_AUC','delta_AP'],['a  ΔROC-AUC','b  ΔAP'])):
        vals=np.array([float(r[key]) for r in data]);ax.axvline(0,color='#9DA8B3',lw=.7,ls='--');ax.scatter(vals,y,s=28,c=[COLORS[5],COLORS[5],COLORS[5],COLORS[1],COLORS[2]],edgecolor=INK,lw=.4)
        ax.set_xlim(-.015,.235);ax.set_ylim(-.7,4.7);ax.set_yticks(y);ax.set_yticklabels(labels if i==0 else ['']*5,fontsize=6.6)
        ax.set_xticks([0,.05,.1,.15,.2]);ax.set_xlabel('Descriptive point difference',fontsize=6.7);ax.set_title(title,loc='left',fontweight='bold',fontsize=8);ax.tick_params(labelsize=6.3)
        ax.grid(axis='x',color='#ECEFF2',lw=.5);ax.set_axisbelow(True)
        for yy,v in zip(y,vals):ax.annotate(f'+{v:.6f}',(v,yy),xytext=(5,0),textcoords='offset points',ha='left',va='center',fontsize=6)
    footer(fig,'No confidence intervals, p-values or isolated causal component effects are claimed.')
    return fig


def fig08():
    fig,axs=page(145,[1.7,.75]);a,b=axs
    diagram(a,'a','Existing independent checkpoint forwards and probability ensembles')
    for x,label in [(1,'Branch'),(27,'Original selected checkpoints'),(62,'Existing replay outputs'),(85,'Ensemble')]:note(a,x,88,label,6.4)
    branch=['REAL-only','Full B16v1','Plain-SD','history_only','temporal_only','Original Progression']
    for i,s in enumerate(branch):
        y=74-i*12;box(a,1,y,22,10,s,COLORS[i],6.1,edge='#738699')
        for j,seed in enumerate([42,3407,2026]):box(a,27+j*10.5,y,9,10,str(seed),GREY,5.7)
        arrow(a,23,y+5,27,y+5);arrow(a,57,y+5,61,y+5)
        box(a,61,y,21,10,'3 × 189 positive scores',GREY,5.7);arrow(a,82,y+5,86,y+5)
        box(a,86,y,13,10,'mean p(HT)',COLORS[i],5.7)
    note(a,1,3,'Forward happened in the preceding stage: strict complete state load, eval/inference mode, original dataset and metrics.',5.7)
    diagram(b,'b','Version and release evidence, not additional clinical validation')
    strip(b,1,43,98,43,['18 real forwards\n6 complete ensembles\n189 shared Val patients',
             '42 display fields matched\n6-decimal reference equality\nnot bit-exact probabilities',
             'Replay base Git commit\nd97379a\n32 existing public files protected'],BLUE,6.0,3)
    note(b,1,24,'This paper stage: new training0 / new generation0 / new clinical forward0 / Official Test0.',6.0)
    note(b,1,7,'Safe public sources contain aggregate tables and nonmedical figures; clinical weights and patient rows stay private.',5.7)
    footer(fig,'Historical computation reproducibility ≠ independent clinical generalization')
    return fig


CAPTIONS=[
('Historical architecture and stage separation','a Original F16 and query encoders provide the Full condition or the replacement Original Progression route. b Frozen B14 hooks modulate the configured diffusion UNet. c Existing synthetic training inputs and CLIMATv2 classification belong to separate historical stages. No new generation, training or forward was performed in this packaging stage. Later Clean/Strict systems are excluded. Source data are provided in the corresponding source CSV.',
 '历史架构与阶段分离','a 原F16及查询编码器形成Full条件或替换式Original Progression条件。b 冻结B14通过hook调制扩散UNet。c 既有合成训练输入与CLIMATv2分类属于独立历史阶段。本封存阶段没有新生成、训练或前向，后来Clean/Strict系统不参与。对应源CSV提供结构证据。'),
('Original Progression layer structure','a Adjacent differences are constructed from frozen pre-query exam representations. b A shared delta MLP and valid-mask mean produce p16. c Four 16-dimensional inputs enter 64→128→64 fusion and a separate 64→16 tanh adapter; the adapter is not B14 CPFM. d The original historical losses and source roles are shown. Zero valid deltas imply p=0, not necessarily a zero final condition. Source data and original-source anchors accompany the figure.',
 'Original Progression层结构','a 冻结查询前检查表征构成相邻差分。b 共享差分MLP与有效mask均值得到p16。c 四项16维输入进入64→128→64融合及独立64→16 tanh适配头，适配头不是B14 CPFM。d 展示原历史损失与输入角色。没有有效差分意味着p=0，不意味着最终条件必为零。附源数据与源码锚点。'),
('Historical B14 modulation and effective UNet injection','a Magnitude, zero-referenced direction and patient residual sub-networks. b Directions combine with bounded magnitudes and residual activation modulation. c Pre-hooks operate at the first ResNet input of D0–D3, with channels320/320/640/1280. d Configured ResNet interfaces are shown; the exact historical installed-library revision is not recovered. No new UNet forward or medical image test was run. Source data include the strict-loaded module inventory.',
 '历史B14调制及有效UNet注入','a 展示幅值、零参考方向和病人残差子网络。b 方向与有界幅值结合，形成激活残差调制。c pre-hook位于D0–D3首个ResNet输入，通道为320/320/640/1280。d 展示配置支持的ResNet接口；原安装库精确revision未恢复。未执行新UNet前向或医学影像测试。源数据包括严格加载模块盘点。'),
('Actual multi-view fixed-K6 CLIMATv2','a Valid images pass through ResNet18 and within-exam masked feature-map mean; the standard GAP/FC is inactive. b Spatial Radiologist and constant-zero-input GP FeaT produce exam tokens. c The original prognosis Transformer is reused for the masked six-slot sequence, followed by the five-logit decoder and two-logit HT projection. d Three-seed averaging occurs after model output. Report/time content is inactive, but masks and learned positions remain active. Source data are provided.',
 '实际多图固定K6 CLIMATv2','a 有效图像经ResNet18并在检查内按mask平均特征图；标准GAP/FC不活动。b 空间Radiologist与零输入常量上下文GP FeaT形成检查token。c 原prognosis Transformer复用于六槽mask序列，接五logit解码和二logit HT投影。d 三seed平均位于模型输出之后。报告/时间内容不活动，mask与学习位置仍活动。附源数据。'),
('Six-branch historical discrimination scores','a ROC-AUC. b AP, defined as average_precision_score, not trapezoidal PR integration. Bars show equal-mean positive-probability ensembles from seeds42,3407,2026 on the same historical internal Val (n=189; HT0=131, HT1=58). No variability, CI or p-values are estimated here. Plain-SD above Full is retained. Source data are provided as Fig05_source_data.csv.',
 '六分支历史区分指标','a ROC-AUC。b AP定义为average_precision_score，不是PR梯形积分。柱形来自42、3407、2026三个种子正类概率等权ensemble，同一历史内部Val共189人（HT0=131、HT1=58）。本轮不估计变异、CI或p值，完整保留Plain-SD高于Full的结果。源数据见Fig05_source_data.csv。'),
('Threshold-dependent historical metrics','a Accuracy. b Sensitivity. c Specificity. d F1. Values use each branch’s unchanged native historical threshold strategy, with positive probability greater than or equal to the chosen threshold; n=189, three-seed probability ensemble. No threshold was newly tuned for this figure. Different error trade-offs are retained. Source data are provided as Fig06_source_data.csv.',
 '阈值相关历史指标','a Accuracy。b Sensitivity。c Specificity。d F1。各值沿用该分支原历史阈值策略，正类判定为概率大于或等于阈值；n=189、三seed概率ensemble。本图未重新调阈值，保留不同错误类型权衡。源数据见Fig06_source_data.csv。'),
('Descriptive historical ensemble differences','a ROC-AUC differences. b AP differences. Points are automatically subtracted from the full-precision six-branch table; positive values favour the first named condition. These are observed point differences, not isolated causal module effects. No CI or p-value is computed. The Plain-SD–Full comparison is included explicitly. Source data are provided as Fig07_source_data.csv.',
 '描述性历史ensemble差值','a ROC-AUC差值。b AP差值。点由完整精度六分支表自动相减，正值有利于前一个条件。这些是观察到的点差值，不是独立因果模块效应。未计算CI或p值，明确包括Plain-SD−Full比较。源数据见Fig07_source_data.csv。'),
('Historical replay provenance and evidence','a Six branches each map to three original selected checkpoints and previously recomputed patient-matched probability outputs; only the counts, not patient records, are displayed. b Eighteen real forwards, six ensembles and42 matching display fields are tied to the existing public replay commit. New forwards in this paper stage are zero. Reproducibility is not independent clinical validation. Source data are provided as Fig08_source_data.csv.',
 '历史重放来源与证据','a 六分支各对应三个原selected checkpoint与前阶段真实重新计算的配对概率输出；仅展示计数，不展示患者记录。b 18次真实前向、6个ensemble与42项匹配展示字段连接到既有公开重放commit。本论文封存阶段新增前向为零；计算可复现性不等于独立临床验证。源数据见Fig08_source_data.csv。')]


def fit_and_check(fig):
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    fitted=0
    for t in fig.findobj(match=matplotlib.text.Text):
        if hasattr(t,'_paper_box'):
            ax,x,y,w,h=t._paper_box
            p0=ax.transData.transform((x,y));p1=ax.transData.transform((x+w,y+h))
            while t.get_fontsize()>5.25:
                b=t.get_window_extent(renderer)
                if b.width<=(p1[0]-p0[0])*.92 and b.height<=(p1[1]-p0[1])*.88:break
                t.set_fontsize(t.get_fontsize()-.15);fitted+=1;fig.canvas.draw();renderer=fig.canvas.get_renderer()
            b=t.get_window_extent(renderer)
            if b.width>(p1[0]-p0[0])*.98 or b.height>(p1[1]-p0[1])*.97:
                raise ValueError('Box label overflow: '+t.get_text())
    fig.canvas.draw();renderer=fig.canvas.get_renderer();border=fig.bbox;outside=[]
    for t in fig.findobj(match=matplotlib.text.Text):
        if not t.get_visible() or not t.get_text():continue
        b=t.get_window_extent(renderer)
        if b.x0<border.x0-1 or b.y0<border.y0-1 or b.x1>border.x1+1 or b.y1>border.y1+1:outside.append(t.get_text())
    if outside:raise ValueError('Figure text outside export: '+repr(outside))
    fonts=[t.get_fontsize() for t in fig.findobj(match=matplotlib.text.Text) if t.get_text()]
    return dict(label_bounds='PASS',minimum_font_pt=min(fonts),font_fit_adjustments=fitted)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    root=args.root;start=time.time();data=csvrows(root/'figures/source_data/Table01_Historical_Six_Branch_Ensemble.csv');diff=csvrows(root/'figures/source_data/Table02_Historical_Six_Branch_Differences.csv')
    facts=json.loads((root/'figures/source_data/ARCHITECTURE_FACTS_PUBLIC_V1.json').read_text())
    assert len(data)==6 and all(int(x['Val_N'])==189 for x in data)
    assert facts['Original_P16_seed']==20260921 and facts['CPFM']['hook_input_channels']==[320,320,640,1280]
    assert not facts['CLIMATv2']['standard_resnet_GAP_FC_active'] and facts['CLIMATv2']['T']==6
    builders=[lambda:fig01(facts),lambda:fig02(facts),lambda:fig03(facts),lambda:fig04(facts),lambda:fig05(data),lambda:fig06(data),lambda:fig07(diff),fig08]
    out=root/'figures/final';out.mkdir(parents=True,exist_ok=True);q=[];all_caps=[]
    for i,(name,build,cap) in enumerate(zip(NAMES,builders,CAPTIONS),1):
        fig=build();qa=fit_and_check(fig);base=out/name
        fig.savefig(str(base)+'.svg',metadata={'Date':None,'Creator':'Historical source-verified paper package'})
        fig.savefig(str(base)+'.pdf',metadata={'CreationDate':None,'ModDate':None})
        fig.savefig(str(base)+'.png',dpi=600)
        with Image.open(str(base)+'.png') as im:
            rgb=im.convert('RGB');rgb.save(str(base)+'.png',dpi=(600,600))
        with Image.open(str(base)+'.png') as im:
            assert im.mode=='RGB' and min(im.size)>2000
            size=im.size; dpi=im.info.get('dpi');arr=np.asarray(im);assert np.std(arr)>5
        tree=ET.parse(str(base)+'.svg');texts=tree.findall('.//{http://www.w3.org/2000/svg}text');assert len(texts)>10
        for fmt in ['svg','pdf','png']:assert (out/(name+'.'+fmt)).stat().st_size>1000
        caption=f'## Fig. {i} | {cap[0]}\n\n{cap[1]}\n\n## 图 {i} | {cap[2]}\n\n{cap[3]}\n'
        (root/f'figures/captions/Fig{i:02d}_figure_caption.md').write_text(caption,encoding='utf-8');all_caps.append(caption)
        q.append(dict(figure=name,auto_status='PASS',visual_review='NOT_RUN',png_mode='RGB',png_pixels=list(size),dpi=list(dpi),editable_SVG_text_elements=len(texts),
                      width_mm=183,height_mm=fig.get_size_inches()[1]*25.4,source_data=f'Fig{i:02d}_source_data.csv',**qa))
        plt.close(fig);print(json.dumps(dict(figure=name,auto_QA='PASS',png_pixels=list(size))),flush=True)
    tablecaps=[('Historical six-branch probability ensembles','The same historical internal Val contains189 patients (HT0=131, HT1=58); positive probabilities of seeds42,3407,2026 are equally averaged. AP is average_precision_score; other threshold metrics retain original branch-specific rules. No new independent Test is reported.',
          '历史六分支概率ensemble','同一历史内部Val189人（HT0=131、HT1=58），三个种子正类概率等权平均。AP为average_precision_score，阈值指标沿用原分支策略，无新增独立Test。'),
       ('Descriptive ensemble comparisons','Differences are derived automatically from full-precision Table01; no CI/p-value or independent causal component estimate is added.','描述性ensemble比较','差值从完整精度Table01自动计算，未增加CI/p值或独立因果模块估计。'),
       ('Historical selected checkpoint inventory','Private absolute locators remain on the server; the public table contains only semantic branch, seed, original best epoch and existence/size metadata. These weights were replayed in the preceding stage, not forwarded again here.','历史selected checkpoint盘点','具体私有路径留服务器；公开表只含语义分支、seed、原best epoch与存在/大小信息。这些权重在前阶段重放，本轮没有再次前向。'),
       ('Source-verified original layer inventory','Module attributes and learned-token/scale parameters come from original source and strict CPU schema loads. Inactive and metadata-only paths are marked. Complete SD/LoRA weights and historical library revisions are not claimed verified.','源码核验原版层清单','模块属性及学习token/尺度参数来自原源码与CPU严格schema加载，明确标出不活动/元数据路径，不声称已验证完整SD/LoRA权重或历史库revision。'),
       ('Claim evidence and limitations','Each statement is marked supported, observed, unsupported or not evaluated. Computation reproducibility does not establish external clinical efficacy.','陈述证据与局限','逐条标记支持、观察、不支持或未评价；计算可复现性不建立外部临床有效性。')]
    for i,(en,ep,cn,cp) in enumerate(tablecaps,1):all_caps.append(f'## Table {i} | {en}\n\n{ep}\n\n## 表 {i} | {cn}\n\n{cp}\n')
    (root/'writing/TABLE_AND_FIGURE_CAPTIONS_CN_EN_V1.md').write_text('\n'.join(all_caps),encoding='utf-8')
    (root/'figures/qa/AUTOMATED_FIGURE_QA_V1.json').write_text(json.dumps(dict(status='AUTO_PASS_MANUAL_NOT_RUN',figure_count=8,figures=q,wall_seconds=time.time()-start,
        no_medical_images=True,new_clinical_forward=0,confidence_intervals_computed=False,metric_axes='0–1',backend='Python/matplotlib'),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
