"""Prepare server-private fixed panels and blank review cards; never display or approve pixels."""
from __future__ import annotations
import argparse
import json
import secrets
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
from review_common import setup,read_json,csv_out,json_out,state,now


def image512(path):
    with Image.open(path) as im:im.load();return im.convert('RGB').resize((512,512),Image.Resampling.BILINEAR)


def panel(tiles,labels,columns,title,dest):
    assert len(tiles)==len(labels);rows=(len(tiles)+columns-1)//columns
    canvas=Image.new('RGB',(columns*532+20,rows*548+78),'white');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    draw.text((20,12),title,fill='black',font=font)
    draw.text((20,36),'PRIVATE / NOT REVIEWED / NOT PUBLICATION APPROVED; original pixels or labeled technical derivatives',fill='black',font=font)
    for i,(im,label) in enumerate(zip(tiles,labels)):
        x=20+(i%columns)*532;y=78+(i//columns)*548;canvas.paste(im,(x,y));draw.text((x,y+515),label,fill='black',font=font)
    canvas.save(dest,dpi=(600,600));return canvas.size


def overlay(im,row):
    a=im.copy();draw=ImageDraw.Draw(a)
    for d in json.loads(row.line_candidates_private_json):draw.line((d['x1'],d['y1'],d['x2'],d['y2']),fill=(255,200,0),width=2)
    for d in json.loads(row.cross_candidates_private_json):
        x,y=d['x'],d['y'];draw.rectangle((x-5,y-5,x+5,y+5),outline=(255,30,100),width=2)
    # Image positions, not anatomy; identical global rules in both branches.
    draw.rectangle((26,26,485,485),outline=(0,190,255),width=1)
    draw.rectangle((102,102,409,409),outline=(0,220,180),width=1)
    return a


FIELDS=['thyroid_like_region_visibility','anatomical_border_readability','local_texture_readability','line_or_cross_visible','line_candidate_interpretation']
SCORES=['thyroid_like_structure_identifiability','border_anatomical_plausibility_subjective','echo_texture_naturalness','unnatural_geometry_severity_HIGH_IS_WORSE','overall_clinical_credibility_subjective']


def main(output_root):
    root=setup(output_root);assert read_json(root/'release/LOCALIZATION_COMPLETE_V1.json')['status']=='COMPLETE'
    complete=root/'release/PRIVATE_PANEL_PREPARATION_COMPLETE_V1.json'
    if complete.exists():print('Existing fixed private panels retained; no review status promotion');return
    core=pd.read_parquet(root/'manifests/CORE_ON_OFF_PAIRED_PANEL72_V1.parquet');sample=pd.read_parquet(root/'manifests/STRATIFIED_ARTIFACT_REVIEW48_V1.parquet')
    blind=pd.read_parquet(root/'manifests/CLINICIAN_BLIND_SET_V1.parquet');q=pd.read_parquet(root/'manifests/PRIVATE_UPSTREAM_IMAGE_QUEUE_V1.parquet')
    loc=pd.read_parquet(root/'metrics/FULL_QUEUE_FLAG_LOCATION_DIAGNOSTICS_V1.parquet').set_index('image_id')
    outputs=[];dest=root/'figures/private_medical_panels'
    def save(name,kind,tiles,labels,cols,title,**kw):
        p=dest/(name+'.png');dims=panel(tiles,labels,cols,title,p)
        outputs.append(dict(file_private=str(p),group=kind,tile_count=len(tiles),dimensions=f'{dims[0]}x{dims[1]}',
                            agent_medical_visual_review='NOT_RUN',human_review='NOT_RUN',public_approval='PENDING',**kw))
    for code,g in core.groupby('panel_node_code',sort=True):
        g=g.sort_values('image_index');assert len(g)==6
        imgs=[];labels=[];aux=[];al=[];dif=[];dl=[]
        for r in g.itertuples(index=False):
            a=image512(r.png_abs_path_on);b=image512(r.png_abs_path_off);imgs.extend([a,b]);labels.extend([f'ON index {r.image_index}',f'OFF index {r.image_index}'])
            for branch,name in [('FULL_B16V1_HIST','Full route-only'),('PLAIN_SD_HIST','Plain route-only')]:
                z=q[(q.branch==branch)&(q.anonymous_node_key==r.anonymous_node_key_on)&(q.image_index==r.image_index)]
                assert len(z)==1 and int(z.iloc[0].seed)==int(r.master_seed)
                aux.append(image512(z.iloc[0].png_abs_path));al.append(f'{name} index {r.image_index}')
            # Fixed gray difference on the original RGB512 grid; same0..255 mapping for all cases.
            delta=np.abs(np.asarray(a).astype(np.float64)-np.asarray(b).astype(np.float64)).mean(2)
            heat=Image.fromarray(np.repeat(np.rint(delta).astype(np.uint8)[...,None],3,2))
            ac=a.crop((192,192,320,320)).resize((512,512),Image.Resampling.NEAREST)
            bc=b.crop((192,192,320,320)).resize((512,512),Image.Resampling.NEAREST)
            dif.extend([a,b,heat,ac,bc]);dl.extend([f'ON index {r.image_index}',f'OFF index {r.image_index}','Mean absolute RGB difference0..255','ON fixed center crop x4','OFF fixed center crop x4'])
        section=g.iloc[0].display_section
        save('MedicalPanel_A_'+code,'MedicalPanel_A_Core12_ON_OFF_All6Seed',imgs,labels,6,code+' / all six indices / ON-OFF pairs',section=section)
        save('MedicalPanel_B_'+code,'MedicalPanel_B_Core12_Full_Plain_Aux',aux,al,6,code+' / auxiliary generation routes, NOT isolated CPFM effects',section=section)
        save('MedicalPanel_D_'+code,'MedicalPanel_D_Uniform_Difference_and_Crops',dif,dl,5,code+' / unchanged originals, fixed differences and center crops (NOT thyroid ROI)',section=section)
    for i,r in enumerate(sample.itertuples(index=False)):
        a=image512(r.png_abs_path_on);b=image512(r.png_abs_path_off)
        save(f'MedicalPanel_C_Q{i+1:02d}','MedicalPanel_C_Artifact48_Original_and_Overlay',
             [a,overlay(a,loc.loc[r.image_id_on]),b,overlay(b,loc.loc[r.image_id_off])],
             ['ON original','ON technical candidate positions','OFF original','OFF technical candidate positions'],2,
             f'Frozen stratum {r.stratum} / Q{i+1:02d}; yellow line / pink cross / cyan geometric zones; no anatomy inference',stratum=r.stratum)
    csv_out(dest/'PRIVATE_MEDICAL_PANEL_MANIFEST_V1.csv',outputs)
    (dest/'MedicalPanel_E_Real_Reference_LocalOnly_WITHHELD.md').write_text('# REAL_VISUAL_REFERENCE=WITHHELD\n\nNo approved local display and verified de-identification route has been supplied. No REAL pixel panels were rendered or displayed.\n',encoding='utf-8')
    rows=[]
    for r in blind.itertuples(index=False):
        for side,branch in [('on','OP_CPFM_ON'),('off','OP_CPFM_OFF')]:
            rows.append(dict(image_private=getattr(r,'png_abs_path_'+side),image_id_private=getattr(r,'image_id_'+side),
                         anonymous_patient_key_private=r.anonymous_patient_key_on,anonymous_node_key_private=r.anonymous_node_key_on,
                         branch_private=branch,image_index_private=r.image_index))
    rng=np.random.default_rng(20261012);order=rng.permutation(len(rows));mapping=[]
    for presentation,j in enumerate(order,1):
        r=rows[int(j)];code='C'+secrets.token_hex(8);p=root/'figures/private_expert_review_cards'/(code+'.png')
        im=image512(r['image_private']);im.save(p,dpi=(600,600))
        r.update(blind_card_id=code,presentation_order=presentation,card_private=str(p),derived_whole512=True,
                 original_pixel_file_unchanged=True,PHI_clearance='PENDING',display_approval='NOT_APPROVED',branch_labels_visible=False)
        mapping.append(r)
    csv_out(root/'manifests/PRIVATE_RANDOMIZED_IMAGE_ID_MAP_V1.csv',mapping)
    fields=['blind_card_id','reviewer_code','observer_role','review_quality_status','reviewed_utc','blind_single_image_stage']+SCORES+['notes_without_PHI']
    csv_out(root/'reviews_private/blank_clinician_review_form.csv',pd.DataFrame([dict(blind_card_id=r['blind_card_id'],observer_role='NONE',review_quality_status='NOT_VIEWED',blind_single_image_stage='PENDING_APPROVAL') for r in mapping],columns=fields))
    csv_out(root/'metrics/CLINICIAN_BLIND_RATINGS_V1.csv',pd.DataFrame(columns=fields))
    forms=[]
    for cohort,frame in [('CORE',core),('STRATIFIED',sample)]:
        for i,r in enumerate(frame.itertuples(index=False),1):
            for side in ['on','off']:
                forms.append(dict(review_code=f'{cohort}_{i:03d}_{side.upper()}',image_id_private=getattr(r,'image_id_'+side),
                           cohort=cohort,observer_role='NONE',reviewer_code='',reviewed_utc='',branch_labels_visible=True,
                           review_quality_status='NOT_VIEWED',free_text_observation='',**{f:'NOT_REVIEWED' for f in FIELDS}))
    csv_out(root/'reviews_private/blank_review_form.csv',forms)
    csv_out(root/'reviews_private/reviewers_metadata_private.csv',pd.DataFrame(columns=['reviewer_code','actual_role','professional_background','credential_verified_by_human','approved_local_review','approval_record','blind_stage','scale_version']))
    guide='''# Private review preparation / 私有复核准备

No review is approved or performed by generating these assets. Do not open pixel cards outside the institution-approved local route. Branch labels are removed from card presentation only; perceptual inference may still break blinding. No anonymization/PHI certification is implied.
生成这些资产不代表已经获批或开展评审；未经机构批准不得在外部查看。去除分支标签不保证感知盲法；不宣称已脱敏。

Actual observer roles: NONE, MACHINE_PROXY, NONCLINICAL_HUMAN, CODEX_VISUAL, CLINICIAN. A machine never obtains a human grade. NOT_REVIEWED means no observation; UNCERTAIN means an actual observer was unsure. A derivative-only review cannot assert full-original review.
记录真实角色；机器不写人工分级。NOT_REVIEWED 是未观察，UNCERTAIN 是观察后不确定。只看派生图不能声明看过完整原图。

Visibility: CLEAR/PARTIAL/NOT_IDENTIFIABLE/UNCERTAIN/NOT_REVIEWED; texture also SMOOTHED/NOISY. Line visibility: PRESENT/ABSENT/UNCERTAIN/NOT_REVIEWED. Possible cause: POSSIBLE_DEVICE_UI/POSSIBLE_ACQUISITION_MARK/POSSIBLE_GENERATIVE_ARTIFACT/AMBIGUOUS/NOT_REVIEWED. These are observations, not diagnoses or exact future-year truth.
上述类别仅表示观察，不是诊断或未来真值。

Clinician scale v1, precommitted before review: 1-5. For structure, border, texture and overall subjective credibility, high=more favorable; artifact-severity item high=more severe. Structure/texture anchor1=not interpretable,3=partly interpretable,5=clearly interpretable; subjective border/credibility anchor1=low,3=intermediate,5=high; geometry severity1=no apparent unnatural geometry,3=moderate,5=prominent. UNSURE/NOT_ASSESSABLE/SKIPPED are valid and not zeros. No disease certification. No patient-based calibration or fabricated example scores. No illustrative pixel examples were provided; clinician scale training remains an unmet prerequisite.
医生量表预声明1–5分：结构、边界、纹理和整体主观可信度高分较有利；几何伪影严重度高分更严重。允许不确定、无法评价、跳过，不能当0分。未提供非患者示例，医生量表培训仍待开展。

Any future expert import must have authentic role/qualification, local approval, full image viewing logs and complete fixed72-card single-image stage before unblinding. At least two actual clinicians are required for inter-rater consistency; a single clinician only supports single-rater subjective description. Independent unit is patient, not image.
未来评分导入必须有真实资格、批准、完整图像查看记录，并完成全部固定72卡单图阶段后才解盲。至少两名真实医生才能评价评分者间一致性；统计主要单元为患者。
'''
    (root/'reviews_private/review_guideline_cn_en.md').write_text(guide,encoding='utf-8')
    interface=root/'reviews_private/review_interface_local_only'
    (interface/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>PRIVATE pending approval</title><h1>Review NOT_RUN; no pixel interface activated</h1><p>Instituted local authorization and PHI clearance are prerequisites. Prepared cards remain in a separate protected directory. No external links, web service or remote image display.</p>',encoding='utf-8')
    json_out(complete,dict(status='PREPARED_NOT_REVIEWED',medical_panel_PNGs=len(outputs),blind_card_PNGs=len(mapping),blank_nonclinical_form_rows=len(forms),
                          fixed_indices=[0,2,5],expert_candidate_pairs=36,actual_observers=0,actual_scores=0,PHI_clearance='PENDING',real_panels=0,
                          external_medical_pixel_transfer=False,completed_utc=now()))
    state(root,'PRIVATE_REVIEW_ASSETS_PREPARED_NOT_REVIEWED',panels=len(outputs),blind_cards=len(mapping));print({'private_panels':len(outputs),'blind_cards':len(mapping),'actual_review':'NOT_RUN'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',required=True);a=p.parse_args();main(a.output_root)
