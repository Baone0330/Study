"""Patient-equal technical differences; no human/clinical observations are invented."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from review_common import setup,csv_out,parquet_out,json_out,read_json,state


def summary(v):
    a=np.asarray(v,float)
    return dict(mean=float(a.mean()),median=float(np.median(a)),sd=float(a.std(ddof=1)) if len(a)>1 else 0,
                q25=float(np.quantile(a,.25)),q75=float(np.quantile(a,.75)),p025=float(np.quantile(a,.025)),p975=float(np.quantile(a,.975)))


def paired_bootstrap(on,off):
    a=np.asarray(on,float);b=np.asarray(off,float);assert a.shape==b.shape
    diff=a-b;rng=np.random.default_rng(20261012);s=diff[rng.integers(0,len(diff),size=(2000,len(diff)))].mean(1)
    lo,hi=np.quantile(s,[.025,.975]);return dict(on_patient_mean=float(a.mean()),off_patient_mean=float(b.mean()),
             delta_on_minus_off=float(diff.mean()),ci_low=float(lo),ci_high=float(hi),bootstrap_sd=float(s.std(ddof=1)),
             repeats=2000,seed=20261012,unit='paired patient',direction='ON minus OFF',clinical_quality_claim=False)


def human_agreement_unperformed():
    return dict(status='NOT_RUN',TP=None,FP=None,TN=None,FN=None,sensitivity=None,specificity=None,precision=None,
                kappa=None,icc=None,reason='No actually viewed, approved human reference annotation; proxies are not human truth')


def validate_observation(row,reviewer):
    role=row.get('observer_role','NONE');status=row.get('review_quality_status','NOT_VIEWED')
    visibility=row.get('thyroid_like_region_visibility','NOT_REVIEWED')
    if role in {'NONE','MACHINE_PROXY'}:
        assert visibility=='NOT_REVIEWED' and status=='NOT_VIEWED','Machine/unviewed rows must not acquire human visibility grades'
    if role=='CLINICIAN':
        assert reviewer and reviewer.get('credential_verified_by_human') is True and reviewer.get('approved_local_review') is True
        assert status=='FULL_IMAGE_VIEWED'
    return True


def main(output_root):
    root=setup(output_root);assert read_json(root/'release/LOCALIZATION_COMPLETE_V1.json')['status']=='COMPLETE'
    d=pd.read_parquet(root/'metrics/FULL_QUEUE_FLAG_LOCATION_DIAGNOSTICS_V1.parquet')
    metrics=['old_long_line_flag','old_cross_proxy_flag','line_candidates','cross_candidates','line_total_length_px','line_max_length_px']
    for kind in ['line','cross']:
        for zone in ['border','central','intermediate','multizone']:metrics.extend([f'{kind}_{zone}_any',f'{kind}_{zone}_candidate_count'])
    rows=[];patient_frames=[]
    for b,g in d.groupby('branch',sort=False):
        f=g.copy();f['unit']=np.where(f.branch=='REAL_HISTORY_REFERENCE',f.real_exam_key_private,f.anonymous_node_key)
        node=f.groupby(['anonymous_patient_key','unit'])[metrics].mean();patient=node.groupby('anonymous_patient_key').mean()
        pf=patient.reset_index();pf['branch']=b;patient_frames.append(pf)
        for m in metrics:
            image=summary(g[m]);ps=summary(patient[m]);rows.append(dict(branch=b,metric=m,images=len(g),intermediate_units=len(node),patients=len(patient),
                       image_sum=float(g[m].sum()),**{'image_'+k:v for k,v in image.items()},**{'patient_'+k:v for k,v in ps.items()},
                       method='view-to-exam-to-patient' if b=='REAL_HISTORY_REFERENCE' else 'image-to-six-seed-node-to-patient',
                       evidence_role='POSITIONAL_TECHNICAL_PROXY_NOT_ANATOMY',overlapping_regions_allowed=True))
    patients=pd.concat(patient_frames,ignore_index=True);parquet_out(root/'metrics/PRIVATE_PATIENT_TECHNICAL_AGGREGATES_V1.parquet',patients)
    csv_out(root/'metrics/FULL_QUEUE_REGION_LEVEL_AGGREGATES_V1.csv',rows)
    on=patients[patients.branch=='OP_CPFM_ON'].sort_values('anonymous_patient_key');off=patients[patients.branch=='OP_CPFM_OFF'].sort_values('anonymous_patient_key')
    assert on.anonymous_patient_key.tolist()==off.anonymous_patient_key.tolist() and len(on)==860
    cis=[]
    for m in metrics:cis.append(dict(metric=m,patients=860,nodes_per_arm=1545,images_per_arm=9270,**paired_bootstrap(on[m],off[m])))
    csv_out(root/'metrics/PAIR_CLUSTER_BOOTSTRAP_V1.csv',cis)
    pairs=pd.read_parquet(root/'manifests/ALL9270_PAIRS_AND_FLAG_REFERENCES_V1.parquet');agreements=[]
    for a,b,name in [('on_line','off_line','long_line_flag'),('on_cross','off_cross','cross_proxy_flag')]:
        x=pairs[a].astype(bool);y=pairs[b].astype(bool)
        agreements.append(dict(metric=name,pairs=len(pairs),both_proxy_true=int((x&y).sum()),ON_only_proxy_true=int((x&~y).sum()),
                               OFF_only_proxy_true=int((~x&y).sum()),both_proxy_false=int((~x&~y).sum()),
                               meaning='joint machine proxy status, NOT TP/FP/TN/FN versus anatomy or human truth'))
    csv_out(root/'metrics/FULL_QUEUE_PAIRED_PROXY_JOINT_COUNTS_V1.csv',agreements)
    csv_out(root/'metrics/PROXY_TO_ANNOTATION_AGREEMENT_V1.csv',[human_agreement_unperformed()])
    csv_out(root/'metrics/QUALITATIVE_PAIRED_OBSERVATIONS_V1.csv',pd.DataFrame(columns=['review_code','observer_role','thyroid_like_region_visibility','line_or_cross_visible','review_quality_status','free_text_observation']))
    csv_out(root/'metrics/METRIC_STATUS_AND_NA_REASON_V1.csv',[
        dict(metric='inherited_global_flags',status='REPRODUCED_WITH_ZERO_MISMATCH',reason='Actual prior per-image cache and unchanged flag algorithm'),
        dict(metric='region_location',status='COMPUTED_TECHNICAL_ONLY',reason='Frozen geometric zones; not thyroid ROI or confirmed UI'),
        dict(metric='human_visibility_distribution',status='NOT_RUN',reason='No medical pixels were viewed by an approved human/Codex observer'),
        dict(metric='proxy_human_TP_FP_TN_FN',status='NOT_RUN',reason='No actual human reference annotations'),
        dict(metric='clinician_scores_or_agreement',status='NOT_RUN',reason='No approved qualified clinician ratings; score cells remain empty'),
        dict(metric='thyroid_ROI_Dice_IoU',status='NOT_ESTABLISHED',reason='No validated domain ROI truth supplied; no external segmentation model introduced'),
        dict(metric='REAL_visual_reference',status='WITHHELD',reason='No approved verified de-identification/local viewing route supplied'),
        dict(metric='new_FID_KID',status='NOT_RUN_BY_SCOPE',reason='Prior completed CSV values are only referenced'),
        dict(metric='population_prevalence_from48',status='NOT_ESTABLISHED',reason='Nonuniform distinct-patient stratified sampling; no calibrated inclusion weights')])
    json_out(root/'release/TECHNICAL_AGGREGATION_COMPLETE_V1.json',dict(status='COMPLETE',patient_bootstrap_repeats=2000,primary_patients=860,
                    actual_human_observations=0,actual_clinician_scores=0,new_disease_inference=0))
    state(root,'PATIENT_TECHNICAL_AGGREGATION_COMPLETE',metrics=len(rows),paired_cluster_comparisons=len(cis),human_review='NOT_RUN')
    print({'aggregate_rows':len(rows),'patient_CI_comparisons':len(cis),'human_reference':'NOT_RUN'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',required=True);a=p.parse_args();main(a.output_root)
