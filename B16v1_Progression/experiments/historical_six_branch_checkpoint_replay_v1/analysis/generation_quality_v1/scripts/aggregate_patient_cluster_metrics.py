"""Image/node/exam/patient denominators, equal-patient estimates and fixed cluster CIs."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from b16_quality_common import setup, atomic_parquet, write_csv, state


def summarize(values):
    v=np.asarray(values,dtype=float);assert np.isfinite(v).all()
    return dict(mean=float(v.mean()),median=float(np.median(v)),sd=float(v.std(ddof=1)) if len(v)>1 else 0.0,
                q25=float(np.quantile(v,.25)),q75=float(np.quantile(v,.75)),p025=float(np.quantile(v,.025)),p975=float(np.quantile(v,.975)))


def patient_ci(values):
    v=np.asarray(values,dtype=float);rng=np.random.default_rng(20261012)
    sampled=v[rng.integers(0,len(v),size=(2000,len(v)))].mean(1)
    lo,hi=np.quantile(sampled,[.025,.975]);return float(lo),float(hi)


def patient_values(frame,metrics):
    # Each native exam or generated node is one intermediate unit, never a copied K6 LOCF slot.
    f=frame.copy(); f['unit']=np.where(f.branch=='REAL_HISTORY_REFERENCE',f.real_exam_key_private,f.anonymous_node_key)
    return f.groupby(['branch','anonymous_patient_key','unit'],sort=True)[metrics].mean().groupby(['branch','anonymous_patient_key'],sort=True).mean().reset_index()


def main(output_root):
    root=setup(output_root);qa=pd.read_parquet(root/'metrics/IMAGE_DECODE_AND_TECHNICAL_QA_V1.parquet')
    metrics=['gray_mean','gray_std','gray_p01','gray_p99','dynamic_range_p99_p01','local_std_median','local_std_q25','local_std_q75',
             'sobel_gradient_mean','laplacian_variance','high_frequency_fraction','black_fraction','white_fraction','border_black_fraction','border_edge_fraction']
    flags=['low_contrast_flag','very_smooth_flag','extreme_black_flag','extreme_white_flag','high_frequency_flag','long_line_flag','cross_proxy_flag']
    patient=patient_values(qa,metrics+flags);atomic_parquet(root/'metrics/PRIVATE_TECHNICAL_PATIENT_AGGREGATES_V1.parquet',patient)
    texture=[];artifact=[]
    for b,g in qa.groupby('branch',sort=False):
        pv=patient[patient.branch==b];nodes=g[g.anonymous_node_key!=''].anonymous_node_key.nunique();exams=g[g.real_exam_key_private!=''].real_exam_key_private.nunique()
        for metric in metrics:
            lo,hi=patient_ci(pv[metric]);texture.append(dict(branch=b,metric=metric,images=len(g),nodes=nodes,exams=exams,patients=len(pv),
                                 patient_equal_mean=float(pv[metric].mean()),patient_ci_low=lo,patient_ci_high=hi,
                                 image_mean=float(g[metric].mean()),image_median=float(g[metric].median()),
                                 image_sd=float(g[metric].std()),image_q25=float(g[metric].quantile(.25)),image_q75=float(g[metric].quantile(.75)),
                                 image_p025=float(g[metric].quantile(.025)),image_p975=float(g[metric].quantile(.975)),
                                 unit='pixel gray0..255 or image proxy as prespecified',clinical_score=False))
        for metric in flags:
            lo,hi=patient_ci(pv[metric]);artifact.append(dict(branch=b,proxy=metric,flagged_images=int(g[metric].sum()),images=len(g),nodes=nodes,patients=len(pv),
                             image_flag_fraction=float(g[metric].mean()),patient_equal_fraction=float(pv[metric].mean()),patient_ci_low=lo,patient_ci_high=hi,
                             status='EXPLORATORY_PROXY_NOT_CLINICAL_ARTIFACT_DIAGNOSIS',expert_review='NOT_RUN'))
    write_csv(root/'metrics/TEXTURE_AND_ARTIFACT_DESCRIPTIVES_V1.csv',texture);write_csv(root/'metrics/STRUCTURE_EXPLORATORY_V1.csv',artifact)
    paired=pd.read_parquet(root/'metrics/PAIRED_ON_OFF_PIXEL_FEATURE_STATS_V1.parquet');aux=pd.read_parquet(root/'metrics/PAIRED_FULL_ON_AUXILIARY_ROUTE_STATS_V1.parquet')
    changes=[];pa=[]
    for d in [paired,aux]:
        comparison=d.comparison.iloc[0]
        node=d.groupby(['anonymous_patient_key','anonymous_node_key'])[['rgb_mae','rgb_rmse','gray_ssim','lpips','bit_exact']].mean()
        pv=node.groupby('anonymous_patient_key').mean();pv['comparison']=comparison;pa.append(pv.reset_index())
        for metric in ['rgb_mae','rgb_rmse','gray_ssim','lpips','bit_exact']:
            lo,hi=patient_ci(pv[metric]);s=summarize(d[metric])
            changes.append(dict(comparison=comparison,metric=metric,images=len(d),nodes=len(node),patients=len(pv),
                                **{'image_'+k:v for k,v in s.items()},patient_equal_mean=float(pv[metric].mean()),patient_ci_low=lo,patient_ci_high=hi,
                                zero_response_images=int(d.bit_exact.sum()),meaning='paired change, not reconstruction accuracy'))
    write_csv(root/'metrics/ON_OFF_PATIENT_AGGREGATE_V1.csv',pd.concat(pa,ignore_index=True))
    write_csv(root/'metrics/PAIRED_CHANGE_AGGREGATES_V1.csv',changes)
    div=pd.read_parquet(root/'metrics/NODE_SIX_SEED_DIVERSITY_V1.parquet');dr=[]
    for b,g in div.groupby('branch'):
        node=g.groupby(['anonymous_patient_key','anonymous_node_key'])[['rgb_rmse','gray_ssim','lpips','bit_exact']].mean();pv=node.groupby('anonymous_patient_key').mean()
        for metric in ['rgb_rmse','gray_ssim','lpips','bit_exact']:
            lo,hi=patient_ci(pv[metric]);dr.append(dict(branch=b,metric=metric,seed_pairs=len(g),nodes=len(node),patients=len(pv),images=len(node)*6,
                                        node_mean=float(node[metric].mean()),patient_equal_mean=float(pv[metric].mean()),
                                        patient_ci_low=lo,patient_ci_high=hi,**{'pair_'+k:v for k,v in summarize(g[metric]).items()}))
    write_csv(root/'metrics/NODE_SEED_DIVERSITY_AGGREGATES_V1.csv',dr)
    write_csv(root/'metrics/METRIC_NOT_AVAILABLE_OR_EXCLUDED_V1.csv',[
        dict(metric='clinical_expert_review',status='NOT_RUN',reason='No expert review performed; score template remains empty'),
        dict(metric='medical_agent_visual_review',status='NOT_RUN',reason='No task-specific authorization to transfer medical images'),
        dict(metric='thyroid_ROI_Dice_or_anatomy_score',status='NOT_AVAILABLE',reason='No validated annotated thyroid ROI assets with applicable-domain provenance supplied'),
        dict(metric='FID_patient_CI',status='NOT_RUN',reason='Resource-bounded omission frozen before scores; paired KID cluster CI computed'),
        dict(metric='target_year_reconstruction_accuracy',status='NOT_ESTABLISHED',reason='REAL historical context is not query-year paired ground truth'),
        dict(metric='new_independent_HT_validation',status='NOT_RUN',reason='This task is offline image quality evaluation only')])
    state(root,'PATIENT_AGGREGATION_COMPLETED',technical_rows=len(texture),proxy_rows=len(artifact),paired_rows=len(changes),diversity_rows=len(dr))
    print({'technical_rows':len(texture),'proxy_rows':len(artifact),'paired_rows':len(changes),'diversity_rows':len(dr)},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',required=True);a=p.parse_args();main(a.output_root)
