"""Only actual two-arm patient ensembles, paired internal-Val contrasts and safe aggregate tables."""
import argparse
import numpy as np
import pandas as pd
from common import config,setup,jread,jout,cout,postflight,ARMS,SEEDS,now,state
from runtime_ht import paired_bootstrap


def main(path):
    c=config(path);root=setup(c['project_private']);rows=[]
    for arm in ARMS:
        for seed in SEEDS:
            run=jread(root/'checkpoints'/arm/f'seed_{seed}'/'RUN_COMPLETED.json');assert run['status']=='COMPLETE_RELOAD_VERIFIED'
            rows.append(dict(arm=arm,seed=seed,best_epoch=run['best_epoch'],actual_epochs=run['actual_epochs'],reload=run['reload'],**run['metrics']))
    real=pd.read_csv(root/'metrics/private'/f'{ARMS[0]}_THREE_SEED_ENSEMBLE.csv');op=pd.read_csv(root/'metrics/private'/f'{ARMS[1]}_THREE_SEED_ENSEMBLE.csv')
    assert len(real)==len(op)==189 and np.array_equal(real.sample_index,op.sample_index) and np.array_equal(real.y_true,op.y_true)
    cout(root/'tables/LTSA_TWO_ARM_ALL_SIX_RUNS_V3.csv',rows)
    ensemble=pd.concat([pd.read_csv(root/'tables'/f'{arm}_ENSEMBLE_METRICS_V3.csv') for arm in ARMS],ignore_index=True)
    cout(root/'tables/LTSA_TWO_ARM_ENSEMBLE_COMPARISON_V3.csv',ensemble)
    intervals=paired_bootstrap(real.y_true,real.p_positive,op.p_positive,c['bootstrap_repeats'],c['bootstrap_seed'])
    cout(root/'tables/LTSA_OP_MINUS_REAL_PAIRED_BOOTSTRAP_V3.csv',intervals)
    cout(root/'tables/LTSA_TWO_ARM_SEED_VARIABILITY_V3.csv',pd.DataFrame(rows).groupby('arm')[['ROC_AUC','AP']].agg(['mean','std','min','max']).reset_index().pipe(lambda d:d.set_axis(['arm','ROC_AUC_mean','ROC_AUC_SD','ROC_AUC_min','ROC_AUC_max','AP_mean','AP_SD','AP_min','AP_max'],axis=1)))
    cout(root/'tables/LTSA_VS_CLIMAT_WITHIN_PREDICTOR_REFERENCE_V3.csv',[dict(predictor='CLIMATv2_historical_runbook_reference',REAL_ROC_AUC=.763885,OP_ROC_AUC=.871808,delta_OP_minus_REAL=.871808-.763885,
            status='REFERENCE_FROM_PROVIDED_WORK_DOCUMENT_NOT_RECOMPUTED',protocol='historical fixedK6/full9270synthetic; not same LTSA native/bounded686 protocol'),
            dict(predictor='LTSA_HT_adapted',REAL_ROC_AUC=float(ensemble.iloc[0].ROC_AUC),OP_ROC_AUC=float(ensemble.iloc[1].ROC_AUC),delta_OP_minus_REAL=float(ensemble.iloc[1].ROC_AUC-ensemble.iloc[0].ROC_AUC),
            status='ACTUALLY_MEASURED_TWO_ARM_PATIENT_ENSEMBLES',protocol='allnative observedhistory;686bounded synthetic query tokens; no OfficialTest')])
    postflight(root);status=dict(status='TWO_ARM_COMPLETED',runs=6,arms=ARMS,seeds=SEEDS,Val_N=189,all_strict_independent_reload='PASS',ensemble_definition='mean positive probability',
                 paired_bootstrap_repeats=c['bootstrap_repeats'],eligible_synthetic_nodes=686,eligible_synthetic_views=4116,source_protection='PASS',new_SD_sampling=0,new_upstream_training=0,
                 Test_access=0,clinical_certification=False,git_status='PUSH_NOT_DONE_UNTIL_ARCHIVED',nonmedical_visual_QA='NOT_RUN_UNTIL_ACTUALLY_VIEWED',completed_utc=now())
    jout(root/'releases/FINAL_LTSA_TWO_ARM_STATUS_V3.json',status);state(root,'TWO_ARM_TRAINING_RELOAD_ENSEMBLE_AND_COMPARISON_COMPLETE',completed_runs=6)
    print(status,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();main(a.config)
