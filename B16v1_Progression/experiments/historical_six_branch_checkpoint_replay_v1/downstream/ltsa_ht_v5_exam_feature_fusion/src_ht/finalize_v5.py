"""Actual six-run inference aggregates; old predictions allowed only after all new runs."""
import argparse
import json
import numpy as np
import pandas as pd
from common import config,setup,jread,jout,cout,ARMS,SEEDS,now,postflight
from runtime_ht import metric_values,paired_bootstrap
from v5_safety import V3,V4,enable_history_reference
OLD_ARMS=['LTSA_REAL_ONLY','LTSA_ORIGINAL_PROGRESSION_HIST']
def main(path):
    c=config(path);root=setup(c['project_private']);enable_history_reference(root)
    seeds=[];ensembles=[];frames=[];initial=[]
    for arm in ARMS:
        for seed in SEEDS:
            run=root/'checkpoints'/arm/f'seed_{seed}';done=jread(run/'RUN_COMPLETED.json');start=jread(run/'RUN_INITIALIZATION_V5.json')
            assert done['status']=='COMPLETE_RELOAD_VERIFIED' and done['max_probability_abs_delta']<=1e-6
            assert start['fresh_model'] and start['fresh_optimizer'] and not start['old_checkpoint_loaded']
            initial.append({k:start[k] for k in ['arm','seed','pid','started_utc','trainable_parameters','optimizer_state_entries','fresh_model','fresh_optimizer','old_checkpoint_loaded']})
            seeds.append(dict(arm=arm,seed=seed,best_epoch=done['best_epoch'],actual_epochs=done['actual_epochs'],val_loss=done['val_loss'],reload=done['reload'],reload_max_abs=done['max_probability_abs_delta'],**done['metrics'],**{'fixed05_'+k:v for k,v in done['fixed05_metrics'].items()}))
        d=pd.read_csv(root/'metrics/private'/f'{arm}_THREE_SEED_ENSEMBLE.csv');assert len(d)==189 and d.sample_index.is_unique and np.isfinite(d.p_positive).all()
        assert d.y_true.value_counts().to_dict()=={0:131,1:58};frames.append(d)
        ensembles.append(dict(arm=arm,aggregation='arithmetic mean of 3 seed positive probabilities',**metric_values(d.y_true,d.p_positive),**{'fixed05_'+k:v for k,v in metric_values(d.y_true,d.p_positive,.5).items()}))
    a,b=frames;assert np.array_equal(a.sample_index,b.sample_index) and np.array_equal(a.y_true,b.y_true)
    assert len({r['pid'] for r in initial})==6,'Each arm/seed needs an independent process'
    intervals=paired_bootstrap(a.y_true,a.p_positive,b.p_positive,c['bootstrap_repeats'],c['bootstrap_seed'])
    cout(root/'tables/V5_SEED_LEVEL_METRICS.csv',seeds);cout(root/'tables/V5_THREE_SEED_ENSEMBLE.csv',ensembles);cout(root/'tables/V5_PAIR_BOOTSTRAP.csv',intervals)
    cross=[];diagnostics=[]
    for version,base in [('V3',V3),('V4',V4)]:
        for i,arm in enumerate(OLD_ARMS):
            d=pd.read_csv(base/'metrics/private'/f'{arm}_THREE_SEED_ENSEMBLE.csv')
            assert np.array_equal(d.sample_index,a.sample_index) and np.array_equal(d.y_true,a.y_true)
            met=metric_values(d.y_true,d.p_positive)
            expected=[(.859041853119242,.8109424012443032),(.7902079494603844,.6985018834028683)][i]
            assert abs(met['ROC_AUC']-expected[0])<1e-14 and abs(met['AP']-expected[1])<1e-14
            cross.append(dict(version=version,input_protocol='REAL tokens' if i==0 else '686 extra virtual query tokens',arm='REAL_ONLY' if i==0 else 'ORIGINAL_PROGRESSION',**met))
        # Q3 numerical diagnostics, not a tuning or completion performance gate.
        for seed in SEEDS+[None]:
            oldname=f'{OLD_ARMS[0]}_seed{seed}_val_reload.csv' if seed is not None else f'{OLD_ARMS[0]}_THREE_SEED_ENSEMBLE.csv'
            newname=f'{ARMS[0]}_seed{seed}_val_reload.csv' if seed is not None else f'{ARMS[0]}_THREE_SEED_ENSEMBLE.csv'
            old=pd.read_csv(base/'metrics/private'/oldname);new=pd.read_csv(root/'metrics/private'/newname)
            assert np.array_equal(old.sample_index,new.sample_index) and np.array_equal(old.y_true,new.y_true)
            delta=old.p_positive.to_numpy()-new.p_positive.to_numpy()
            diagnostics.append(dict(reference=version,seed=seed if seed is not None else 'ensemble',N=189,max_abs_probability_difference=float(np.abs(delta).max()),MAE=float(np.abs(delta).mean()),
                 rank_changed_count=int(np.sum(old.p_positive.rank(method='average').to_numpy()!=new.p_positive.rank(method='average').to_numpy())),within_1e6=bool(np.abs(delta).max()<=1e-6)))
    cross.extend(dict(version='V5',input_protocol='REAL tokens; shared fusion module identity' if i==0 else 'next-real Exam residual gated feature fusion, no extra token',arm='REAL_ONLY' if i==0 else 'ORIGINAL_PROGRESSION',**{k:v for k,v in row.items() if k not in ['arm','aggregation']}) for i,row in enumerate(ensembles))
    cout(root/'tables/V3_V4_V5_CROSS_PROTOCOL_COMPARISON.csv',cross);cout(root/'tables/V5_REAL_NUMERICAL_PARITY_DIAGNOSTIC.csv',diagnostics)
    protection=postflight(root)
    for r in initial:
        access=root/'audit'/f"FILE_ACCESS_BOUNDARY_V5_PID_{r['pid']}.json";assert access.is_file()
        counters=jread(access);assert counters['protected_write_attempts']==counters['old_trained_weight_read_attempts']==counters['old_probability_read_attempts']==counters['raw_target_image_reads']==0
    jout(root/'audit/V5_FRESH_TRAINING_EVIDENCE.json',dict(status='PASS',independent_processes=6,records=initial,old_checkpoint_or_prediction_loads_during_training=0,strict_reload_runs=6))
    (root/'audit/V5_FRESH_TRAINING_EVIDENCE.md').write_text('# Actual fresh training evidence\n\n'+json.dumps(initial,indent=2)+'\n\nAll six OWN best checkpoints strictly reloaded in independent processes within1e-6. Python open-event training counters contain no protected writes or old trained weight/probability reads; this is not an OS-wide proof. Full source metadata postflight checked separately. No smoke or previous-arm weights reused.\n')
    jout(root/'releases/FINAL_V5_STATUS.json',dict(RUNS_AND_METRICS_COMPLETE=True,DELIVERABLES_COMPLETE=False,GITHUB_PUSH_VERIFIED=False,
        status='RUNS_AND_METRICS_COMPLETE',completed_runs=6,strict_reloads=6,Val_N=189,bootstrap_repeats=2000,synthetic_nodes=686,synthetic_views=4116,
        source_metadata_protection=protection['status'],Official_Test_content_access=0,new_SD_sampling=0,visual_review='NOT_RUN',Git='NOT_ATTEMPTED',completed_utc=now()))
    print(dict(status='RUNS_AND_METRICS_COMPLETE',runs=6,ensembles=ensembles,paired_bootstrap=intervals,REAL_parity=diagnostics),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);main(p.parse_args().config)
