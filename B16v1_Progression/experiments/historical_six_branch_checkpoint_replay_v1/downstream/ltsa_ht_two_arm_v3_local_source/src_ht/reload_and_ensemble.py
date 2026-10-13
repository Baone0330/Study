"""Separate-process strict best-state loading and patient-positive-probability ensemble."""
import argparse
import pandas as pd
import numpy as np
import torch
from common import setup,config,jread,jout,cout,ARMS,SEEDS,now
from model_ht import LTSAHT
from dataset_ht import HTLongitudinalDataset
from train_ltsa_ht import evaluate
from runtime_ht import configure,metric_values


def main(path,arm,seed=None):
    c=config(path);root=setup(c['project_private'])
    if seed is not None:
        run=root/'checkpoints'/arm/f'seed_{seed}';assert (run/'TRAINING_COMPLETED_PENDING_RELOAD_V3.json').exists()
        if (run/'RUN_COMPLETED.json').exists():print({'reload':'EXISTING_VERIFIED_RETAINED','arm':arm,'seed':seed});return
        configure(seed);a=torch.load(run/'best.pt',map_location='cpu',weights_only=False);assert a['contract']==c and a['arm']==arm and a['seed']==seed
        model=LTSAHT(c,c['max_observed_relative_month'],pretrained=True).to('cuda');model.load_state_dict(a['model'],strict=True)
        d,loss=evaluate(model,HTLongitudinalDataset(root,'val',arm),c,seed);saved=pd.read_csv(root/'metrics/private'/f'{arm}_seed{seed}_best_saved.csv')
        assert np.array_equal(saved.sample_index,d.sample_index) and np.array_equal(saved.y_true,d.y_true) and len(d)==189
        delta=float(np.max(np.abs(saved.p_positive.to_numpy()-d.p_positive.to_numpy())));assert delta<=c['proba_reload_atol'],'Strict independent reload probability mismatch'
        cout(root/'metrics/private'/f'{arm}_seed{seed}_val_reload.csv',d)
        jout(run/'RUN_COMPLETED.json',dict(status='COMPLETE_RELOAD_VERIFIED',arm=arm,seed=seed,best_epoch=a['best_epoch'],actual_epochs=jread(run/'TRAINING_COMPLETED_PENDING_RELOAD_V3.json')['actual_epochs'],
             val_loss=loss,metrics=metric_values(d.y_true,d.p_positive),fixed05_metrics=metric_values(d.y_true,d.p_positive,.5),reload='PASS',max_probability_abs_delta=delta,
             independent_process=True,strict_state_dict=True,Val_N=189,Test_access=0,completed_utc=now()))
        print({'reload':'PASS','arm':arm,'seed':seed,'Val_N':189,'max_abs_delta':delta},flush=True)
    else:
        frames=[];rows=[]
        for seed in SEEDS:
            done=jread(root/'checkpoints'/arm/f'seed_{seed}'/'RUN_COMPLETED.json');assert done['status']=='COMPLETE_RELOAD_VERIFIED'
            frames.append(pd.read_csv(root/'metrics/private'/f'{arm}_seed{seed}_val_reload.csv'));rows.append(dict(arm=arm,seed=seed,best_epoch=done['best_epoch'],actual_epochs=done['actual_epochs'],val_loss=done['val_loss'],reload=done['reload'],**done['metrics']))
        for d in frames:assert np.array_equal(d.sample_index,frames[0].sample_index) and np.array_equal(d.y_true,frames[0].y_true)
        probabilities=np.stack([d.p_positive.to_numpy() for d in frames]).mean(0);out=frames[0].copy();out.p_positive=probabilities
        cout(root/'metrics/private'/f'{arm}_THREE_SEED_ENSEMBLE.csv',out);cout(root/'tables'/f'{arm}_THREE_SEED_METRICS_V3.csv',rows)
        metrics=metric_values(out.y_true,out.p_positive);fixed=metric_values(out.y_true,out.p_positive,.5)
        cout(root/'tables'/f'{arm}_ENSEMBLE_METRICS_V3.csv',[dict(arm=arm,aggregation='mean positive probabilities across3seeds',**metrics)])
        cout(root/'tables'/f'{arm}_FIXED_05_METRICS_V3.csv',[dict(arm=arm,**fixed)])
        status=dict(status='REAL_BASELINE_COMPLETED' if arm==ARMS[0] else 'OP_THREE_SEED_COMPLETED',arm=arm,seeds=SEEDS,Val_N=189,ensemble_metrics=metrics,fixed05_metrics=fixed,
                     all3_independent_reload='PASS',Test_access=0,completed_utc=now())
        jout(root/'releases'/('REAL_ONLY_STAGE_COMPLETE_V3.json' if arm==ARMS[0] else 'OP_STAGE_COMPLETE_V3.json'),status);print(status,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--arm',required=True,choices=ARMS);p.add_argument('--seed',type=int,choices=SEEDS);a=p.parse_args();main(a.config,a.arm,a.seed)
