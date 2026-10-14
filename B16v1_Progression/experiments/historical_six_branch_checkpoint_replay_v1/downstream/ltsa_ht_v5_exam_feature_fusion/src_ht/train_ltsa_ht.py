"""One fresh arm/seed process; patient loss, epoch-boundary resume and private Val probabilities."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Subset
from common import setup,config,jread,jout,cout,ARMS,SEEDS,now,state
from model_ht import LTSAHT,assert_finite_gradients
from dataset_ht import HTLongitudinalDataset,collate_patients,model_inputs
from runtime_ht import configure,worker_init,metric_values,selection_key,lr_for_epoch,rng_state,restore_rng,save_checkpoint
from v5_safety import validate_checkpoint_path,initialization_evidence


def loader(dataset,c,order=None,epoch=0,seed=0):
    from ipc_runtime import ensure_short_ipc
    ensure_short_ipc(dataset.root)
    subset=Subset(dataset,order) if order is not None else dataset
    g=torch.Generator().manual_seed(seed*100000+epoch)
    return DataLoader(subset,batch_size=c['patient_microbatch'],shuffle=False,num_workers=c['data_loader_workers'],collate_fn=collate_patients,
                      pin_memory=True,worker_init_fn=worker_init,generator=g,multiprocessing_context='spawn' if c['data_loader_workers'] else None)


def evaluate(model,dataset,c,seed=0):
    model.eval();indices=[];labels=[];probabilities=[];loss_sum=0
    with torch.inference_mode():
        for batch in loader(dataset,c,seed=seed):
            z=model(**model_inputs(batch,torch.device('cuda')));y=batch['labels'].to('cuda')
            assert torch.isfinite(z).all();loss_sum+=float(torch.nn.functional.binary_cross_entropy_with_logits(z,y,reduction='sum'))
            indices.extend(batch['sample_index'].tolist());labels.extend(batch['labels'].to(torch.int64).tolist());probabilities.extend(torch.sigmoid(z).cpu().tolist())
    assert len(indices)==len(set(indices))==189
    d=pd.DataFrame(dict(sample_index=indices,y_true=labels,p_positive=probabilities));return d,loss_sum/len(d)


def main(path,arm,seed,resume):
    c=config(path);root=setup(c['project_private']);assert arm in ARMS and seed in SEEDS and not c['preliminary_config']
    assert jread(root/'releases/TRAINING_CONTRACT_FROZEN_V3.json')['status']=='PASS_FROZEN_BEFORE_FORMAL_VAL'
    assert jread(root/'releases/ORIGINAL_PROGRESSION_LINEAGE_STATUS_V3.json')['status']=='PASS_MAPPING_FROZEN_BEFORE_FORMAL_VAL'
    if arm==ARMS[1]:assert jread(root/'releases/REAL_ONLY_STAGE_COMPLETE_V3.json')['status']=='REAL_BASELINE_COMPLETED'
    run=root/'checkpoints'/arm/f'seed_{seed}';run.mkdir(parents=True,exist_ok=True)
    if (run/'RUN_COMPLETED.json').exists():print({'arm':arm,'seed':seed,'existing_complete':'NO_RETRAIN'});return
    configure(seed);device=torch.device('cuda');assert torch.cuda.is_available()
    dataset=HTLongitudinalDataset(root,'train',arm);val=HTLongitudinalDataset(root,'val',arm)
    model=LTSAHT(c,c['max_observed_relative_month'],pretrained=True).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay']);shuffle=torch.Generator().manual_seed(seed)
    best_key=None;best_epoch=None;stale=0;start_epoch=1;history=[];last=run/'last_epoch.pt'
    validate_checkpoint_path(last,root);validate_checkpoint_path(run/'best.pt',root)
    if last.exists():
        assert resume,'Existing epoch checkpoint requires explicit valid-resume, not overwrite'
        a=torch.load(last,map_location='cpu',weights_only=False)
        assert a['task']==c['task'] and a['arm']==arm and a['seed']==seed and a['contract']==c,'Different frozen contract must never silently resume'
        model.load_state_dict(a['model'],strict=True);optimizer.load_state_dict(a['optimizer']);restore_rng(a['rng'],shuffle)
        best_key=a['best_key'];best_epoch=a['best_epoch'];stale=a['early_stopping_stale'];start_epoch=a['epoch']+1;history=a['history']
        print({'arm':arm,'seed':seed,'resume_from_completed_epoch':a['epoch'],'partial_epoch_replayed':True},flush=True)
    else:
        assert not (run/'best.pt').exists(),'Unexpected prior checkpoint must not silently warm start'
        jout(run/'RUN_INITIALIZATION_V3.json',dict(arm=arm,seed=seed,initialization='fresh officialImageNetResNet18+freshTransformer+HTHead, no previousarm warm start',
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),contract=c,created_utc=now()))
        initialization_evidence(root,run,c,arm,seed,model,optimizer)
    state(root,'FORMAL_TRAINING_RUNNING',arm=arm,seed=seed,start_epoch=start_epoch,pid=os.getpid(),precision='FP32',patient_microbatch=c['patient_microbatch'],effective_patient_batch=8)
    stop_existing=best_epoch is not None and (start_epoch-1)>=c['min_epoch'] and stale>=c['patience']
    epochs=range(start_epoch,c['max_epoch']+1) if not stop_existing else []
    for epoch in epochs:
        started=time.monotonic();model.train();order=torch.randperm(len(dataset),generator=shuffle).tolist();lr=float(lr_for_epoch(epoch,c))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True);processed=0;train_loss_sum=0;step=0;group_processed=0;effective=c['effective_patient_batch'];group_size=min(effective,len(dataset))
        torch.cuda.reset_peak_memory_stats()
        for batch in loader(dataset,c,order,epoch,seed):
            n=len(batch['labels']);logits=model(**model_inputs(batch,device));y=batch['labels'].to(device)
            loss_sum=torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='sum')
            assert torch.isfinite(logits).all() and torch.isfinite(loss_sum),'NaN/Inf engineering failure: stop'
            (loss_sum/group_size).backward();train_loss_sum+=float(loss_sum.detach());processed+=n;group_processed+=n
            if group_processed==group_size:
                assert_finite_gradients(model)
                optimizer.step();optimizer.zero_grad(set_to_none=True);step+=1;group_processed=0;group_size=min(effective,len(dataset)-processed)
            if processed%200==0 or processed==len(dataset):
                print({'arm':arm,'seed':seed,'epoch':epoch,'patients_processed':processed,'train_patients':len(dataset),'optimizer_steps':step,
                      'train_loss_so_far':train_loss_sum/processed,'lr':lr,'own_peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3},flush=True)
        assert processed==860 and group_processed==0
        prediction,val_loss=evaluate(model,val,c,seed);metrics=metric_values(prediction.y_true,prediction.p_positive)
        key=selection_key(metrics,val_loss,epoch);improved=best_key is None or key>best_key
        if improved:best_key=key;best_epoch=epoch;stale=0
        else:stale+=1
        seconds=time.monotonic()-started;row=dict(epoch=epoch,train_loss=train_loss_sum/processed,val_loss=val_loss,lr=lr,epoch_seconds=seconds,selected_best=improved,
                        optimizer_steps=step,own_peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,**metrics)
        history.append(row)
        checkpoint=dict(task=c['task'],arm=arm,seed=seed,contract=c,model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=dict(last_epoch=epoch,formula='5warmup_then_cosine_to60',next_lr=float(lr_for_epoch(min(epoch+1,60),c))),
                        epoch=epoch,best_key=best_key,best_epoch=best_epoch,early_stopping_stale=stale,rng=rng_state(shuffle),history=history,completed_epoch_boundary=True)
        if improved:
            save_checkpoint(run/'best.pt',checkpoint);cout(root/'metrics/private'/f'{arm}_seed{seed}_best_saved.csv',prediction)
        save_checkpoint(last,checkpoint);cout(run/'train_log.csv',history)
        jout(run/'EPOCH_BOUNDARY_STATE_V3.json',dict(arm=arm,seed=seed,last_completed_epoch=epoch,best_epoch=best_epoch,stale=stale,max_epoch=c['max_epoch'],latest_val=metrics,updated_utc=now()))
        print({'EPOCH_COMPLETE':row,'arm':arm,'seed':seed,'best_epoch':best_epoch,'stale':stale},flush=True)
        if epoch>=c['min_epoch'] and stale>=c['patience']:break
    assert best_epoch is not None and (run/'best.pt').exists()
    jout(run/'TRAINING_COMPLETED_PENDING_RELOAD_V3.json',dict(status='TRAINING_COMPLETED_PENDING_INDEPENDENT_RELOAD',arm=arm,seed=seed,best_epoch=best_epoch,
          actual_epochs=int(history[-1]['epoch']),best_selection_key=best_key,fresh_initialization=True,Test_access=0,completed_utc=now()))
    state(root,'ONE_RUN_TRAINED_PENDING_INDEPENDENT_RELOAD',arm=arm,seed=seed,best_epoch=best_epoch,actual_epochs=int(history[-1]['epoch']))
    print({'status':'TRAINING_COMPLETED_PENDING_RELOAD','arm':arm,'seed':seed,'best_epoch':best_epoch,'actual_epochs':int(history[-1]['epoch'])},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--arm',required=True,choices=ARMS);p.add_argument('--seed',required=True,type=int,choices=SEEDS);p.add_argument('--resume-if-valid',action='store_true');a=p.parse_args();main(a.config,a.arm,a.seed,a.resume_if_valid)
