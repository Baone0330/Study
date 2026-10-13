"""Fresh offline initialized GPU forward/backward benchmark; freeze engineering settings before Val."""
import argparse
import gc
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch
import yaml
from common import setup,config,jread,jout,cout,state,now
from model_ht import LTSAHT
from dataset_ht import HTLongitudinalDataset,collate_patients,model_inputs
from runtime_ht import configure


def main(config_path):
    c=config(config_path);root=setup(c['project_private']);assert jread(root/'tests/ANONYMOUS_CPU_TEST_RESULTS_V3.json')['status']=='PASS'
    finished=root/'releases/TRAINING_CONTRACT_FROZEN_V3.json'
    if finished.exists():print('Same completed frozen training contract retained');return
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
    assert not apps.stdout.strip(),'Other GPU task present: queue instead of interrupting/sharing unbenchmarked memory'
    counts=pd.read_csv(root/'tables/REAL_COHORT_NATIVE_SUPPORT_V3.csv');max_month=int(counts.max_relative_month.max())
    dataset=HTLongitudinalDataset(root,'train');rank=sorted(range(len(dataset)),key=lambda i:dataset.records[i]['observed_history_view_count'],reverse=True)
    trials=[]
    for n in [1,2,4]:
        configure(42);model=LTSAHT(c,max_month,pretrained=True).to('cuda');optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
        batch=collate_patients([dataset[i] for i in rank[:n]]);args=model_inputs(batch,torch.device('cuda'));labels=batch['labels'].to('cuda');model.train()
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic();losses=[]
        for step in range(2):
            optimizer.zero_grad(set_to_none=True);logits=model(**args);assert logits.shape==(n,) and torch.isfinite(logits).all()
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels);assert torch.isfinite(loss);loss.backward()
            grads=[p.grad for p in model.parameters() if p.requires_grad];assert all(g is not None and torch.isfinite(g).all() for g in grads)
            optimizer.step();losses.append(float(loss.detach()))
        torch.cuda.synchronize();elapsed=time.monotonic()-start;peak=torch.cuda.max_memory_allocated()/1024**3
        row=dict(patient_microbatch=n,gradient_accumulation=8//n,effective_patient_batch=8,patient_views=sum(batch['exam_view_counts']),max_patient_views=max(dataset.records[i]['observed_history_view_count'] for i in rank[:n]),
                 two_optimizer_steps_seconds=elapsed,patients_per_second=2*n/elapsed,views_per_second=2*len(batch['images'])/elapsed,peak_allocated_GiB=peak,
                 logits_finite=True,all_trainable_grads_finite=True,optimizer_steps=2,precision='FP32',parameters=sum(p.numel() for p in model.parameters()),sm120=torch.cuda.get_device_capability()==(12,0),
                 protocol='largest native-history patients; no Val metrics; fresh pretrained model; smoke weights discarded')
        trials.append(row);print({'GPU_SMOKE':row},flush=True)
        del model,optimizer,args,labels,batch,logits,loss,grads;gc.collect();torch.cuda.empty_cache()
    eligible=[r for r in trials if r['peak_allocated_GiB']<=24];assert eligible
    chosen=max(eligible,key=lambda r:r['patients_per_second']);c['patient_microbatch']=chosen['patient_microbatch'];c['gradient_accumulation']=chosen['gradient_accumulation'];c['data_loader_workers']=2
    c['max_observed_relative_month']=max_month;c['tpe_max_len']=max_month+1;c['preliminary_config']=False
    c['autograd_activation_checkpointing']=False;c['BatchNorm']='standard_resnet18_train_running_stats; no freeze or checkpoint recomputation'
    c['augmentation_policy']='exact authority-associated pre-target-year historical Exam views only; no inferred/unknown-future mapping or new tokens unless separately confirmed'
    c['augmentation_policy_reason']='conservative preregistration for semantic audit; fixed before formal Val; if unsupported preserve REAL-only and stop OP'
    c['FP32_runtime']='deterministic_algorithms; TF32off; mathSDPA only; MHAfastpathoff; viewchunk16; sample loss once; final partial optimizer group normalized by actual patients'
    c['frozen_utc']=now()
    backup=Path(config_path).with_name(Path(config_path).name+'.bak_'+datetime.utcnow().strftime('%Y%m%dT%H%M%SZ'));shutil.copy2(config_path,backup)
    Path(config_path).write_text(yaml.safe_dump(c,sort_keys=False,allow_unicode=True),encoding='utf-8')
    cout(root/'tables/GPU_MEMORY_THROUGHPUT_SMOKE_V3.csv',trials)
    jout(finished,dict(status='PASS_FROZEN_BEFORE_FORMAL_VAL',chosen=chosen,config_private=str(config_path),all_native_history_retained=True,
               arms=c['arms'],seeds=c['seeds'],max_month=max_month,initialization_weights_smoke_not_reused=True,Val_access_for_benchmark=0,completed_utc=now()))
    (root/'audit/PRIVACY_BOUNDARY_AND_INCIDENT_V3.md').write_text('# Privacy boundary\n\nPatient pixels remain server-private; no medical images displayed or downloaded. Initial schema probing accidentally returned30 dictionary-key patient codes to an internal tool response. It was not a public repository upload; this must not be represented as zero identity-code tool output. Subsequent probes only emit schemas/counts, no real keys/dates/mappings. Public packaging must scan actual private values and exclude all per-patient tables, probabilities, dates, pixels and weights.\n',encoding='utf-8')
    state(root,'ANONYMOUS_CPU_AND_GPU_SMOKE_PASS_CONTRACT_FROZEN',chosen_microbatch=c['patient_microbatch'],gradient_accumulation=c['gradient_accumulation'],precision='FP32')
    print({'frozen_patient_microbatch':c['patient_microbatch'],'effective_batch':8,'GPU_smoke':'PASS','all_views':'RETAINED','formal_Val_observed':False},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();main(a.config)
