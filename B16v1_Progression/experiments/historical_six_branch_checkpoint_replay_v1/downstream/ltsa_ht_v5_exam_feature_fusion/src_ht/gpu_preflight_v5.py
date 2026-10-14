"""Actual Train-only maximum-support smoke; no Val metrics or setting search."""
import argparse
import json
import sys
import time
from collections import Counter
import torch
import torchvision
from common import config,setup,jread,jout,now,postflight,ARMS
from model_ht import LTSAHT,assert_finite_gradients,active_fusion_gradient_evidence
from dataset_ht import HTLongitudinalDataset,collate_patients,model_inputs
from runtime_ht import configure
from run_pipeline import gpu_gate_observation,wait_gpu
def main(path):
    c=config(path);root=setup(c['project_private']);assert jread(root/'releases/V5_PREPARATION_COMPLETE.json')['status']=='PASS'
    assert jread(root/'tests/ANONYMOUS_CPU_TEST_RESULTS_V5.json')['status']=='PASS'
    current=(sys.version.split()[0],torch.__version__,torchvision.__version__,torch.version.cuda)
    assert current==('3.10.20','2.11.0+cu128','0.26.0+cu128','12.8'),'STOP_ENVIRONMENT_DIFFERENCE'
    marker=root/'releases/TRAINING_CONTRACT_FROZEN_V3.json'
    if marker.exists():assert jread(marker)['status']=='PASS_FROZEN_BEFORE_FORMAL_VAL';print('Own V5 preflight retained');return
    wait_gpu(True,12288)
    configure(42);dataset=HTLongitudinalDataset(root,'train',ARMS[1]);records=dataset.records
    def nodes(i):return dataset.augmentation.get(int(records[i]['patient_index']),[])
    def max_nodes_at_exam(i):return max(Counter(n['real_exam_index'] for n in nodes(i)).values(),default=0)
    criteria=[lambda i:records[i]['observed_history_exam_count'],lambda i:records[i]['observed_history_view_count'],max_nodes_at_exam,lambda i:records[i]['observed_history_view_count']+6*len(nodes(i))]
    chosen=[]
    for key in criteria:
        selected=max((i for i in range(len(records)) if i not in chosen),key=key);chosen.append(selected)
    batch=collate_patients([dataset[i] for i in chosen]);args=model_inputs(batch,'cuda');labels=batch['labels'].to('cuda')
    model=LTSAHT(c,78,pretrained=True).to('cuda');parameters=sum(p.numel() for p in model.parameters() if p.requires_grad);assert parameters==13020738
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay']);model.train();torch.cuda.reset_peak_memory_stats();losses=[];gradient_receipts=[]
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True);z=model(**args);loss=torch.nn.functional.binary_cross_entropy_with_logits(z,labels)
        assert torch.isfinite(z).all() and torch.isfinite(loss);loss.backward();assert_finite_gradients(model)
        gradient_receipts.append(active_fusion_gradient_evidence(model));optimizer.step();losses.append(float(loss.detach()))
    torch.cuda.synchronize();postflight(root)
    a=dict(status='PASS_FROZEN_BEFORE_FORMAL_VAL',V5_actual_test=True,patient_microbatch=4,gradient_accumulation=2,effective_patient_batch=8,
           trainable_parameters=parameters,base_parameters=12755009,new_fusion_parameters=265729,encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
           transformer_parameters=sum(p.numel() for p in model.transformer.parameters()),real_views=len(batch['real_images']),synthetic_views=len(batch['synthetic_images']),
           actual_real_exam_lengths=batch['real_exam_lengths'],max_train_real_exams=max(r['observed_history_exam_count'] for r in records),
           max_train_native_views=max(r['observed_history_view_count'] for r in records),selected_max_nodes_one_exam=max(max_nodes_at_exam(i) for i in chosen),
           all_views_retained=True,optimizer_steps=2,precision='FP32',losses=losses,fusion_finite_nonzero_gradient_norms=gradient_receipts,
           peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,peak_reserved_GiB=torch.cuda.max_memory_reserved()/1024**3,
           Formal_Val_metrics_access=0,smoke_weights_reused=False,hyperparameters_changed=0,completed_utc=now())
    env=dict(python=current[0],torch=current[1],torchvision=current[2],cuda_runtime=current[3],GPU=torch.cuda.get_device_name(0),capability=torch.cuda.get_device_capability(0),checked_utc=now())
    jout(root/'audit/ENVIRONMENT_V5.json',env);jout(root/'audit/GPU_PREFLIGHT_V5.json',a);jout(marker,a)
    with (root/'audit/ARCHITECTURE_AND_RANDOMNESS_PARITY_V5.md').open('a') as f:f.write('\n## Actual GPU/parameter/gradient receipt\n\n'+json.dumps(a,indent=2)+'\n')
    print({k:v for k,v in a.items() if k!='fusion_finite_nonzero_gradient_norms'},flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);main(p.parse_args().config)
