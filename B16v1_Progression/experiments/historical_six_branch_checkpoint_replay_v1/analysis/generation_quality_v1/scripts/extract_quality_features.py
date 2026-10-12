"""Frozen standard Inception feature extraction with identity-checked resumable chunks."""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from b16_quality_common import setup, read_json, atomic_json, now, stat_record, state
from check_decode_and_image_stats import native_rgb


def main(output_root,device='cuda',batch_size=16,smoke_only=False):
    root=setup(output_root)
    import torch
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    c=read_json(root/'image_features/encoder_contract.json')
    assert c['inception_weight_metadata']==stat_record(c['feature_extractor_weights_path'])
    model=FeatureExtractorInceptionV3('inception-v3-compat',['2048'],
                                     feature_extractor_weights_path=c['feature_extractor_weights_path'],feature_extractor_internal_dtype='float32').eval().to(device)
    assert all(not p.requires_grad for p in model.parameters())
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        toy=torch.arange(3*64*64,dtype=torch.int64).remainder(256).to(torch.uint8).reshape(1,3,64,64).repeat(2,1,1,1).to(device)
        f=model(toy)[0];assert f.shape==(2,2048) and torch.isfinite(f).all()
        repeated_input_max_feature_diff=float(torch.max(torch.abs(f[0]-f[1])).item())
        assert torch.allclose(f[0],f[1],rtol=1e-5,atol=1e-5), repeated_input_max_feature_diff
    resource=dict(status='FAKE_INPUT_GPU_SMOKE_PASS',device=device,batch_size=batch_size,dtype='float32',tf32=False,
                  torch_version=torch.__version__,torch_cuda_runtime=torch.version.cuda,
                  gpu_name=torch.cuda.get_device_name() if device=='cuda' else None,
                  allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None,
                  repeated_input_max_feature_diff=repeated_input_max_feature_diff,
                  repeated_input_test='allclose rtol=1e-5 atol=1e-5, not pixel parity',
                  new_training_steps=0,new_SD_forward=0,updated_utc=now())
    atomic_json(root/'image_features/FEATURE_RUNTIME_CONTRACT_V1.json',resource)
    if smoke_only:print(resource,flush=True);return
    q=pd.read_parquet(root/'manifests/COMBINED_QUALITY_EVAL_FROZEN_QUEUE_V1.parquet')
    qa=pd.read_parquet(root/'metrics/IMAGE_DECODE_AND_TECHNICAL_QA_V1.parquet')
    assert q.image_id.tolist()==qa.image_id.tolist()
    for branch, idx in q.groupby('branch',sort=False).groups.items():
        dst=root/'image_features'/branch;dst.mkdir(exist_ok=True)
        positions=np.asarray(list(idx))
        # Output remains frozen queue order; native same-shape groups only within each cache block.
        for start in range(0,len(positions),256):
            ids=positions[start:start+256];block=q.iloc[ids];out=dst/f'{start:06d}.npz'
            if out.exists():
                with np.load(out,allow_pickle=False) as old:
                    assert old['image_ids'].tolist()==block.image_id.tolist() and old['features'].shape==(len(ids),2048)
                    assert np.isfinite(old['features']).all()
                old_ledger=read_json(dst/f'{start:06d}.ledger.json')
                assert old_ledger['encoder_contract']==c and old_ledger['torch_version']==torch.__version__
                assert old_ledger['batch_size']==batch_size and old_ledger['tf32'] is False
                continue
            values=np.empty((len(ids),2048),dtype=np.float32);shapes=qa.iloc[ids]
            tic=time.monotonic()
            for _,rel in shapes.reset_index(drop=True).groupby(['native_width','native_height'],sort=False).groups.items():
                rel=np.asarray(list(rel))
                for j in range(0,len(rel),batch_size):
                    k=rel[j:j+batch_size]
                    arr=np.stack([native_rgb(block.iloc[n].png_abs_path)[0].transpose(2,0,1) for n in k])
                    x=torch.from_numpy(arr).to(device)
                    with torch.inference_mode(): z=model(x)[0].cpu().numpy()
                    assert z.shape==(len(k),2048) and np.isfinite(z).all()
                    values[k]=z
            temp=out.with_suffix('.tmp')
            with temp.open('wb') as stream:np.savez(stream,image_ids=np.asarray(block.image_id.tolist()),features=values)
            os.replace(temp,out)
            atomic_json(dst/f'{start:06d}.ledger.json',dict(status='COMPLETE',images=len(ids),seconds=time.monotonic()-tic,
                        source_queue='COMBINED_QUALITY_EVAL_FROZEN_QUEUE_V1',encoder_version=c['torch_fidelity_version'],
                        feature_preprocess=c['encoder'],encoder_contract=c,torch_version=torch.__version__,tf32=False,
                        batch_size=batch_size,completed_utc=now()))
            print(f'FEATURES {branch} {min(start+256,len(positions))}/{len(positions)}',flush=True)
    if device=='cuda':resource['allocated_peak_mib']=torch.cuda.max_memory_allocated()/2**20
    resource.update(status='ALL_FROZEN_FEATURES_COMPLETE',images=len(q),updated_utc=now());atomic_json(root/'image_features/FEATURE_RUNTIME_CONTRACT_V1.json',resource)
    state(root,'INCEPTION_FEATURES_COMPLETED',images=len(q));print(resource,flush=True)


def load_features(root,branch):
    chunks=sorted((root/'image_features'/branch).glob('*.npz'));assert chunks
    ids=[];features=[]
    for path in chunks:
        with np.load(path,allow_pickle=False) as d:ids.extend(d['image_ids'].tolist());features.append(d['features'])
    q=pd.read_parquet(root/'manifests/COMBINED_QUALITY_EVAL_FROZEN_QUEUE_V1.parquet')
    selected=q[q.branch==branch].reset_index(drop=True)
    assert ids==selected.image_id.tolist()
    return selected,np.concatenate(features)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',required=True);p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=16);p.add_argument('--smoke-only',action='store_true')
    a=p.parse_args();main(a.output_root,a.device,a.batch_size,a.smoke_only)
