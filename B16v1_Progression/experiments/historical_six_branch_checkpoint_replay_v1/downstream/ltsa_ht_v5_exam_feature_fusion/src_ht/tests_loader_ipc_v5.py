"""Actual two-worker Train-only IPC transport smoke; no model or Val evaluation."""
import argparse
import json
from multiprocessing.connection import Listener
import torch
from common import config,setup,jout,now,ARMS
from dataset_ht import HTLongitudinalDataset
from train_ltsa_ht import loader
from runtime_ht import configure
from ipc_runtime import ensure_short_ipc
def main(path):
    c=config(path);root=setup(c['project_private']);configure(42);cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state_all()
    ensure_short_ipc(root)
    with Listener(authkey=b'V5_ANONYMOUS_LOCAL_IPC_TEST') as listener:assert len(listener.address)<100
    assert torch.equal(cpu,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(gpu,torch.cuda.get_rng_state_all()))
    d=HTLongitudinalDataset(root,'train',ARMS[1]);batches=[]
    # Exhaust a two-batch Subset so spawn workers/pin-memory transport shut down normally.
    for b in loader(d,c,order=list(range(8)),epoch=1,seed=42):
        assert torch.isfinite(b['real_images']).all() and len(b['labels'])==4
        batches.append(dict(patients=len(b['labels']),real_views=len(b['real_images']),synthetic_views=len(b['synthetic_images'])))
    assert len(batches)==2
    a=dict(status='PASS',actual_spawn_workers=2,Train_only=True,Formal_Val_access=0,model_updates=0,batches=batches,IPC_setup_Torch_RNG_unchanged=True,
           temporary_files_scope='relative cache/ipc within V5 root',completed_utc=now())
    jout(root/'tests/ACTUAL_LOADER_IPC_TEST_V5.json',a);print(a,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);main(p.parse_args().config)
