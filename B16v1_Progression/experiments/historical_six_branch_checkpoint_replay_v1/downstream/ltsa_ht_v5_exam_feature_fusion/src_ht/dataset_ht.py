"""Read-only frozen pixels, ragged Exam sets and privately frozen next-real indices."""
import json
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import ARMS,CFG_NAME,jread
MEAN=torch.tensor([.485,.456,.406])[None,:,None,None]
STD=torch.tensor([.229,.224,.225])[None,:,None,None]
def next_real_index(real_dates,query):
    assert real_dates==sorted(real_dates) and len(set(real_dates))==len(real_dates)
    assert real_dates[0]<query<real_dates[-1] and query not in real_dates
    i=next(i for i,d in enumerate(real_dates) if d>query)
    assert real_dates[i-1]<=query<real_dates[i];return i
def normalize(a):
    x=torch.from_numpy(a).permute(0,3,1,2).contiguous().to(torch.float32).div_(255)
    return (x-MEAN)/STD
class HTLongitudinalDataset(torch.utils.data.Dataset):
    def __setstate__(self,state):
        self.__dict__.update(state)
        from common import config
        config(self.root/'configs'/CFG_NAME)
    def __init__(self,root,split,arm=ARMS[0]):
        assert split in {'train','val'},'Official Test forbidden including Dataset metadata'
        assert arm in ARMS
        self.root=Path(root);self.split=split;self.arm=arm
        master=pd.read_parquet(self.root/'manifests_private/HT_LTSA_CANONICAL_PATIENT_WINDOWS_V1.parquet')
        self.records=master[master.split.eq(split)].sort_values('patient_index').to_dict('records')
        assert len(self.records)==(860 if split=='train' else 189)
        self.augmentation={}
        if split=='train' and arm==ARMS[1]:
            assert jread(self.root/'releases/V5_PREPARATION_COMPLETE.json')['status']=='PASS'
            for n in jread(self.root/'manifests_private/OP_NODE_TO_REAL_EXAM_V5.json'):
                self.augmentation.setdefault(int(n['patient_index']),[]).append(n)
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        r=self.records[index];assert r['eligibility_status']=='ELIGIBLE'
        exams=json.loads(r['exams_private_json']);counts=[int(e['views']) for e in exams];dates=[e['date'] for e in exams]
        assert all(int(e['date'][:4])<int(r['target_year']) and e['source_type']=='REAL' for e in exams)
        nodes=self.augmentation.get(int(r['patient_index']),[])
        for n in nodes:
            i=next_real_index(dates,n['query_date_private']);assert i==n['real_exam_index']
        # The ledger above existed before any image cache was opened.
        a=np.load(r['native_cache_private'],mmap_mode='r');assert a.dtype==np.uint8 and a.shape==(sum(counts),224,224,3)
        real=normalize(np.array(a,copy=True));del a
        synth=[np.stack([np.load(p) for p in n['synthetic_cache_private']]) for n in nodes]
        synthetic=normalize(np.concatenate(synth)) if synth else torch.empty((0,3,224,224),dtype=torch.float32)
        mapping=[int(n['real_exam_index']) for n in nodes];mask=[i in mapping for i in range(len(exams))]
        origin=date.fromisoformat(dates[0]);months=[int((date.fromisoformat(d)-origin).days/30.4375) for d in dates]
        assert months==json.loads(r['relative_months_private_json'])
        return dict(sample_index=int(r['patient_index']),real_images=real,real_exam_view_counts=counts,real_relative_months=months,
                    synthetic_images=synthetic,synthetic_node_image_counts=[6]*len(nodes),node_to_real_exam_index=mapping,
                    has_synth_mask=mask,label=int(r['target_HT']))
def collate_patients(samples):
    lengths=[len(s['real_exam_view_counts']) for s in samples];months=torch.zeros((len(samples),max(lengths)),dtype=torch.long)
    mapping=[];offset=0
    for i,s in enumerate(samples):
        months[i,:lengths[i]]=torch.tensor(s['real_relative_months'],dtype=torch.long)
        mapping.extend(offset+n for n in s['node_to_real_exam_index']);offset+=lengths[i]
    return dict(sample_index=torch.tensor([s['sample_index'] for s in samples]),real_images=torch.cat([s['real_images'] for s in samples]),
                real_exam_view_counts=[n for s in samples for n in s['real_exam_view_counts']],real_exam_lengths=lengths,real_relative_months=months,
                synthetic_images=torch.cat([s['synthetic_images'] for s in samples]),synthetic_node_image_counts=[n for s in samples for n in s['synthetic_node_image_counts']],
                node_to_real_exam_index=mapping,has_synth_mask=[m for s in samples for m in s['has_synth_mask']],labels=torch.tensor([s['label'] for s in samples],dtype=torch.float32))
def model_inputs(batch,device):
    keys=['real_images','real_exam_view_counts','real_exam_lengths','real_relative_months','synthetic_images','synthetic_node_image_counts','node_to_real_exam_index','has_synth_mask']
    return {k:batch[k].to(device,non_blocking=True) if isinstance(batch[k],torch.Tensor) else batch[k] for k in keys}
