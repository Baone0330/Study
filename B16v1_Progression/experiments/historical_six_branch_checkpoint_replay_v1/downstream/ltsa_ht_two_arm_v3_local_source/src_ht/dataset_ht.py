"""Private cached native pixels; only numeric sample identity and observed-time tensors are returned."""
import json
from datetime import date
import numpy as np
import pandas as pd
import torch
from common import ARMS,jread

MEAN=torch.tensor([.485,.456,.406])[None,:,None,None]
STD=torch.tensor([.229,.224,.225])[None,:,None,None]


class HTLongitudinalDataset(torch.utils.data.Dataset):
    def __init__(self,root,split,arm='LTSA_REAL_ONLY'):
        assert split in {'train','val'},'Official Test is forbidden, including metadata input through Dataset'
        assert arm in ARMS
        self.root=root;self.split=split;self.arm=arm
        master=pd.read_parquet(root/'manifests_private/HT_LTSA_CANONICAL_PATIENT_WINDOWS_V1.parquet')
        self.records=master[master.split.eq(split)].sort_values('patient_index').to_dict('records')
        assert len(self.records)==(860 if split=='train' else 189)
        self.augmentation={}
        if split=='train' and arm=='LTSA_ORIGINAL_PROGRESSION_HIST':
            status=jread(root/'releases/ORIGINAL_PROGRESSION_LINEAGE_STATUS_V3.json')
            assert status['status']=='PASS_MAPPING_FROZEN_BEFORE_FORMAL_VAL','No inferred/unknown-future synthetic mapping is accepted'
            for row in jread(root/'manifests_private/OP_BOUNDED_SYNTHETIC_TOKEN_MAP_V3.json'):
                self.augmentation.setdefault(int(row['patient_index']),[]).append(row)

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        r=self.records[index];assert r['eligibility_status']=='ELIGIBLE'
        exams=json.loads(r['exams_private_json']);counts=[e['views'] for e in exams]
        assert all(int(e['date'][:4])<int(r['target_year']) and e['source_type']=='REAL' for e in exams)
        real=np.load(r['native_cache_private'],mmap_mode='r');assert len(real)==sum(counts)
        arrays=[];timeline=[];offset=0
        for e,n in zip(exams,counts):
            timeline.append(dict(date=e['date'],array=np.array(real[offset:offset+n],copy=True),source_type='REAL'));offset+=n
        for t in self.augmentation.get(r['patient_index'],[]):
            query=date.fromisoformat(t['query_date_private']);first=date.fromisoformat(exams[0]['date']);last=date.fromisoformat(exams[-1]['date'])
            assert first<query<last and query.year<int(r['target_year']) and t['source_type']=='SYNTH_BOUNDED_ANNUAL_QUERY_NOT_REAL_EXAM'
            timeline.append(dict(date=query.isoformat(),array=np.stack([np.load(p) for p in t['synthetic_cache_private']]),source_type=t['source_type']))
        timeline.sort(key=lambda e:e['date']);assert len({e['date'] for e in timeline})==len(timeline) and timeline[-1]['source_type']=='REAL'
        counts=[len(e['array']) for e in timeline];a=np.concatenate([e['array'] for e in timeline],axis=0)
        x=torch.from_numpy(a).permute(0,3,1,2).contiguous().to(torch.float32).div_(255)
        origin=date.fromisoformat(exams[0]['date']);months=[int((date.fromisoformat(e['date'])-origin).days/30.4375) for e in timeline]
        x=(x-MEAN)/STD
        assert len(months)==len(counts) and len(x)==sum(counts)
        return dict(sample_index=int(r['patient_index']),images=x,exam_view_counts=counts,relative_months=months,label=int(r['target_HT']))


def collate_patients(samples):
    lengths=[len(s['exam_view_counts']) for s in samples];months=torch.zeros((len(samples),max(lengths)),dtype=torch.long)
    for i,s in enumerate(samples):months[i,:lengths[i]]=torch.tensor(s['relative_months'],dtype=torch.long)
    return dict(sample_index=torch.tensor([s['sample_index'] for s in samples]),images=torch.cat([s['images'] for s in samples]),
                exam_view_counts=[n for s in samples for n in s['exam_view_counts']],lengths=lengths,relative_months=months,
                labels=torch.tensor([s['label'] for s in samples],dtype=torch.float32))


def model_inputs(batch,device):
    return dict(images=batch['images'].to(device,non_blocking=True),exam_view_counts=batch['exam_view_counts'],lengths=batch['lengths'],relative_months=batch['relative_months'].to(device))
