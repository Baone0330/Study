"""Freeze next-real ledger BEFORE opening any pixels; exact private item comparison."""
import json
import shutil
from collections import Counter
from datetime import date,datetime
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from common import ROOT,TASK,ARMS,SEEDS,CFG_NAME,setup,config,jread,jout,cout,metadata,now,postflight
from v5_safety import V3,V4
from dataset_ht import next_real_index
def parse_date(text):
    for fmt in ['%Y-%m-%d','%Y.%m.%d','%Y_%m_%d','%Y%m%d']:
        try:return datetime.strptime(text,fmt).date()
        except ValueError:pass
    return None
def main():
    root=setup(ROOT);ready=root/'releases/V5_PREPARATION_COMPLETE.json'
    if ready.exists():assert jread(ready)['status']=='PASS';postflight(root);print('Frozen V5 preparation retained');return
    assert not list((root/'checkpoints').rglob('*.pt')) and not list((root/'metrics/private').iterdir())
    old=yaml.safe_load((V3/'configs/LTSA_HT_TWO_ARM_CONTRACT_V3.yaml').read_text())
    assert old['patient_microbatch']==4 and old['gradient_accumulation']==2 and old['bootstrap_seed']==20261013
    assert old['precision']=='FP32' and old['augmentation_eligible_nodes']==686 and old['augmentation_eligible_views']==4116
    c=dict(old);c.update(task=TASK,project_private=str(root),arms=ARMS,seeds=SEEDS,frozen_utc=now(),fusion_scale=0.1,
        fusion_norms='separate trainable real and synthetic LayerNorm512; both arms identical',fusion_attachment='first REAL date strictly later than frozen query',
        fusion_init='default PyTorch, after unchanged baseline constructor, preserving Python/NumPy/TorchCPU/CUDA RNG',
        encoder_order='all REAL views in batch first, then all synthetic views; shared weights, chunk16; standard train BN',
        augmentation_policy='same686 frozen nodes, no extra tokens; node-equal mean + masked residual fusion',last_prediction_token='latest REAL historical Exam; REAL timeline identical in both V5 arms')
    cfg=root/'configs'/CFG_NAME;assert not cfg.exists();cfg.write_text(yaml.safe_dump(c,sort_keys=False,allow_unicode=True));config(cfg)
    changed=[k for k in old if old[k]!=c[k]]
    jout(root/'audit/CONFIG_DIFFERENCES_V3_V5.json',dict(changed_fields=changed,added_fields=[k for k in c if k not in old],performance_tuning=False))
    names=['HT_LTSA_CANONICAL_PATIENT_WINDOWS_V1.parquet','REAL_NATIVE_VIEWS_V3.parquet','REAL_NATIVE_EXAMS_V3.parquet','OP_FULL_SOURCE_LINEAGE_AND_ELIGIBILITY_V3.json','OP_BOUNDED_SYNTHETIC_TOKEN_MAP_V3.json']
    for name in names:
        p=V3/'manifests_private'/name;dest=root/'manifests_private'/name;assert not dest.exists();shutil.copy2(p,dest);dest.chmod(0o600)
        if p.suffix=='.parquet':assert pd.read_parquet(p).equals(pd.read_parquet(dest)) and pd.read_parquet(p).equals(pd.read_parquet(V4/'manifests_private'/name))
        else:assert jread(p)==jread(dest)==jread(V4/'manifests_private'/name)
    master=pd.read_parquet(root/'manifests_private/HT_LTSA_CANONICAL_PATIENT_WINDOWS_V1.parquet')
    assert len(master)==1049 and master.patient_key_private.is_unique and master.patient_index.is_unique
    train=master[master.split.eq('train')].set_index('patient_index');val=master[master.split.eq('val')]
    assert set(train.patient_key_private).isdisjoint(val.patient_key_private)
    mapping=jread(root/'manifests_private/OP_BOUNDED_SYNTHETIC_TOKEN_MAP_V3.json');source=jread(root/'manifests_private/OP_FULL_SOURCE_LINEAGE_AND_ELIGIBILITY_V3.json')
    eligible={n['node_key_private']:n for n in source if n['eligibility']=='ELIGIBLE_BOUNDED_SYNTHETIC_QUERY_TOKEN'}
    categories=Counter(n['eligibility'] for n in source);assert len(source)==1545 and sorted(categories.values())==[101,686,758]
    assert len(mapping)==len(eligible)==686 and len({n['node_key_private'] for n in mapping})==686
    attached=[];gaps=[];ordinal=Counter();exam_nodes=Counter()
    for n in mapping:
        r=train.loc[int(n['patient_index'])];exams=json.loads(r.exams_private_json);dates=[e['date'] for e in exams]
        source_node=eligible[n['node_key_private']]
        assert source_node['source_original_CPfm_ON'] is True
        assert source_node['patient_index']==n['patient_index'] and source_node['query_date_private']==n['query_date_private']
        assert source_node['seeds_private']==n['source_seeds_private'] and source_node['source_paths_private']==n['original_reference_paths_private']
        assert len(n['synthetic_cache_private'])==6 and len(n['original_reference_paths_private'])==6
        i=next_real_index(dates,n['query_date_private']);assert int(dates[i][:4])<int(r.target_year)
        gap=(date.fromisoformat(dates[i])-date.fromisoformat(n['query_date_private'])).days
        attached.append(dict(n,real_exam_index=i,real_anchor_date_private=dates[i],query_to_anchor_days_private=gap))
        exam_nodes[(int(n['patient_index']),i)]+=1;gaps.append(gap);ordinal[i+1]+=1
    assert len({n['patient_index'] for n in attached})==597 and len(train)-597==263
    # Persist complete date-derived mapping prior to any np.load/image opens.
    jout(root/'manifests_private/OP_NODE_TO_REAL_EXAM_V5.json',attached)
    jout(root/'audit/NEXT_REAL_LEDGER_FROZEN_BEFORE_PIXEL_ACCESS_V5.json',dict(status='PASS',nodes=686,views=4116,duplicated_bindings=0,missing_nodes=0,
         no_future_to_earlier_real_binding=True,no_target_year_anchor=True,no_new_tokens=True,frozen_utc=now()))
    # Reconstruct raw/history identity only from pre-target directory listings; never open clinical files.
    raw=Path(c['raw_data_private']);support=[];mismatch=Counter();real_total=0
    for split,n,negative,positive,exam_total,view_total in [('train',860,617,243,4585,18413),('val',189,131,58,991,3861)]:
        seq=pd.read_csv(Path(c['source_old_ablation_private'])/'manifests'/('ORIGINAL_TRAIN_SEQUENCE_V1.csv' if split=='train' else 'ORIGINAL_VAL_SEQUENCE_V1.csv'),dtype={'patient_id':str})
        groups={str(pid):g for pid,g in seq.groupby('patient_id',sort=True)};rows=master[master.split.eq(split)]
        assert len(rows)==n and Counter(rows.target_HT.astype(int))=={0:negative,1:positive} and set(rows.patient_key_private.astype(str))==set(groups)
        exams_n=views_n=0
        for r in rows.itertuples(index=False):
            exams=json.loads(r.exams_private_json);g=groups[str(r.patient_key_private)]
            assert g.target_label.nunique()==g.target_year.nunique()==1
            mismatch['label_or_target_year']+=int(int(g.target_label.iloc[0])!=int(r.target_HT) or int(g.target_year.iloc[0])!=int(r.target_year))
            folders=[raw/str(lab)/str(r.patient_key_private) for lab in [0,1] if (raw/str(lab)/str(r.patient_key_private)).is_dir()]
            assert len(folders)==1 and folders[0].parent.name==str(int(r.target_HT))
            native=[]
            for p in sorted(folders[0].iterdir()):
                if not p.is_dir():continue
                d=parse_date(p.name);assert d is not None
                if d.year>=int(r.target_year):continue  # Target-year contents not even enumerated.
                paths=sorted(str(z) for z in p.rglob('*') if z.is_file() and z.suffix.lower() in ['.png','.jpg','.jpeg','.bmp','.tif','.tiff'])
                if paths:native.append(dict(date=d.isoformat(),paths=paths,source_type='REAL',views=len(paths)))
            native.sort(key=lambda e:e['date']);mismatch['real_exam_date_view_order_or_path']+=int(native!=exams)
            assert r.eligibility_status=='ELIGIBLE' and all(int(e['date'][:4])<int(r.target_year) for e in exams)
            counts=[int(e['views']) for e in exams];exams_n+=len(exams);views_n+=sum(counts)
            pix=np.load(r.native_cache_private,mmap_mode='r');assert pix.dtype==np.uint8 and pix.shape==(sum(counts),224,224,3);del pix
        assert (exams_n,views_n)==(exam_total,view_total);real_total+=exams_n
        support.append(dict(split=split,patients=n,HT0=negative,HT1=positive,REAL_exams=exams_n,REAL_views=views_n,synthetic_nodes=686 if split=='train' else 0,
            synthetic_views=4116 if split=='train' else 0,patients_with_synthetic=597 if split=='train' else 0,patients_without_synthetic=263 if split=='train' else n,
            attached_REAL_exams=len(exam_nodes) if split=='train' else 0,real_exams_zero_nodes=exams_n-len(exam_nodes) if split=='train' else exams_n,
            real_exams_one_node=sum(v==1 for v in exam_nodes.values()) if split=='train' else 0,real_exams_two_plus_nodes=sum(v>=2 for v in exam_nodes.values()) if split=='train' else 0,
            duplicate_bindings=0,missing_nodes=0,Val_synthetic_access=0))
    assert not any(mismatch.values())
    for node in attached:
        for p in node['synthetic_cache_private']+node['original_reference_paths_private']:assert Path(p).is_file()
        for p in node['synthetic_cache_private']:
            a=np.load(p,mmap_mode='r');assert a.dtype==np.uint8 and a.shape==(224,224,3);del a
    aggregate=[dict(category='anchor_ordinal',value=i,count=n) for i,n in sorted(ordinal.items())]
    aggregate.extend(dict(category='nodes_per_attached_exam',value=i,count=n) for i,n in sorted(Counter(exam_nodes.values()).items()))
    for label,v in [('min',min(gaps)),('median',float(np.median(gaps))),('p25',float(np.quantile(gaps,.25))),('p75',float(np.quantile(gaps,.75))),('max',max(gaps)),('mean',float(np.mean(gaps)))]:aggregate.append(dict(category='query_to_anchor_days',value=label,count=v))
    cout(root/'tables/FEATURE_FUSION_INPUT_COUNTS_V5.csv',support);cout(root/'tables/V5_NODE_TO_REAL_EXAM_AGGREGATE.csv',aggregate)
    cout(root/'tables/OP_CANDIDATE_MAPPING_COUNTS_V5.csv',[dict(category=k,nodes=v,images=6*v) for k,v in categories.items()])
    parity=dict(status='PASS',Train=860,Val=189,private_item_level_mismatch_counts=dict(mismatch),V3_V4_frozen_manifest_equal=True,
         mapping_reselected=False,synthetic_nodes=686,synthetic_views=4116,raw_image_content_reads=0,Official_Test_content_access=0,real_token_count_A_B_identical=True,checked_utc=now())
    jout(root/'audit/INPUT_IDENTITY_PARITY_V5.json',parity)
    assert (root/'src_ht/model_base.py').read_bytes()==(V3/'src_ht/model_ht.py').read_bytes()==(V4/'src_ht/model_ht.py').read_bytes()
    assert (root/'src_ht/runtime_ht.py').read_bytes()==(V3/'src_ht/runtime_ht.py').read_bytes()==(V4/'src_ht/runtime_ht.py').read_bytes()
    paths={r['path'] for r in jread(V3/'audit/PROTECTED_SOURCE_METADATA_V3.json')}
    for p in [V3,V4,Path(c['local_ltsa_source_private'])]:paths.update(str(q) for q in p.rglob('*') if q.is_file())
    paths.add(c['resnet18_weights_private']);jout(root/'audit/PROTECTED_SOURCE_METADATA_V5.json',[metadata(p) for p in sorted(paths)])
    for arm in ARMS:
        for seed in SEEDS:(root/'checkpoints'/arm/f'seed_{seed}').mkdir(parents=True,exist_ok=True)
    lineage=dict(status='PASS_MAPPING_FROZEN_BEFORE_FORMAL_VAL',eligible_synthetic_nodes=686,eligible_synthetic_images=4116,mapping_reselected=False,verified_in_V5_utc=now())
    jout(root/'releases/ORIGINAL_PROGRESSION_LINEAGE_STATUS_V3.json',lineage)  # New verification; legacy internal name, not old PASS copy.
    (root/'audit/UPSTREAM_LINEAGE_AND_SOURCE_ANCHORS_V5.md').write_text('# Source anchors\n\nAuthoritative local LTSA source and successful V3/V4 are read-only. LTSA source has no Git revision available: UNKNOWN, not invented. model_base.LTSAHT.encode_exams and temporal_forward are byte-identical to the actual V3/V4 model source. runtime_ht.configure, loader RNG, scheduler, selection, checkpoint/RNG and metrics are retained. V5 dataset.next_real_index maps dates before pixels; model_ht.ExamResidualFusion adds two separate LayerNorm512, Linear1024→1 sigmoid gate and Linear512→512 residual projection. Trainer allows expected None gradients for unused fusion; no baseline numerical-loop change. Pixel caches are readonly uint8RGB224, not old trained features. Original Progression images come from B14 CPFM ON.\n\n'+json.dumps(parity,ensure_ascii=False,indent=2)+'\n')
    (root/'audit/V5_CONFIG_AND_TRAINING_CONTRACT.md').write_text('# Frozen training contract\n\n'+yaml.safe_dump({k:v for k,v in c.items() if not k.endswith('_private')},sort_keys=False)+'\nNo performance gate or hyperparameter search. Epoch-boundary resume belongs only to V5. Internal V3 filenames in new run directories are newly computed evidence, never old completion marks.\n')
    (root/'audit/ARCHITECTURE_AND_RANDOMNESS_PARITY_V5.md').write_text('# V5 shared architecture\n\nBoth arms independently construct the same model. Baseline constructor, temporal forward and runtime are unchanged; newly initialized fusion preserves Python/NumPy/Torch CPU/all initialized CUDA RNG. Separate real and synthetic LayerNorm parameters, fixed 0.1 residual. REAL-only identity bypass consumes no fusion RNG. All real images encoded first, all synthetic images second with the same encoder and chunk16; train BN is standard and synthetic images can alter its running statistics. Actual parameter counts, initial-state/RNG and gradients will be recorded by anonymous tests and GPU smoke; not assumed passed here.\n')
    postflight(root);jout(ready,dict(status='PASS',input_identity=parity,prepared_utc=now(),new_empty_runs=6,old_checkpoints_or_predictions_copied=0))
    print(dict(preparation='PASS',support=support,attached_real_exams=len(exam_nodes),mismatch_counts=dict(mismatch)),flush=True)
if __name__=='__main__':main()
