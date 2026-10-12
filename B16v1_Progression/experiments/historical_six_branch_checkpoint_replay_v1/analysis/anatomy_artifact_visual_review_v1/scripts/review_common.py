"""Private-first, CPU-only review IO; no source writes, medical transfer, hashes or training."""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

TASK='B16V1_HISTORICAL_OP_ANATOMY_ARTIFACT_VISUAL_REVIEW_V1'
ROOT_NAME='b16v1_historical_original_progression_anatomy_artifact_review_v1'
RESULT_NAME='B16V1_HISTORICAL_OP_ANATOMY_ARTIFACT_VISUAL_REVIEW_RESULT_V1.md'


def now(): return datetime.now(timezone.utc).isoformat()


def setup(root):
    os.umask(0o077)
    r=Path(root).resolve();assert r.name==ROOT_NAME
    r.mkdir(parents=True,exist_ok=True);os.chmod(r,0o700)
    for name in ['protocol','source_audit','manifests','metrics','reviews_private/raw_completed_forms',
                 'reviews_private/completed_local_review_only','reviews_private/review_interface_local_only',
                 'figures/private_medical_panels','figures/private_expert_review_cards','figures/public_nonmedical',
                 'figures/source_data_public','figures/captions_cn_en','figures/export_qa',
                 'tables','writing','scripts','tests','logs','release','cache/localization']:
        (r/name).mkdir(parents=True,exist_ok=True)
    for name in ['manifests','reviews_private','figures/private_medical_panels','figures/private_expert_review_cards']:
        os.chmod(r/name,0o700)
    os.environ.setdefault('MPLCONFIGDIR',str(r/'cache/matplotlib'))
    return r


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def json_out(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n',encoding='utf-8');os.replace(tmp,p)


def parquet_out(path,frame):
    p=Path(path);tmp=p.with_name(p.name+'.tmp');frame.to_parquet(tmp,index=False);os.replace(tmp,p)


def csv_out(path,frame):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    (frame if isinstance(frame,pd.DataFrame) else pd.DataFrame(frame)).to_csv(tmp,index=False);os.replace(tmp,p)


def metadata(path):
    p=Path(path);s=p.stat();return dict(path=str(p),resolved_path=str(p.resolve()),size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino)


def postflight(root):
    baseline=read_json(root/'source_audit/PROTECTED_SOURCE_METADATA_THIS_TASK_V1.json');changed=[]
    for r in baseline:
        try:
            if metadata(r['path'])!=r:changed.append(r['path'])
        except OSError:changed.append(r['path'])
    out=dict(status='PASS' if not changed else 'FAIL',unique_files_actually_checked=len(baseline),changed_paths_private=changed,
             comparison='path,resolved_path,size,mtime_ns,inode; not a file-content integrity certification',checked_utc=now())
    json_out(root/'source_audit/ORIGINAL_SOURCE_POSTFLIGHT_V1.json',out)
    if changed:raise RuntimeError('Read-only source metadata changed; stop and inspect private postflight, never overwrite source')
    return out


def state(root,phase,**kw):json_out(root/'release/EXECUTION_STATE_V1.json',dict(task=TASK,phase=phase,updated_utc=now(),**kw))


def unique_pairs(q,pairs,expected_pairs=9270,expected_nodes=1545,expected_patients=860,images_per_node=6):
    """Match authoritative source pair-map, not filenames, then exact keys/seeds."""
    assert len(pairs)==expected_pairs and not pairs.duplicated(['historical_node_key','image_index']).any()
    on=q[q.branch=='OP_CPFM_ON'];off=q[q.branch=='OP_CPFM_OFF']
    key=['patient_key_private','historical_node_key_private','image_index']
    p=on.merge(off,on=key,suffixes=('_on','_off'),how='outer',validate='1:1',indicator=True)
    assert len(p)==expected_pairs and p['_merge'].eq('both').all()
    assert (p.seed_on==p.seed_off).all() and (p.source_date_private_on==p.source_date_private_off).all()
    assert (p.anonymous_patient_key_on==p.anonymous_patient_key_off).all() and (p.anonymous_node_key_on==p.anonymous_node_key_off).all()
    old=pairs.rename(columns={'historical_node_key':'historical_node_key_private'})
    p=p.drop(columns='_merge').merge(old,on=key,how='left',validate='1:1')
    assert p.nominal_slot_id.notna().all() and (p.master_seed==p.seed_on).all()
    assert (p.png_abs_path_on==p.reference_ON_image_path).all() and (p.png_abs_path_off==p.new_OFF_image_path).all()
    assert p.patient_key_private.nunique()==expected_patients and p.historical_node_key_private.nunique()==expected_nodes
    assert p.groupby('historical_node_key_private').image_index.apply(lambda x:set(x)==set(range(images_per_node))).all()
    return p.sort_values(['patient_key_private','historical_node_key_private','nominal_slot_id','image_index','master_seed'],kind='stable').reset_index(drop=True)
