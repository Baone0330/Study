"""Private, local-source-only LTSA-HT execution helpers; no Test content or file checksums."""
from __future__ import annotations
import json
import os
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
import yaml

ROOT_NAME='ltsa_ht_real_vs_original_progression_v3_local_source'
TASK='LTSA_HT_LOCAL_SOURCE_REAL_VS_ORIGINAL_PROGRESSION_TWO_ARM_V3'
RESULT='LTSA_HT_TWO_ARM_REAL_VS_ORIGINAL_PROGRESSION_LOCAL_SOURCE_RESULT_V3.md'
ARMS=['LTSA_REAL_ONLY','LTSA_ORIGINAL_PROGRESSION_HIST']
SEEDS=[42,3407,2026]


def now():return datetime.now(timezone.utc).isoformat()


def setup(path):
    os.umask(0o077);r=Path(path).resolve();assert r.name==ROOT_NAME
    r.mkdir(parents=True,exist_ok=True);os.chmod(r,0o700)
    for f in ['upstream','src_ht','configs','manifests_private','logs','checkpoints','metrics/private','metrics/public','tables','figures/nonmedical','figures/source_data','figures/captions','figures/export_qa','writing','audit','tests','releases','cache/real224','cache/synthetic224']:(r/f).mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(r/'cache/matplotlib'));return r


def jread(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def jout(p,a):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n',encoding='utf-8');os.replace(t,p)


def cout(p,d):
    p=Path(p);t=p.with_name(p.name+'.tmp');(d if isinstance(d,pd.DataFrame) else pd.DataFrame(d)).to_csv(t,index=False);os.replace(t,p)


def pout(p,d):
    p=Path(p);t=p.with_name(p.name+'.tmp');d.to_parquet(t,index=False);os.replace(t,p)


def config(p):return yaml.safe_load(Path(p).read_text(encoding='utf-8'))


def metadata(p):
    p=Path(p);s=p.stat();return dict(path=str(p),resolved_path=str(p.resolve()),size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino)


def postflight(root):
    rows=jread(root/'audit/PROTECTED_SOURCE_METADATA_V3.json');changed=[]
    for r in rows:
        try:
            if metadata(r['path'])!=r:changed.append(r['path'])
        except OSError:changed.append(r['path'])
    a=dict(status='PASS' if not changed else 'FAIL',unique_files_checked=len(rows),changed_paths_private=changed,
           check='path,resolved_path,size,mtime_ns,inode; no content checksum',checked_utc=now())
    jout(root/'audit/SOURCE_PROTECTION_POSTFLIGHT_V3.json',a)
    if changed:raise RuntimeError('Protected metadata changed; private audit lists paths; no source overwrite')
    return a


def state(root,phase,**kw):jout(root/'releases/EXECUTION_STATE_V3.json',dict(task=TASK,phase=phase,updated_utc=now(),**kw))
