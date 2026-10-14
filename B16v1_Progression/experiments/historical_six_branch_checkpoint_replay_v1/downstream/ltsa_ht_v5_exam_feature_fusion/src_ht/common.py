"""V5-only writes, safe aggregates and metadata protection; no file-content digests."""
import json
import os
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import yaml

ROOT = Path('REQUIRED_AUTHORIZED_LOCAL_PATH_0')
TASK = 'LTSA_HT_V5_EXAM_LEVEL_FEATURE_FUSION_REAL_VS_OP'
RESULT = 'LTSA_HT_V5_EXAM_FEATURE_FUSION_REAL_VS_ORIGINAL_PROGRESSION_RESULT.md'
CFG_NAME = 'LTSA_HT_V5_FROZEN_CONTRACT.yaml'
ARMS = ['LTSA_V5_REAL_ONLY', 'LTSA_V5_ORIGINAL_PROGRESSION_FUSION']
SEEDS = [42, 3407, 2026]

def now(): return datetime.now(timezone.utc).isoformat()
def setup(path):
    os.umask(0o077); r = Path(path).resolve(); assert r == ROOT
    r.mkdir(parents=True, exist_ok=True); r.chmod(0o700)
    for f in ['src_ht','upstream','configs','manifests_private','audit','tests','logs','checkpoints','metrics/private','tables','figures/nonmedical','figures/source_data','figures/captions','figures/export_qa','writing','releases','cache']:
        (r/f).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(r/'cache/matplotlib')); return r
def jread(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def jout(p, a):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); t=p.with_name(p.name+'.tmp')
    t.write_text(json.dumps(a, ensure_ascii=False, indent=2, allow_nan=False, default=str)+'\n', encoding='utf-8'); os.replace(t,p)
def cout(p, d):
    p=Path(p); t=p.with_name(p.name+'.tmp'); (d if isinstance(d,pd.DataFrame) else pd.DataFrame(d)).to_csv(t,index=False); os.replace(t,p)
def config(p):
    c=yaml.safe_load(Path(p).read_text()); assert Path(c['project_private']).resolve()==ROOT and c['task']==TASK
    assert c['arms']==ARMS and c['seeds']==SEEDS and c['fusion_scale']==0.1
    from v5_safety import install_boundary
    install_boundary(c); return c
def metadata(p):
    p=Path(p); s=p.stat(); return dict(path=str(p),resolved_path=str(p.resolve()),size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino)
def postflight(root):
    rows=jread(root/'audit/PROTECTED_SOURCE_METADATA_V5.json'); changed=[]
    for row in rows:
        try:
            if metadata(row['path'])!=row: changed.append(row['path'])
        except OSError: changed.append(row['path'])
    a=dict(status='PASS' if not changed else 'FAIL',unique_files_checked=len(rows),changed_count=len(changed),changed_paths_private=changed,
           scope='path/resolved_path/size/mtime_ns/inode only; not content equality',checked_utc=now())
    jout(root/'audit/SOURCE_PROTECTION_POSTFLIGHT_V5.json',a)
    if changed: raise RuntimeError('Protected source metadata changed; stop')
    return a
def state(root,phase,**kw): jout(root/'releases/EXECUTION_STATE_V5.json',dict(task=TASK,phase=phase,updated_utc=now(),**kw))
