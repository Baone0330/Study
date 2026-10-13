"""Safe aggregate/code archive only; no original LTSA network acquisition, no force push."""
import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd
import yaml
from common import config,setup,jread,jout,now

REMOTE='git@github-baone:Baone0330/Study.git'
PREFIX='B16v1_Progression/experiments/historical_six_branch_checkpoint_replay_v1/downstream/ltsa_ht_two_arm_v3_local_source'
OLD_PREFIX='B16v1_Progression/experiments/historical_six_branch_checkpoint_replay_v1/'
SCRIPTS=['common.py','audit_sources.py','build_ht_manifest.py','model_ht.py','dataset_ht.py','runtime_ht.py','tests_ht.py','gpu_smoke_and_freeze.py',
         'validate_original_progression_lineage.py','train_ltsa_ht.py','reload_and_ensemble.py','build_result.py','summarize_ltsa_ht.py','run_pipeline.py',
         'draw_ltsa_figures.py','write_paper_materials.py','publish_ltsa_package.py']


def git(folder,*args):
    env=os.environ.copy();env['GIT_OPTIONAL_LOCKS']='0'
    r=subprocess.run(['git',*args],cwd=folder,env=env,capture_output=True,text=True,timeout=90)
    if r.returncode:raise RuntimeError('Git '+args[0]+' failed; preserved private checkout, no force/reset')
    return r.stdout.strip()


def literal_pattern(values):
    trie={}
    for value in set(values):
        if len(value)<5:continue
        node=trie
        for char in value:node=node.setdefault(char,{})
        node['']=None
    if not trie:return None
    def emit(node):
        parts=[re.escape(k)+emit(v) for k,v in sorted(node.items()) if k]
        s=parts[0] if len(parts)==1 else '(?:'+'|'.join(parts)+')' if parts else ''
        return '(?:'+s+')?' if '' in node and parts else s
    return re.compile(r'(?<![A-Za-z0-9])'+emit(trie)+r'(?![A-Za-z0-9])')


def main(path):
    c=config(path);root=setup(c['project_private']);final=jread(root/'releases/FINAL_LTSA_TWO_ARM_STATUS_V3.json');assert final['status']=='TWO_ARM_COMPLETED' and final['runs']==6
    previous=root/'releases/GIT_PUBLICATION_STATUS_V3.json'
    if previous.exists() and jread(previous)['status']=='PUSH_VERIFIED':print('Existing exact public release retained');return
    package=root/'github_public_stage';package.mkdir(exist_ok=True);files=[]
    def write(relative,text):
        dest=package/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text,encoding='utf-8');files.append(relative)
    def copy(source,relative):
        dest=package/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest);files.append(relative)
    # Published adapters are auditable templates. Runtime private root literals are deliberately absent.
    for name in SCRIPTS:
        text=(root/'src_ht'/name).read_text(encoding='utf-8')
        roots=sorted(set(re.findall(r'/(?:data|home)/[^\s\x27\x22]+',text)),key=len,reverse=True)
        for i,p in enumerate(roots):text=text.replace(p,'LOCAL_PRIVATE_INPUT_REQUIRED_'+str(i))
        ast.parse(text);write('src_ht/'+name,text)
    template={k:('REQUIRED_LOCAL_PRIVATE_PATH' if k.endswith('_private') else v) for k,v in c.items()}
    template['frozen_utc']='PRIVATE_RUN_TIMESTAMP_OMITTED';template['augmentation_mapping_frozen_utc']='PRIVATE_RUN_TIMESTAMP_OMITTED'
    write('configs/LTSA_HT_TWO_ARM_CONFIG_TEMPLATE_V3.yaml',yaml.safe_dump(template,sort_keys=False,allow_unicode=True))
    for p in sorted((root/'tables').glob('*.csv')):copy(p,'tables/'+p.name)
    for p in sorted((root/'writing').iterdir()):
        if p.is_file() and p.suffix in ['.md','.csv']:copy(p,'writing/'+p.name)
    for folder in ['nonmedical','source_data','captions','export_qa']:
        for p in sorted((root/'figures'/folder).iterdir()):
            if p.is_file() and p.suffix in ['.png','.svg','.pdf','.csv','.json','.md']:copy(p,'figures/'+folder+'/'+p.name)
    copy(root/'tests/ANONYMOUS_CPU_TEST_RESULTS_V3.json','tests/ANONYMOUS_CPU_TEST_RESULTS_V3.json')
    copy(Path(c['snapshot_private'])/'LICENSE','LICENSE_LTSA_SOURCE_ATTRIBUTION.txt')
    write('README.md','''# LTSA-HT local-source two-arm V3

Only two independent downstream conditions: REAL-only and historical Original Progression CPFM-ON augmentation, seeds42/3407/2026. Existing local LTSA implementation is required; no LTSA source is cloned, downloaded, pulled or redistributed here. Source ZIP revision was unknown. The original sinusoidal class is loaded from a privately supplied local snapshot; the ResNet18 offline strict initialization, multiview grouping, dynamic causal masks and binary head are adapted.17adapter scripts and a config template are provided.

Privacy: no patient IDs/dates, reports, pixels, per-patient truth/probabilities, private manifest, caches or model checkpoints. Original dataset and historical experiments remain read-only. This public release is not directly runnable without authorized private inputs: fill the config template and the explicit LOCAL_PRIVATE_INPUT_REQUIRED adapter-template placeholders. Anonymous tests still require the local LTSA source but no clinical files/weights. Optional timm alternatives and unavailable pytest are not installed; stdlib unittest is used.

Observed support:860Train(617negative/243positive),189Val(131/58). Allnative histories:4,585/991exams and18,413/3,861views.686bounded virtual annual-query nodes/4,116synthetic views were selected before any formal Val metrics;758target-year and101unknown-future nodes were excluded.597patients had synthetic tokens,263did not. Tokens remain explicitly synthetic, not fabricated real exams; latest token in both arms is the last real history exam. Same architecture, initializer, optimization, selection and Val inputs. No third arm, new generator training/sampling, source hashes or Official Test.

Actual six best states passed independent strict189-patient probability reload. Seed probability-average ensembles and patient-paired2,000bootstrap contrasts are in aggregate tables. Internal Val also determined stopping/checkpoint/threshold; intervals omit training/selection uncertainty. Earlier CLIMATv2 values are provided-runbook references with a different K6/fullsynthetic protocol, not newly computed or interchangeable results. No unmodified survival-paper reproduction, external test generalization, clinical image authenticity or isolated Progression-causality claim.

Five nonmedical figures are exported atRGB600dpi/editableSVG/embeddedPDF. Automatic export checks do not imply actual agent or human visual review; QA status is stated honestly. No clinical image figures are included. Earlier privacy probing incident was restricted to30dictionary-key patient codes in an internal tool response; no such codes occur in this public archive.

Original attribution: Holste et al., Harnessing the power of longitudinal medical imaging for eye disease prognosis using Transformer-based sequence modeling, npj Digital Medicine7,216(2024), DOI10.1038/s41746-024-01207-4, as supplied in the execution document. Local source license is included for attribution; source code itself remains an external local prerequisite.
''')
    master=pd.read_parquet(root/'manifests_private/HT_LTSA_CANONICAL_PATIENT_WINDOWS_V1.parquet');pid=literal_pattern(master.patient_key_private.astype(str).tolist())
    dates=set()
    for text in master.exams_private_json:
        dates.update(e['date'] for e in json.loads(text))
    date_pattern=literal_pattern(dates);issues=[]
    for rel in files:
        p=package/rel
        if p.suffix in ['.md','.py','.csv','.yaml','.txt','.json','.svg']:
            t=p.read_text(encoding='utf-8')
            if re.search(r'/(?:data|home)/|[A-Z]:[/\\]Users[/\\]',t):issues.append(dict(file=rel,reason='PRIVATE_ABSOLUTE_PATH'))
            if pid and pid.search(t):issues.append(dict(file=rel,reason='ACTUAL_PRIVATE_PATIENT_VALUE'))
            if date_pattern and date_pattern.search(t):issues.append(dict(file=rel,reason='ACTUAL_CLINICAL_DATE_VALUE'))
            if re.search(r'gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{24,}|sk-[A-Za-z0-9_-]{24,}|-----BEGIN (OPENSSH|RSA|EC) PRIVATE KEY-----',t):issues.append(dict(file=rel,reason='CREDENTIAL_PATTERN'))
            if p.suffix=='.svg' and '<image' in t:issues.append(dict(file=rel,reason='SVG_RASTER_EMBEDDED'))
        if p.suffix in ['.png','.pdf'] and not rel.startswith('figures/nonmedical/Fig0'):issues.append(dict(file=rel,reason='NONMEDICAL_BINARY_ALLOWLIST_FAILURE'))
    assert len(files)==len(set(files)) and set(files)=={p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()}
    jout(root/'releases/PUBLIC_PRIVACY_AUDIT_V3.json',dict(status='PASS' if not issues else 'FAIL',files=files,count=len(files),issues=issues,medical_images_public=0))
    assert not issues,'Public privacy scan failed; no Git staging/push'
    checkout=root/'publication_checkout_v3'
    if not (checkout/'.git').exists():
        source=Path(c['quality_private']).parent/'b16v1_historical_original_progression_anatomy_artifact_review_v1/publication_checkout_local_v2'
        assert source.is_dir()
        r=subprocess.run(['git','clone','--no-hardlinks','--branch','main',str(source),str(checkout)],capture_output=True,text=True,timeout=120);assert r.returncode==0
        git(checkout,'remote','set-url','origin',REMOTE)
    assert not git(checkout,'status','--porcelain')
    remote_ref=git(checkout,'ls-remote','origin','refs/heads/main').split()[0]
    if git(checkout,'rev-parse','HEAD')!=remote_ref:
        git(checkout,'fetch','origin','main');git(checkout,'merge','--ff-only','FETCH_HEAD')
    base=git(checkout,'rev-parse','HEAD');assert base==remote_ref and not git(checkout,'ls-tree','-r','--name-only',base,'--',PREFIX)
    git(checkout,'config','user.name','study');git(checkout,'config','user.email','l3026714004@outlook.com')
    target=checkout/PREFIX
    for rel in files:
        p=target/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(package/rel,p);assert p.read_bytes()==(package/rel).read_bytes()
    git(checkout,'add','-f','--',*[PREFIX+'/'+rel for rel in files]);staged=git(checkout,'diff','--cached','--name-only').splitlines()
    assert len(staged)==len(files) and all(p.startswith(PREFIX+'/') for p in staged)
    git(checkout,'commit','-m','Add local-source LTSA HT two-arm downstream validation')
    commit=git(checkout,'rev-parse','HEAD');old=git(checkout,'ls-tree','-r','--name-only',base,'--',OLD_PREFIX).splitlines()
    for rel in old:
        a=subprocess.run(['git','show',base+':'+rel],cwd=checkout,capture_output=True,timeout=60);assert a.returncode==0 and a.stdout==(checkout/rel).read_bytes()
    status=dict(status='COMMITTED_PUSH_PENDING',commit=commit,parent=base,files=len(files),old_prefix_files_byte_unchanged=len(old),remote_verified=False,medical_images_public=0,push_returncode='NOT_RUN')
    jout(previous,status)
    push=subprocess.run(['git','push','origin','HEAD:main'],cwd=checkout,capture_output=True,text=True,timeout=600);status['push_returncode']=push.returncode
    if push.returncode:jout(previous,status);raise RuntimeError('Ordinary public push failed; local commit retained, no force')
    ref=git(checkout,'ls-remote','origin','refs/heads/main').split()[0];assert ref==commit
    status.update(status='PUSH_VERIFIED',remote_verified=True,remote_main_commit=ref,working_tree_clean=not bool(git(checkout,'status','--porcelain')),updated_utc=now());jout(previous,status)
    final['git_status']='PUSH_VERIFIED';final['git_commit']=commit;jout(root/'releases/FINAL_LTSA_TWO_ARM_STATUS_V3.json',final);print(status,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();main(a.config)
