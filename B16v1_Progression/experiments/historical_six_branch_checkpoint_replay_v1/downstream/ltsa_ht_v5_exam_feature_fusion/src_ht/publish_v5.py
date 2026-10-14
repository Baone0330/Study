"""Explicit safe public allowlist; call only after actual four-figure review."""
import argparse
import ast
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
import pandas as pd
import yaml
from common import config,setup,jread,jout,now
REMOTE='git@github-baone:Baone0330/Study.git'
PREFIX='B16v1_Progression/experiments/historical_six_branch_checkpoint_replay_v1/downstream/ltsa_ht_v5_exam_feature_fusion'
def git(folder,*args):
    env=os.environ.copy();env['GIT_OPTIONAL_LOCKS']='0'
    a=subprocess.run(['git',*args],cwd=folder,env=env,capture_output=True,text=True,timeout=60)
    if a.returncode:raise RuntimeError('Git '+args[0]+' failed; own checkout retained, no force/reset')
    return a.stdout.strip()
def values_pattern(values):
    values={str(x) for x in values if len(str(x))>=5}
    return re.compile(r'(?<![A-Za-z0-9])(?:'+'|'.join(re.escape(x) for x in sorted(values,key=len,reverse=True))+r')(?![A-Za-z0-9])') if values else None
def main(path):
    c=config(path);root=setup(c['project_private']);final=jread(root/'releases/FINAL_V5_STATUS.json')
    assert final['RUNS_AND_METRICS_COMPLETE']
    review=jread(root/'figures/export_qa/V5_ACTUAL_VISUAL_REVIEW.json');assert review['status']=='PASS_AGENT_ACTUAL_VIEW' and review['reviewed_figure_groups']==4
    assert jread(root/'writing/V5_WRITING_QA.json')['status']=='PASS_GENERATED_FROM_ACTUAL_AGGREGATES'
    receipt=root/'releases/GITHUB_STATUS_V5.json'
    if receipt.exists() and jread(receipt)['status']=='GITHUB_PUSH_VERIFIED':print('Existing public push retained');return
    package=root/'github_public_stage_v5';package.mkdir(exist_ok=True);files=[]
    def write(rel,text):
        p=package/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text);files.append(rel)
    def copy(p,rel):
        dest=package/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);files.append(rel)
    for p in sorted((root/'src_ht').glob('*.py')):
        text=p.read_text();paths=sorted(set(re.findall(r'/(?:data|home)/[^\s\x27\x22]+',text)),key=len,reverse=True)
        for i,value in enumerate(paths):text=text.replace(value,'REQUIRED_AUTHORIZED_LOCAL_PATH_'+str(i))
        ast.parse(text);write('src_ht/'+p.name,text)
    template={k:('REQUIRED_LOCAL_PRIVATE_PATH' if k.endswith('_private') else v) for k,v in c.items()}
    for k in ['frozen_utc','augmentation_mapping_frozen_utc']:template[k]='PRIVATE_TIMESTAMP_OMITTED'
    write('configs/LTSA_HT_V5_CONFIG_TEMPLATE.yaml',yaml.safe_dump(template,sort_keys=False,allow_unicode=True))
    for p in sorted((root/'tables').glob('*.csv')):copy(p,'tables/'+p.name)
    for p in sorted((root/'writing').glob('*.md')):copy(p,'writing/'+p.name)
    for folder in ['nonmedical','source_data','captions','export_qa']:
        for p in sorted((root/'figures'/folder).iterdir()):
            if p.is_file() and p.suffix in ['.png','.svg','.csv','.md','.json']:copy(p,'figures/'+folder+'/'+p.name)
    copy(root/'tests/ANONYMOUS_CPU_TEST_RESULTS_V5.json','tests/ANONYMOUS_CPU_TEST_RESULTS_V5.json')
    copy(Path(c['snapshot_private'])/'LICENSE','LICENSE_LTSA_SOURCE_ATTRIBUTION.txt')
    write('README.md', '''# LTSA-HT V5 matched next-real Exam feature fusion

Six fresh downstream runs: REAL-only and historical Original Progression Exam fusion, seeds42/3407/2026. Both arms contain the same fusion architecture and13,020,738 trainable parameters. No old trained checkpoint warm start. Strict independent reload, same189-patient three-seed probability ensembles and2,000 paired bootstrap contrasts. Source LTSA is an authorized local prerequisite, not downloaded or redistributed here; available source Git revision was unknown. Local original sinusoidal TPE is loaded from its supplied snapshot; offline official ResNet18 initialization is required.

Support:860Train(617negative/243positive),189Val(131negative/58positive),4,585/991 REAL Exams and18,413/3,861 native views. Same686 bounded CPFM-ON Original Progression nodes/4,116 images,597augmented/263unaugmented training patients. Frozen queries attach to the first strictly later REAL Exam;618REAL Exams receive nodes(550one,68two). Exclude758target-year and101after-last-real nodes. No new images, third arm, Official Test content, file-content digests or input reselection.

Every REAL view and six images per node pass the shared trainable ResNet18. Node means are equally averaged per attached Exam. Two separate LayerNorm512 modules, scalar sigmoid gate1024→1 and difference projection512→512 yield a masked residual of fixed scale0.1. Unattached Exams bypass all new operations exactly; no extra time token, date replacement or synthetic-real visit equivalence. Main initialization/RNG and inherited temporal masks remain unchanged; standard BatchNorm statistics can change under synthetic exposure. Native REAL views are encoded first, then synthetic views, in differentiable chunks16. The added module preserves baseline global RNG on initialization.

Public adapters are auditable templates, not directly runnable without authorized private inputs. Fill every REQUIRED_AUTHORIZED_LOCAL_PATH and config path placeholder. No patient IDs/dates/reports/pixels/per-patient labels or probabilities/checkpoints/mappings/caches/credentials are released. Anonymous date fixtures are explicitly fictional. The IPC-only engineering repair uses a short relative cache/ipc directory under the private working root without changing two spawn workers, data, RNG or optimizer arithmetic. Failure logs are retained privately.

Internal V3 filename suffixes are compatibility names newly written inside the V5 project, not copied V3 completion flags or trained results. Canonical V5 seed, ensemble, bootstrap and cross-protocol tables explicitly identify their version and arms. After training, only non-numerical figure, caption, bilingual-report and publication code was repaired; completed weights, predictions, aggregate statistics, frozen configs and scientific model/trainer code were checked unchanged by metadata.

Four nonmedical600dpi PNG/editableSVG figure groups and five bilingual texts are included. Automatic export QA and actual agent visual review are distinct receipts; no clinician or external clinical validation is implied. Same internal Val informed model design, selection, stopping and thresholds. Paired bootstrap omits adaptive-selection uncertainty. V3/V4 versus V5 is a descriptive cross-protocol comparison, not unique causality of token count. CLIMATv2 fixedK6 historical references are not recomputed or input-equivalent.

Original attribution: Holste et al., Harnessing the power of longitudinal medical imaging for eye disease prognosis using Transformer-based sequence modeling, npj Digital Medicine7,216(2024), DOI10.1038/s41746-024-01207-4, as specified in the supplied experimental protocol. Local source license is retained for attribution.
''')
    private_review=jread(root/'audit/V5_ACTUAL_VISUAL_REVIEW_PRIVATE.json')
    from common import metadata
    assert all(metadata(row['path'])==row for row in private_review['image_metadata_private']),'Final reviewed images changed before publication'
    master=pd.read_parquet(root/'manifests_private/HT_LTSA_CANONICAL_PATIENT_WINDOWS_V1.parquet')
    assert master.patient_key_private.astype(str).str.len().min()>=5,'Short private identifiers require contextual review before publication'
    ids=values_pattern(master.patient_key_private.astype(str));dates=values_pattern(e['date'] for s in master.exams_private_json for e in json.loads(s))
    anonymous={'2020-01-01','2022-01-01','2024-01-01','2021-01-01','2025-01-01'};issues=[]
    for rel in files:
        p=package/rel
        if p.suffix=='.csv':
            columns=set(pd.read_csv(p,nrows=0).columns)
            forbidden={'sample_index','patient_index','patient_id','patient_key_private','y_true','p_positive','native_path_private','exam_date_private','target_HT','label','labels'}
            if columns&forbidden:issues.append(dict(file=rel,reason='PATIENT_LEVEL_CSV_COLUMN'))
        if p.suffix in ['.md','.py','.csv','.yaml','.txt','.json','.svg']:
            text=p.read_text()
            if re.search(r'/(?:data|home)/|[A-Z]:[/\\]Users[/\\]',text):issues.append(dict(file=rel,reason='PRIVATE_ABSOLUTE_PATH'))
            if ids and ids.search(text):issues.append(dict(file=rel,reason='ACTUAL_PRIVATE_PATIENT_VALUE'))
            if dates:
                matches={m.group(0) for m in dates.finditer(text)}
                # Only documented fictional source-test date literals may coincide with a clinical date.
                if rel=='src_ht/tests_v5.py':matches-=anonymous
                if matches:issues.append(dict(file=rel,reason='ACTUAL_CLINICAL_DATE_VALUE'))
            if re.search(r'gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{24,}|sk-[A-Za-z0-9_-]{24,}|-----BEGIN (OPENSSH|RSA|EC) PRIVATE KEY-----',text):issues.append(dict(file=rel,reason='CREDENTIAL_PATTERN'))
            if p.suffix=='.svg' and '<image' in text:issues.append(dict(file=rel,reason='SVG_RASTER_EMBEDDED'))
        if p.suffix=='.png' and rel not in {'figures/nonmedical/'+n+'.png' for n in ['Fig01_V5_ROC_PR','Fig02_V3_V4_V5_Descriptive_Comparison','Fig03_V5_Seeds_and_Ensemble','Fig04_V5_Exam_Fusion_Internal_Architecture']}:
            issues.append(dict(file=rel,reason='NONMEDICAL_BINARY_ALLOWLIST_FAILURE'))
    assert len(files)==len(set(files)) and set(files)=={p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()}
    jout(root/'releases/PUBLIC_PRIVACY_AUDIT_V5.json',dict(status='PASS' if not issues else 'FAIL',files=len(files),issues=issues,
         medical_images_public=0,actual_patient_record_public=0,anonymous_fixture_date_exception_scope='source tests only, explicitly fictional'))
    assert not issues,'Privacy scan failure; no staging/push'
    checkout=root/'publication_checkout_v5'
    if not (checkout/'.git').exists():
        source=Path(c['quality_private']).parent/'b16v1_historical_original_progression_anatomy_artifact_review_v1/publication_checkout_local_v2'
        assert source.is_dir()
        a=subprocess.run(['git','clone','--no-hardlinks','--branch','main',str(source),str(checkout)],capture_output=True,text=True,timeout=60);assert a.returncode==0
        git(checkout,'remote','set-url','origin',REMOTE)
    assert not git(checkout,'status','--porcelain')
    remote=git(checkout,'ls-remote','origin','refs/heads/main').split()[0]
    if git(checkout,'rev-parse','HEAD')!=remote:git(checkout,'fetch','origin','main');git(checkout,'merge','--ff-only','FETCH_HEAD')
    base=git(checkout,'rev-parse','HEAD');assert base==remote and not git(checkout,'ls-tree','-r','--name-only',base,'--',PREFIX)
    git(checkout,'config','user.name','study');git(checkout,'config','user.email','l3026714004@outlook.com')
    target=checkout/PREFIX
    for rel in files:
        p=target/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(package/rel,p)
    git(checkout,'add','-f','--',*[PREFIX+'/'+r for r in files]);staged=git(checkout,'diff','--cached','--name-only').splitlines()
    assert len(staged)==len(files) and set(staged)=={PREFIX+'/'+r for r in files}
    git(checkout,'commit','-m','Add matched LTSA HT V5 next-real Exam feature fusion and internal validation')
    commit=git(checkout,'rev-parse','HEAD');changed=git(checkout,'diff-tree','--no-commit-id','--name-only','-r',commit).splitlines()
    assert set(changed)==set(staged),'No change outside new V5 public prefix'
    status=dict(status='COMMITTED_PUSH_PENDING',commit=commit,parent=base,files=len(files),old_V3_V4_paths_changed=0,remote_verified=False)
    jout(receipt,status)
    push=subprocess.run(['git','push','origin','HEAD:main'],cwd=checkout,capture_output=True,text=True,timeout=60)
    if push.returncode:
        status.update(status='FAILED_WITH_REASON',reason='NORMAL_PUSH_FAILED_LOCAL_COMMIT_RETAINED');jout(receipt,status);raise RuntimeError('Normal push failed; no force/reset')
    ref=git(checkout,'ls-remote','origin','refs/heads/main').split()[0];assert ref==commit
    status.update(status='GITHUB_PUSH_VERIFIED',remote_verified=True,remote_main_commit=ref,working_tree_clean=not bool(git(checkout,'status','--porcelain')),verified_utc=now());jout(receipt,status)
    final.update(GITHUB_PUSH_VERIFIED=True,git_commit=commit,Git='GITHUB_PUSH_VERIFIED');jout(root/'releases/FINAL_V5_STATUS.json',final);print(status,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);main(p.parse_args().config)
