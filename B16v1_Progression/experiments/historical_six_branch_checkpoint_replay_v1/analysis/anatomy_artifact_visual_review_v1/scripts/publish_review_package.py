"""Isolated allowlisted public archive, preserving old tracked files and ordinary main history."""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
import pandas as pd
from review_common import setup,read_json,json_out,csv_out,now
from review_publication_scope import scan,SCRIPTS,WRITING,PROTOCOL,TABLES

REMOTE='git@github-baone:Baone0330/Study.git'
PREFIX='B16v1_Progression/experiments/historical_six_branch_checkpoint_replay_v1/analysis/anatomy_artifact_visual_review_v1'
OLD_PREFIX='B16v1_Progression/experiments/historical_six_branch_checkpoint_replay_v1/'
EXPECTED_BASE='5cdb25688fc702da057ce4821db5d15e5c1f991e'


def git(checkout,*args):
    env=os.environ.copy();env['GIT_OPTIONAL_LOCKS']='0'
    p=subprocess.run(['git',*args],cwd=checkout,capture_output=True,text=True,timeout=90,env=env)
    if p.returncode:raise RuntimeError('Git '+args[0]+' failed: '+p.stderr[-1200:])
    return p.stdout.strip()


def build_package(root):
    dest=root/'github_public_stage';dest.mkdir(exist_ok=True)
    def copy(src,rel):
        p=dest/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/src,p);assert (root/src).read_bytes()==p.read_bytes()
    for name in SCRIPTS:copy('scripts/'+name,'scripts/'+name)
    for name in WRITING:copy('writing/'+name,'writing/'+name)
    for name in PROTOCOL:copy('protocol/'+name,'protocol/'+name)
    for name in TABLES:copy('tables/'+name,'tables/'+name)
    for folder in ['public_nonmedical','source_data_public','captions_cn_en']:
        for p in (root/'figures'/folder).iterdir():
            if p.is_file():copy(str(p.relative_to(root)),str(p.relative_to(root)))
    copy('figures/export_qa/NONMEDICAL_FIGURE_EXPORT_QA_V1.csv','figures/export_qa/NONMEDICAL_FIGURE_EXPORT_QA_V1.csv')
    copy('tests/ANONYMOUS_TEST_RESULTS_V1.json','tests/ANONYMOUS_TEST_RESULTS_V1.json')
    copy('writing/PUBLIC_REVIEW_RESULT_V1.md','PUBLIC_REVIEW_RESULT_V1.md')
    copy('reviews_private/review_guideline_cn_en.md','review_templates/REVIEW_GUIDELINE_CN_EN_V1.md')
    f=pd.read_csv(root/'reviews_private/blank_clinician_review_form.csv');csv_out(dest/'review_templates/BLANK_CLINICIAN_SCALE_HEADER_ONLY_V1.csv',f.iloc[:0])
    (dest/'requirements-cpu.txt').write_text('numpy==2.2.6\npandas==2.3.3\nscipy==1.15.3\nopencv-python==4.13.0.92\nPillow==12.2.0\nmatplotlib==3.10.9\npyarrow==24.0.0\n',encoding='utf-8')
    (dest/'README.md').write_text('''# Historical Original Progression anatomy/artifact engineering audit

This new subtree contains only reproducible CPU code, anonymous contract tests, aggregate tables/figures, bilingual supplement material and a case-free blank scale. It is separate from the prior frozen quality and ablation packages. Publication provenance is the Git commit history; no medical pixels, patient mappings, actual ratings, weights or training logs are archived.

Actual scope:9,270 ON/OFF pairs,1,545nodes,860patients;13,200 REAL native views/859patients. Current31,740 reads reproduced prior flags/counts with0 mismatch. Inherited12-node72-pair core;48-pair distinct-patient stratified sample;36-pair/72-image expert subset. Four strata omit420 OFF-only cases from sampling strata only; all remain in population statistics. Nonuniform sampling has no calibrated prevalence weights.

Human/clinician medical viewing: NOT_RUN. Visibility: NOT_REVIEWED. REAL display: WITHHELD. PHI and public medical-image approval: PENDING/NOT_APPROVED. Geometric positions are not thyroid ROI; candidates are not diagnosed artifacts. Prior lower FID/KID for ON and current more candidate geometry are different endpoints. No new sampling/training/FID/KID/HT/Test evaluation.

Anonymous checks: `python -B scripts/test_review_contracts.py`. Full private-data execution is not portable from this public package: frozen private assets/authorization are prerequisites. Do not fabricate source data or run private card displays from a public/cloud service. Optional expert import intentionally fails closed if unapproved ratings appear; a separately approved real-review phase is required.

CPU workflow: freeze identities and source metadata; reproduce/localize flags; aggregate patient-equal paired bootstrap; prepare private review assets without displaying them; record actual missingness; prepare material and draw six aggregate figures; actually inspect nonmedical figures; verify protection/exports and publish only allowlisted files. No GPU or torch dependency in this audit.

本归档是技术复核工程完成，不是医学结构认证；未获批的医学图板和卡片仅服务器候审，不公开。六组图实际非医学视觉QA通过；人工医学观察、医生评分与临床真实性均未建立。
''',encoding='utf-8')
    json_out(dest/'release/PUBLIC_ENGINEERING_STATUS_V1.json',dict(status='ENGINEERING_COMPLETED_WITH_MEDICAL_REVIEW_LIMITATIONS',medical_review='NOT_RUN',clinicians=0,actual_ratings=0,
            technical_localization_images=31740,original_flag_mismatches=0,patient_bootstrap_repeats=2000,medical_pixels_public=0,
            source_protection='PASS',nonmedical_visual_QA='PASS_AFTER_ACTUAL_VIEW',new_training=0,new_SD_sampling=0,new_FID_KID=0,test_access=0,publication_provenance='repository commit history'))
    return dest


def main(output_root,prepare_only=False):
    root=setup(output_root);assert read_json(root/'release/VERIFICATION_SUMMARY_V1.json')['status']=='PASS'
    package=build_package(root);audit=scan(package,root);json_out(root/'release/PUBLIC_PRIVACY_AUDIT_V1.json',audit)
    assert audit['status']=='PASS',audit['issues']
    checkout=root/'publication_checkout_local_v2'
    if not (checkout/'.git').exists():
        assert not checkout.exists()
        source=Path(read_json(root/'protocol/FROZEN_VISUAL_AND_ARTIFACT_REVIEW_CONTRACT_V1.json')['quality_root_private'])/'publication_checkout_v1'
        assert git(source,'rev-parse','HEAD')==EXPECTED_BASE
        p=subprocess.run(['git','clone','--no-hardlinks','--branch','main','--single-branch',str(source),str(checkout)],capture_output=True,text=True,timeout=120)
        if p.returncode:raise RuntimeError('Isolated clone failed: '+p.stderr[-1200:])
        git(checkout,'remote','set-url','origin',REMOTE)
    assert git(checkout,'remote','get-url','origin')==REMOTE
    git(checkout,'config','user.name','study');git(checkout,'config','user.email','l3026714004@outlook.com')
    planfile=root/'release/GIT_PUBLICATION_PLAN_V1.json'
    if planfile.exists():base=read_json(planfile)['base_commit']
    else:
        assert not git(checkout,'status','--porcelain')
        base=git(checkout,'rev-parse','HEAD');assert base==EXPECTED_BASE,'Concurrent upstream change; stop instead of overriding'
        assert git(checkout,'ls-remote','origin','refs/heads/main').split()[0]==base
        assert not git(checkout,'ls-tree','-r','--name-only',base,'--',PREFIX)
        json_out(planfile,dict(base_commit=base,prefix=PREFIX,remote=REMOTE,author='study <l3026714004@outlook.com>',started_utc=now()))
    target=checkout/PREFIX
    for rel in audit['files']:
        p=target/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(package/rel,p);assert p.read_bytes()==(package/rel).read_bytes()
    copied=scan(target,root);assert copied['status']=='PASS'
    env=os.environ.copy();env['PYTHONPATH']=str(target/'scripts')
    test=subprocess.run([sys.executable,'-B',str(target/'scripts/test_review_contracts.py')],cwd=target,capture_output=True,text=True,env=env,timeout=60)
    assert test.returncode==0,test.stderr
    git(checkout,'add','-f','--',*[PREFIX+'/'+p for p in audit['files']])
    staged=git(checkout,'diff','--cached','--name-only').splitlines();assert len(staged)==audit['file_count'] and all(p.startswith(PREFIX+'/') for p in staged)
    assert not git(checkout,'diff','--name-only')
    json_out(root/'release/GIT_PRECOMMIT_AUDIT_V1.json',dict(status='ALLOWLIST_AND_COPIED_ANONYMOUS_TESTS_PASS',files=len(staged),privacy='PASS',base=base,
               explicit_individual_force_add_for_inherited_ignore_rules=True,parent_ignore_changed=False))
    if prepare_only:print({'status':'PRECOMMIT_PASS','files':len(staged),'base':base});return
    assert git(checkout,'ls-remote','origin','refs/heads/main').split()[0]==base,'Concurrent upstream write; no force push'
    git(checkout,'commit','-m','Archive frozen historical CPFM anatomy and artifact positional review')
    commit=git(checkout,'rev-parse','HEAD');names=git(checkout,'diff','--name-only',base,commit).splitlines()
    assert len(names)==audit['file_count'] and all(p.startswith(PREFIX+'/') for p in names)
    old=git(checkout,'ls-tree','-r','--name-only',base,'--',OLD_PREFIX).splitlines()
    for rel in old:
        p=subprocess.run(['git','show',base+':'+rel],cwd=checkout,capture_output=True,timeout=60);assert p.returncode==0 and p.stdout==(checkout/rel).read_bytes()
    assert not git(checkout,'status','--porcelain')
    status=dict(status='COMMITTED_PUSH_PENDING',commit=commit,parent=base,files=len(names),old_prefix_files_byte_unchanged=len(old),privacy='PASS',remote_verified=False,
                new_subtree_only=True,working_tree_clean=True,push_returncode='NOT_RUN',updated_utc=now())
    json_out(root/'release/GIT_PUBLICATION_STATUS_V1.json',status)
    push=subprocess.run(['git','push','origin','HEAD:main'],cwd=checkout,capture_output=True,text=True,timeout=600);status['push_returncode']=push.returncode
    if push.returncode:
        status['status']='PUSH_NOT_DONE';json_out(root/'release/GIT_PUBLICATION_STATUS_V1.json',status);raise RuntimeError('Ordinary push failed; local commit retained, no force or automatic overwrite')
    ref=git(checkout,'ls-remote','origin','refs/heads/main').split()[0];assert ref==commit
    status.update(status='PUSH_VERIFIED',remote_verified=True,remote_main_commit=ref,updated_utc=now());json_out(root/'release/GIT_PUBLICATION_STATUS_V1.json',status)
    print(status,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args();main(a.output_root,a.prepare_only)
