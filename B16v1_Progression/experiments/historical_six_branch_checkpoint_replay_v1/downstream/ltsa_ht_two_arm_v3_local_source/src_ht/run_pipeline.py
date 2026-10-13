"""Six authorized independent runs, never a third arm; auto-sequence under exclusive project lock."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from common import config,setup,jread,jout,postflight,ARMS,SEEDS,now,state


def gpu_gate_observation(allow_shared_gpu=False,min_free_mib=12288):
    """Scheduling only: do not change the frozen model/training contract."""
    if min_free_mib<12288:
        raise ValueError('GPU scheduling requires at least12GiB free before each run')
    a=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True)
    g=subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
    if a.returncode or g.returncode:
        raise RuntimeError('CUDA_DRIVER_OR_GPU_UNAVAILABLE; no automatic repair/retry')
    try:
        free_mib=int(g.stdout.strip())
        other_count=len([int(line.strip()) for line in a.stdout.splitlines() if line.strip()])
    except ValueError as error:
        raise RuntimeError('GPU_STATUS_UNAVAILABLE_OR_NONNUMERIC; no launch') from error
    available=free_mib>=min_free_mib and (allow_shared_gpu or other_count==0)
    return dict(available=available,mode='USER_AUTHORIZED_SHARED_GPU' if allow_shared_gpu else 'EXCLUSIVE_IDLE_GPU',
                free_mib=free_mib,min_free_mib=min_free_mib,existing_compute_processes=other_count)


def main(path,allow_shared_gpu=False,min_free_mib=12288):
    c=config(path);root=setup(c['project_private']);env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    def command(script,args,log):
        with (root/'logs'/log).open('a',encoding='utf-8') as stream:
            r=subprocess.run([sys.executable,'-B','-u',str(root/'src_ht'/script),'--config',str(path),*args],cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError(f'{script} exited{r.returncode}; see private project logs')
    def wait_gpu():
        consecutive=0
        while consecutive<2:
            observation=gpu_gate_observation(allow_shared_gpu,min_free_mib)
            available=observation['available']
            consecutive=consecutive+1 if available else 0
            print(dict(stage='GPU_RESOURCE_GATE_PASS' if available else 'WAIT_GPU_RESOURCES',
                       action='NO_DUPLICATE_NO_INTERRUPTION',**observation),flush=True)
            if consecutive<2:time.sleep(5 if available else 15)
    try:
        if allow_shared_gpu:
            if min_free_mib<12288:raise ValueError('Shared GPU launch requires at least12GiB free')
            jout(root/'releases/GPU_SCHEDULING_POLICY_V3.json',dict(status='USER_AUTHORIZED_SHARED_GPU',min_free_mib=min_free_mib,
                 stable_checks=2,check_interval_seconds=5,training_contract_unchanged=True,other_jobs_interrupted=False,
                 authorization='Human confirmed continue after explicit shared-GPU scheduling question',updated_utc=now()))
        assert jread(root/'tests/ANONYMOUS_CPU_TEST_RESULTS_V3.json')['status']=='PASS'
        assert jread(root/'releases/ORIGINAL_PROGRESSION_LINEAGE_STATUS_V3.json')['status']=='PASS_MAPPING_FROZEN_BEFORE_FORMAL_VAL'
        command('build_result.py',[],'result_update_stdout.log')
        for arm in ARMS:
            prefix='REAL_ONLY' if arm==ARMS[0] else 'OP'
            if arm==ARMS[1]:
                assert jread(root/'releases/REAL_ONLY_STAGE_COMPLETE_V3.json')['status']=='REAL_BASELINE_COMPLETED'
                command('validate_original_progression_lineage.py',['--verify-only'],'op_mapping_revalidation_stdout.log')
            for seed in SEEDS:
                run=root/'checkpoints'/arm/f'seed_{seed}';done=run/'RUN_COMPLETED.json'
                if done.exists():assert jread(done)['status']=='COMPLETE_RELOAD_VERIFIED';continue
                if not (run/'TRAINING_COMPLETED_PENDING_RELOAD_V3.json').exists():
                    wait_gpu();command('train_ltsa_ht.py',['--arm',arm,'--seed',str(seed),'--resume-if-valid'],f'{prefix}_seed_{seed}.log')
                wait_gpu();command('reload_and_ensemble.py',['--arm',arm,'--seed',str(seed)],f'{prefix}_seed_{seed}_reload.log')
                postflight(root);command('build_result.py',[],'result_update_stdout.log')
                print({'verified_run_completed':arm,'seed':seed,'updated_utc':now()},flush=True)
            command('reload_and_ensemble.py',['--arm',arm],f'{prefix}_ensemble.log');command('build_result.py',[],'result_update_stdout.log')
        command('summarize_ltsa_ht.py',[],'final_summaries.log')
        command('draw_ltsa_figures.py',[],'final_figures.log')
        command('write_paper_materials.py',[],'final_writing.log')
        command('build_result.py',[],'result_update_stdout.log')
        command('publish_ltsa_package.py',[],'public_archive.log')
        command('build_result.py',[],'result_update_stdout.log')
        print({'pipeline':'TWO_ARM_COMPLETED','runs':6,'no_third_arm':True},flush=True)
    except Exception as error:
        jout(root/'releases/ENGINEERING_INCOMPLETE_V3.json',dict(status='ENGINEERING_INCOMPLETE',exception_type=type(error).__name__,error_detail_private=repr(error),
                 automatic_restart=False,partial_runs_retained=True,Test_access=0,failed_utc=now()))
        state(root,'ENGINEERING_INCOMPLETE_PENDING_SAFE_REVIEW',error_type=type(error).__name__)
        if (root/'src_ht/build_result.py').exists():
            try:command('build_result.py',[],'result_update_stdout.log')
            except Exception:pass
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--allow-shared-gpu',action='store_true');p.add_argument('--min-free-mib',type=int,default=12288)
    a=p.parse_args();main(a.config,a.allow_shared_gpu,a.min_free_mib)
