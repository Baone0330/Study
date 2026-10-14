"""One locked project: six separate fresh Python runs, strict reload and safe finalization."""
import argparse
import os
import subprocess
import sys
import time
from common import config,setup,jread,jout,ARMS,SEEDS,now,postflight,state
def gpu_gate_observation(allow_shared_gpu=True,min_free_mib=12288):
    if min_free_mib<12288:raise ValueError('At least12GiB launch free memory required')
    a=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True)
    g=subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True)
    if a.returncode or g.returncode:raise RuntimeError('CUDA_DRIVER_OR_GPU_UNAVAILABLE; no automatic retry')
    try:free=int(g.stdout.strip());n=len([int(v.strip()) for v in a.stdout.splitlines() if v.strip()])
    except ValueError as e:raise RuntimeError('Non-numeric GPU observation; stop') from e
    return dict(available=free>=min_free_mib and (allow_shared_gpu or n==0),free_mib=free,existing_compute_processes=n,min_free_mib=min_free_mib)
def wait_gpu(allow_shared_gpu=True,min_free_mib=12288):
    consecutive=0
    while consecutive<2:
        a=gpu_gate_observation(allow_shared_gpu,min_free_mib);consecutive=consecutive+1 if a['available'] else 0
        print(dict(stage='GPU_GATE' if a['available'] else 'WAIT_GPU',stable_checks=consecutive,other_jobs_interrupted=False,**a),flush=True)
        if consecutive<2:time.sleep(5 if a['available'] else 15)
def main(path):
    c=config(path);root=setup(c['project_private']);env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',TMPDIR=str(root/'cache'))
    def command(script,args,log):
        with (root/'logs'/log).open('a') as out:
            a=subprocess.run([sys.executable,'-B','-u',str(root/'src_ht'/script),'--config',str(path),*args],cwd=root,env=env,stdout=out,stderr=subprocess.STDOUT)
        if a.returncode:raise RuntimeError(f'{script} exited{a.returncode}; see own logs')
    try:
        assert jread(root/'releases/V5_PREPARATION_COMPLETE.json')['status']=='PASS'
        assert jread(root/'tests/ANONYMOUS_CPU_TEST_RESULTS_V5.json')['status']=='PASS'
        assert jread(root/'releases/TRAINING_CONTRACT_FROZEN_V3.json')['status']=='PASS_FROZEN_BEFORE_FORMAL_VAL'
        jout(root/'releases/GPU_SCHEDULING_POLICY_V5.json',dict(min_free_mib=12288,stable_checks=2,interval_seconds=5,shared_if_sufficient=True,other_jobs_interrupted=False,source='authorized V5 runbook scheduling contract'))
        command('build_result.py',[],'result_update.log')
        for arm in ARMS:
            prefix='REAL_ONLY' if arm==ARMS[0] else 'OP_FUSION'
            for seed in SEEDS:
                run=root/'checkpoints'/arm/f'seed_{seed}'
                if (run/'RUN_COMPLETED.json').exists():assert jread(run/'RUN_COMPLETED.json')['status']=='COMPLETE_RELOAD_VERIFIED';continue
                if not (run/'TRAINING_COMPLETED_PENDING_RELOAD_V3.json').exists():
                    wait_gpu();command('train_ltsa_ht.py',['--arm',arm,'--seed',str(seed),'--resume-if-valid'],f'{prefix}_seed_{seed}.log')
                wait_gpu();command('reload_and_ensemble.py',['--arm',arm,'--seed',str(seed)],f'{prefix}_seed_{seed}_reload.log')
                postflight(root);command('build_result.py',[],'result_update.log');print(dict(verified_run_completed=arm,seed=seed,utc=now()),flush=True)
            command('reload_and_ensemble.py',['--arm',arm],f'{prefix}_ensemble.log');command('build_result.py',[],'result_update.log')
        command('finalize_v5.py',[],'finalize.log')
        command('draw_v5.py',[],'figures.log')
        command('write_v5.py',[],'writing.log')
        command('build_result.py',[],'result_update.log')
        print(dict(pipeline='RUNS_AND_METRICS_COMPLETE',actual_visual_review='PENDING',Git='NOT_ATTEMPTED',no_performance_gate=True),flush=True)
    except Exception as e:
        jout(root/'releases/ENGINEERING_INCOMPLETE_V5.json',dict(status='ENGINEERING_INCOMPLETE',type=type(e).__name__,error_private=repr(e),automatic_restart=False,partial_runs_retained=True,failed_utc=now()))
        state(root,'ENGINEERING_INCOMPLETE_PENDING_SAFE_REVIEW',error_type=type(e).__name__)
        try:command('build_result.py',[],'result_update.log')
        except Exception:pass
        raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);main(p.parse_args().config)
