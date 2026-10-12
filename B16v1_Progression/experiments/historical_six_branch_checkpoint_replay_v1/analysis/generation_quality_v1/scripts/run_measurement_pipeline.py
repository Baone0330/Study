"""One locked, resumable, offline-only measurement chain; no new image sampling or training."""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import argparse
import fcntl
import subprocess
import sys
from pathlib import Path
from b16_quality_common import setup, state, atomic_json, now, postflight


def main(output_root,resume_at=None):
    root=setup(output_root)
    with (root/'release/measurement.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Measurement pipeline already owns active lock; no duplicate launched')
        atomic_json(root/'release/MEASUREMENT_PID_V1.json',dict(pid=os.getpid(),started_utc=now(),new_training_steps=0,new_image_sampling=0))
        steps=[('extract_quality_features.py',[]),('calculate_paired_image_changes.py',[]),
               ('calculate_fid_kid.py',[]),('aggregate_patient_cluster_metrics.py',[])]
        if resume_at:
            assert resume_at=='aggregate_patient_cluster_metrics.py'
            for name in ['IMAGE_DECODE_AND_TECHNICAL_QA_V1.parquet','PAIRED_ON_OFF_PIXEL_FEATURE_STATS_V1.parquet',
                         'PAIRED_FULL_ON_AUXILIARY_ROUTE_STATS_V1.parquet','NODE_SIX_SEED_DIVERSITY_V1.parquet',
                         'QUALITY_DISTRIBUTION_METRICS_V1.csv','QUALITY_PATIENT_CLUSTER_BOOTSTRAP_V1.csv']:
                assert (root/'metrics'/name).is_file(),name
            atomic_json(root/'release/MEASUREMENT_RECOVERY_V1.json',dict(status='SAME_FROZEN_INPUT_AGGREGATION_RESUME',updated_utc=now(),
                        resolved_issue='Duplicate pair_mean keyword in summary serialization; no metric/queue/feature changes',resume_at=resume_at))
            steps=steps[-1:]
        try:
            for script,args in steps:
                state(root,'MEASUREMENT_STAGE_RUNNING',stage=script,pid=os.getpid())
                print(now(), 'START',script,flush=True)
                log=root/'logs'/f'{Path(script).stem}_stdout.log'
                with log.open('a',encoding='utf-8') as stream:
                    stream.write('\nSESSION '+now()+'\n');stream.flush()
                    subprocess.run([sys.executable,'-B','-u',str(root/'scripts'/script),'--output-root',str(root),*args],
                                   stdout=stream,stderr=subprocess.STDOUT,check=True)
                print(now(),'COMPLETE',script,flush=True)
            protection=postflight(root)
            atomic_json(root/'release/MEASUREMENT_COMPLETED_V1.json',dict(status='MEASUREMENT_COMPLETED',completed_utc=now(),
                        source_metadata_postflight=protection['status'],new_training=0,new_image_sampling=0,official_test_access=0))
            state(root,'MEASUREMENTS_COMPLETE_REPORT_AND_QA_PENDING')
        except BaseException as error:
            atomic_json(root/'release/MEASUREMENT_FAILURE_V1.json',dict(status='FAILED_PRESERVED_CACHES',stage=script,error=repr(error),updated_utc=now(),
                        recovery='Diagnose first; resume same frozen queue and models after confirming no active process/lock. Never quality-filter inputs.'))
            state(root,'MEASUREMENT_FAILED',stage=script,error=repr(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',required=True);p.add_argument('--resume-at',choices=['aggregate_patient_cluster_metrics.py']);a=p.parse_args();main(a.output_root,a.resume_at)
