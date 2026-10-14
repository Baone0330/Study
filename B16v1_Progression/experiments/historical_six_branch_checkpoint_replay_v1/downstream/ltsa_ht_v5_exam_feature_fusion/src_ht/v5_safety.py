"""Python open-event audit only; not an OS-wide access proof."""
import atexit
import json
import os
import sys
from pathlib import Path
V3=Path('REQUIRED_AUTHORIZED_LOCAL_PATH_1')
V4=Path('REQUIRED_AUTHORIZED_LOCAL_PATH_0')
BOUNDARY=None
REFERENCE_ENABLED=False
def validate_checkpoint_path(path,root):
    p=Path(path)
    if not p.resolve().is_relative_to(Path(root).resolve()/'checkpoints') or p.is_symlink():
        raise PermissionError('V5_CHECKPOINT_OUTSIDE_OWN_RUNS_OR_SYMLINK')
    return p
def enable_history_reference(root):
    global REFERENCE_ENABLED
    from common import ARMS,SEEDS,jread
    for arm in ARMS:
        for seed in SEEDS: assert jread(Path(root)/'checkpoints'/arm/f'seed_{seed}'/'RUN_COMPLETED.json')['status']=='COMPLETE_RELOAD_VERIFIED'
    REFERENCE_ENABLED=True
def install_boundary(c):
    global BOUNDARY
    if BOUNDARY is not None:return
    from common import ROOT
    root=Path(c['project_private']).resolve(); assert root==ROOT
    protected=[V3,V4,Path(c['local_ltsa_source_private']),Path(c['source_old_ablation_private']),Path(c['raw_data_private']),Path('REQUIRED_AUTHORIZED_LOCAL_PATH_2'),Path(c['resnet18_weights_private']).parent]
    counts=dict(protected_write_attempts=0,old_trained_weight_read_attempts=0,old_probability_read_attempts=0,official_initial_weight_reads=0,own_checkpoint_reads=0,readonly_v3_pixel_cache_reads=0,raw_target_image_reads=0)
    def observe(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve(); mode=args[1] if len(args)>1 else None; flags=args[2] if len(args)>2 else 0
        writing=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if writing:
            if not p.is_relative_to(root):
                counts['protected_write_attempts']+=1; raise PermissionError('V5_WRITES_OUTSIDE_NEW_ROOT_BLOCKED')
            return
        if p.suffix in ['.pt','.pth']:
            if p==Path(c['resnet18_weights_private']).resolve():counts['official_initial_weight_reads']+=1;return
            if p.is_relative_to(root/'checkpoints'):validate_checkpoint_path(p,root);counts['own_checkpoint_reads']+=1;return
            counts['old_trained_weight_read_attempts']+=1;raise PermissionError('V5_OLD_TRAINED_WEIGHT_READ_BLOCKED')
        if any(p.is_relative_to(v/'metrics/private') or p.is_relative_to(v/'checkpoints') for v in [V3,V4]) and not REFERENCE_ENABLED:
            counts['old_probability_read_attempts']+=1;raise PermissionError('OLD_RESULTS_BEFORE_SIX_V5_RUNS_COMPLETE')
        if p.is_relative_to(Path(c['raw_data_private'])) and p.suffix.lower() in ['.jpg','.jpeg','.png','.bmp','.tif','.tiff']:
            counts['raw_target_image_reads']+=1;raise PermissionError('ONLY_FROZEN_HISTORY_PIXEL_CACHES_ALLOWED')
        if p.is_relative_to(V3/'cache') and p.suffix=='.npy': counts['readonly_v3_pixel_cache_reads']+=1
    sys.addaudithook(observe);BOUNDARY=counts
    def record():
        p=root/'audit'/f'FILE_ACCESS_BOUNDARY_V5_PID_{os.getpid()}.json'
        p.write_text(json.dumps(dict(pid=os.getpid(),reference_enabled_after_six_runs=REFERENCE_ENABLED,**counts),indent=2)+'\n')
    atexit.register(record)
def initialization_evidence(root,run,c,arm,seed,model,optimizer):
    from common import jout,now,metadata
    assert not optimizer.state and not list(run.glob('*.pt'))
    n=sum(p.numel() for p in model.parameters() if p.requires_grad);assert n==13020738
    jout(run/'RUN_INITIALIZATION_V5.json',dict(task=c['task'],arm=arm,seed=seed,pid=os.getpid(),started_utc=now(),fresh_model=True,fresh_optimizer=True,
         optimizer_state_entries=0,warm_start_from_V3_V4=False,old_checkpoint_loaded=False,trainable_parameters=n,base_trainable_parameters=12755009,
         new_fusion_parameters=265729,initialization_weight_private=metadata(c['resnet18_weights_private']),new_module_RNG_preserved=True,
         train_patients=860,Val_patients=189,project_private=str(root),entrypoint_private=str(root/'src_ht/train_ltsa_ht.py')))
