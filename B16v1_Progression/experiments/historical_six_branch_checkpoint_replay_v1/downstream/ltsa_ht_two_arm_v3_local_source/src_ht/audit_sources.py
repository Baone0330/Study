"""Snapshot the existing LTSA directory locally; inspect interfaces without running original trainer."""
import argparse
import ast
import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path
import yaml
from common import setup,jread,jout,metadata,postflight,state,now,TASK,ARMS,SEEDS


def main(output_root):
    root=setup(output_root);source=Path('LOCAL_PRIVATE_INPUT_REQUIRED_4');old=Path('LOCAL_PRIVATE_INPUT_REQUIRED_0')
    quality=Path('LOCAL_PRIVATE_INPUT_REQUIRED_1')
    assert source.is_dir();required=['src/models.py','src/transformers.py','src/datasets.py','src/train.py','src/losses.py','src/utils.py','ltsa.yml']
    missing=[n for n in required if not (source/n).is_file()]
    if missing:jout(root/'releases/SOURCE_INCOMPLETE_V3.json',dict(status='LOCAL_LTSA_SOURCE_INCOMPLETE',missing=missing));raise RuntimeError('Required local-source files absent; no automatic network fallback')
    snapshot=root/'upstream/LTSA_local_snapshot';inventory=[metadata(p) for p in sorted(source.rglob('*')) if p.is_file()]
    if not snapshot.exists():shutil.copytree(source,snapshot,copy_function=shutil.copy2,symlinks=True)
    for r in inventory:
        rel=Path(r['path']).relative_to(source);s=(snapshot/rel).stat();assert s.st_size==r['size'] and s.st_mtime_ns==r['mtime_ns']
    assert len([p for p in snapshot.rglob('*') if p.is_file()])==len(inventory)
    raw=(snapshot/'src/models.py').read_text();tree=ast.parse(raw);classes={n.name:n for n in tree.body if isinstance(n,ast.ClassDef)}
    assert {'ImageEncoder','PositionalEncoding','LTSA'}<=set(classes)
    assert 'dim_feedforward=self.encoder.n_features' in raw and "activation='relu'" in raw
    issues=dict(global_args_forward='need_weights=args.attn_map' in raw,standard_transformer_wrong_tuple_signature='feats, attn_map = self.transformer' in raw,
                hardcoded_cuda0="device='cuda:0'" in raw or ".to('cuda:0')" in raw,
                timm_eager_import='import timm' in raw,destructive_original_trainer='shutil.rmtree' in (snapshot/'src/train.py').read_text())
    if (source/'.git').exists():
        env=dict(__import__('os').environ,GIT_OPTIONAL_LOCKS='0')
        rev=subprocess.run(['git','-C',str(source),'rev-parse','HEAD'],env=env,capture_output=True,text=True,check=True).stdout.strip()
        dirty=subprocess.run(['git','-C',str(source),'status','--porcelain'],env=env,capture_output=True,text=True,check=True).stdout.strip();version=dict(revision=rev,working_tree_entries=len(dirty.splitlines()) if dirty else 0)
    else:version=dict(revision='UPSTREAM_GIT_REVISION_UNKNOWN',reason='local extraction has no .git; no online revision lookup')
    versions={}
    for n in ['torch','torchvision','timm','numpy','pandas','pyarrow','Pillow','scikit-learn','pytest','PyYAML','matplotlib']:
        try:versions[n]=importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:versions[n]='NOT_AVAILABLE'
    weights=Path('LOCAL_PRIVATE_INPUT_REQUIRED_2');assert weights.is_file(),'Pretrained cache missing: stop and ask, never switch to random or download'
    import torch
    import torchvision
    expected_url=torchvision.models.ResNet18_Weights.IMAGENET1K_V1.url
    assert expected_url.rsplit('/',1)[-1]==weights.name
    m=torchvision.models.resnet18(weights=None);a=torch.load(weights,map_location='cpu',weights_only=True);m.load_state_dict(a,strict=True)
    assert m.fc.in_features==512 and m.fc.out_features==1000;del m,a
    protection=root/'audit/PROTECTED_SOURCE_METADATA_V3.json'
    if not protection.exists():
        base={r['path'] for r in jread(quality/'source_audit/PROTECTED_SOURCE_METADATA_V1.json')}
        base.update(r['path'] for r in jread(quality/'source_audit/AUXILIARY_PROTECTED_SOURCE_METADATA_V1.json'))
        base.update(r['path'] for r in inventory);base.add(str(weights))
        base.update(str(old/'manifests'/n) for n in ['ORIGINAL_TRAIN_SEQUENCE_V1.csv','ORIGINAL_VAL_SEQUENCE_V1.csv','FROZEN_REAL_HISTORY_RECORDS_V1.json','TRAIN_K6_SOURCE_SLOT_MAP_V1.parquet','MASTER_HISTORICAL_SYNTHETIC_NODES_V1.parquet'])
        jout(protection,[metadata(p) for p in sorted(base)])
    jout(root/'audit/LOCAL_SOURCE_SNAPSHOT_METADATA_V3.json',dict(source_private=str(source),snapshot_private=str(snapshot),files=inventory,comparison='relative file list,size,mtime_ns only; no checksum',version=version))
    jout(root/'audit/ENVIRONMENT_AND_WEIGHTS_V3.json',dict(python=sys.version,versions=versions,cuda_runtime=torch.version.cuda,cuda_available=torch.cuda.is_available(),torch_architectures=torch.cuda.get_arch_list(),
             pretrained_identity='torchvision ResNet18 IMAGENET1K_V1',pretrained_official_URL=expected_url,weights_metadata_private=metadata(weights),strict_offline_pretrained_load='PASS',torch_legacy_ltsa_yml_not_installed=True,
             missing_optional_deps_handling='ResNet18 pathway uses torchvision and exact AST-extracted original TPE; timm-only alternatives disabled. Anonymous regression uses stdlib unittest, not unavailable pytest. No packages installed.'))
    anchors=['# 本地 LTSA 源码与适配锚点\n',f'源：`{source}`；新私有快照：`{snapshot}`。本地复制，无源码网络请求。版本：{version["revision"]}。\n',
             f'实际可见问题：{issues}。\n','原 ImageEncoder ResNet18使用官方ImageNet初始化；图特征512维；原Transformer FFN512/ReLU/post-norm、batch-first。\n',
             '原PositionalEncoding按原类AST从快照加载，避免顶层缺失timm和同名HF transformers；图encoder以torchvision ResNet18(None)+本地官方state_dict严格加载实现同结构且离线。\n',
             '不实例化有cuda:0和错误standard接口的原LTSA；新HT wrapper保留TPE/causalTransformer，新增多view均值、动态mask/末有效token、logit BCE头，无survival/censor/step-ahead。\n']
    for name in ['ImageEncoder','PositionalEncoding','LTSA']:anchors.append(f'- `{source}/src/models.py:{classes[name].lineno}` — {name}, ends line{classes[name].end_lineno}\n')
    (root/'audit/LOCAL_LTSA_SOURCE_ANCHORS_V3.md').write_text('\n'.join(anchors),encoding='utf-8')
    c=dict(task=TASK,project_private=str(root),local_ltsa_source_private=str(source),snapshot_private=str(snapshot),source_old_ablation_private=str(old),quality_private=str(quality),
           raw_data_private='LOCAL_PRIVATE_INPUT_REQUIRED_3',resnet18_weights_private=str(weights),arms=ARMS,seeds=SEEDS,d_model=512,n_heads=4,n_layers=1,dim_feedforward=512,activation='relu',dropout=.25,
           tpe='original_local_fixed_sinusoidal_month_index',step_ahead=False,attn_map=False,loss='BCEWithLogitsLoss_no_class_weight',optimizer='AdamW',lr=1e-4,weight_decay=1e-4,max_epoch=60,warmup_epoch=5,patience=12,min_epoch=15,
           effective_patient_batch=8,patient_microbatch=1,encoder_view_chunk=16,precision='FP32',ImageNet_init='IMAGENET1K_V1_offline_strict',encoder_trainable=True,
           max_history_limit=None,view_policy='all_native_pre_target_year_views',tpe_month_rule='floor(actual_relative_days/30.4375)',selection='ROC_AUC_then_AP_then_lower_loss_then_earlier_epoch',
           threshold='Youden_J_on_Val_and_fixed_0.5_sensitivity',bootstrap_repeats=2000,bootstrap_seed=20261013,proba_reload_atol=1e-6,preliminary_config=True,
           augmentation_policy='NOT_FROZEN_PENDING_SEMANTIC_AUDIT_BEFORE_FORMAL_VAL',no_LTSAsource_network_requests=True,new_SD_sampling=0,new_upstream_training=0,Official_Test_content_access=0)
    cfg=root/'configs/LTSA_HT_TWO_ARM_CONTRACT_V3.yaml'
    if not cfg.exists():cfg.write_text(yaml.safe_dump(c,allow_unicode=True,sort_keys=False),encoding='utf-8')
    postflight(root);state(root,'LOCAL_SOURCE_AUDITED',local_source_files=len(inventory),offline_pretrained='PASS',optional_dependencies_not_installed=['timm','pytest'])
    print(dict(local_source_files=len(inventory),layout='PASS',strict_offline_resnet18='PASS',source_revision=version['revision'],compatibility_findings=issues),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-root',required=True);a=p.parse_args();main(a.output_root)
