"""Freeze one bounded, explicitly synthetic token policy before Val; never fabricate a REAL Exam."""
import argparse
import json
import shutil
from datetime import date,datetime
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import yaml
from common import setup,config,jread,jout,cout,metadata,postflight,now,state


def eligibility(first,last,query,target_year):
    if query.year>=int(target_year):return 'EXCLUDE_TARGET_YEAR_SYNTH_PREDICTION'
    if query>=last:return 'EXCLUDE_UNKNOWN_FUTURE_AFTER_LAST_REAL'
    if query<=first:return 'EXCLUDE_NO_REAL_HISTORY_BEFORE_QUERY'
    return 'ELIGIBLE_BOUNDED_SYNTHETIC_QUERY_TOKEN'


def main(config_path,verify_only=False):
    c=config(config_path);root=setup(c['project_private']);status_path=root/'releases/ORIGINAL_PROGRESSION_LINEAGE_STATUS_V3.json'
    if status_path.exists():
        prior=jread(status_path);assert prior['status']=='PASS_MAPPING_FROZEN_BEFORE_FORMAL_VAL';postflight(root)
        mapping=jread(root/'manifests_private/OP_BOUNDED_SYNTHETIC_TOKEN_MAP_V3.json');assert len(mapping)==prior['eligible_synthetic_nodes']
        assert all(len(r['synthetic_cache_private'])==6 and all(Path(p).is_file() for p in r['synthetic_cache_private']) for r in mapping)
        print({'lineage_reverification':'PASS','bounded_synthetic_nodes':len(mapping),'mapping_reselected':False});return
    assert not verify_only,'Frozen pre-Val mapping missing'
    assert not list((root/'metrics/private').glob('*best_saved.csv')) and not list((root/'checkpoints').rglob('best.pt')),'No new mapping may be selected after formal Val'
    old=Path(c['source_old_ablation_private']);nodes=pd.read_parquet(old/'manifests/MASTER_HISTORICAL_SYNTHETIC_NODES_V1.parquet')
    records=jread(old/'manifests/FROZEN_REAL_HISTORY_RECORDS_V1.json');master=pd.read_parquet(root/'manifests_private/HT_LTSA_CANONICAL_PATIENT_WINDOWS_V1.parquet');train=master[master.split.eq('train')].set_index('patient_key_private')
    formal_path=Path('LOCAL_PRIVATE_INPUT_REQUIRED_0')
    formal=pd.read_csv(formal_path,dtype={'patient_id':str});assert len(nodes)==1545 and len(formal)==9270 and formal.split.eq('train').all()
    assert formal.ablation_variant.eq('progression').all() and formal.progression_active.eq(1).all()
    for flag in ['target_year_read','target_report_read','target_label_used','future_exam_read','synthetic_history_used']:assert formal[flag].eq(0).all()
    generator=Path('LOCAL_PRIVATE_INPUT_REQUIRED_1');text=generator.read_text()
    assert 'base.parse_date(x["exam_date"])<qdate' in text and 'history_reports":[]' in text and 'mask=torch.zeros((1,1))' in text
    assert 'before_repr[None],after_repr[None],temporal[None],deltas,mask' in text
    ledger=[];mapping=[];exact_real_query=0
    for z in nodes.itertuples(index=False):
        pid=str(z.patient_id);assert pid in train.index;r=train.loc[pid];source=records[pid]
        assert int(z.target_year)==int(r.target_year)==int(source['target_year'])
        native=json.loads(r.exams_private_json);first=date.fromisoformat(r.first_history_exam_date);last=date.fromisoformat(r.last_history_exam_date);q=date.fromisoformat(z.query_date)
        assert source['history'][0]['exam_date']==first.isoformat()
        assert q.year==first.year+int(z.annual_slot_index) and (q.month,q.day)==(first.month,first.day),'Annual query must be derived from observed first-history anchor, not actual target date'
        # Actual conditions only use REAL history strictly before q; no target-year content or reports.
        prior=[e for e in source['history'] if date.fromisoformat(e['exam_date'])<q];assert prior
        assert all(date.fromisoformat(e['exam_date']).year<int(r.target_year) for e in prior)
        native_by_date={e['date']:set(e['paths']) for e in native}
        for e in prior:assert e['exam_date'] in native_by_date and set(e['image_paths'])==native_by_date[e['exam_date']]
        f=formal[(formal.patient_id==pid)&(formal.annual_slot_index==z.annual_slot_index)].sort_values('image_index')
        assert len(f)==6 and f.image_index.tolist()==list(range(6)) and f.seed.tolist()==list(z.seeds) and f.image_path.tolist()==list(z.reference_on_paths)
        assert all(Path(p).is_file() for p in z.reference_on_paths)
        role=eligibility(first,last,q,r.target_year);exact_real_query+=int(q.isoformat() in native_by_date)
        ledger.append(dict(patient_index=int(r.patient_index),patient_key_private=pid,node_key_private=z.node_id,query_date_private=q.isoformat(),annual_slot_index=int(z.annual_slot_index),
                           target_year_metadata_only=int(r.target_year),condition_real_exam_count=len(prior),condition_identity='original_strict_prequery_REAL_no_reports',source_original_CPfm_ON=True,eligibility=role,
                           images=6,seeds_private=list(z.seeds),source_paths_private=list(z.reference_on_paths),query_time_origin='first_observed_REAL_anchor_plus_frozen_annual_index'))
        if role=='ELIGIBLE_BOUNDED_SYNTHETIC_QUERY_TOKEN':
            caches=[]
            for i,p in enumerate(z.reference_on_paths):
                dest=root/'cache/synthetic224'/f'synthetic_{len(mapping):04d}_{i}.npy'
                with Image.open(p) as im:im.load();a=np.asarray(im.convert('RGB').resize((224,224),Image.Resampling.BILINEAR)).copy()
                assert a.shape==(224,224,3);temp=dest.with_suffix('.tmp.npy');np.save(temp,a);temp.replace(dest);caches.append(str(dest))
            mapping.append(dict(patient_index=int(r.patient_index),query_date_private=q.isoformat(),synthetic_cache_private=caches,
                         source_type='SYNTH_BOUNDED_ANNUAL_QUERY_NOT_REAL_EXAM',node_key_private=z.node_id,source_seeds_private=list(z.seeds),original_reference_paths_private=list(z.reference_on_paths)))
    counts=pd.DataFrame(ledger).groupby('eligibility').size().to_dict();assert counts=={'ELIGIBLE_BOUNDED_SYNTHETIC_QUERY_TOKEN':686,'EXCLUDE_TARGET_YEAR_SYNTH_PREDICTION':758,'EXCLUDE_UNKNOWN_FUTURE_AFTER_LAST_REAL':101}
    assert exact_real_query==0 and len(mapping)==686
    jout(root/'manifests_private/OP_FULL_SOURCE_LINEAGE_AND_ELIGIBILITY_V3.json',ledger);jout(root/'manifests_private/OP_BOUNDED_SYNTHETIC_TOKEN_MAP_V3.json',mapping)
    cout(root/'tables/OP_CANDIDATE_MAPPING_COUNTS_V3.csv',[dict(category=k,nodes=v,images=6*v) for k,v in counts.items()])
    protection=jread(root/'audit/PROTECTED_SOURCE_METADATA_V3.json');have={x['path'] for x in protection}
    for p in [str(formal_path),str(generator)]:
        if p not in have:protection.append(metadata(p));have.add(p)
    jout(root/'audit/PROTECTED_SOURCE_METADATA_V3.json',protection)
    c['augmentation_policy']='bounded synthetic annual-query tokens, strict between first and last observed REAL history, anchored to first REAL date; no true-Exam equivalence claim'
    c['augmentation_policy_reason']='authoritative frozen queries equal observed first-history anniversary+annual index; all conditions strict prequery REAL;758target-year+101after-lastREAL excluded before Val'
    c['augmentation_eligible_nodes']=686;c['augmentation_eligible_views']=4116;c['augmentation_patients_with_synthetic']=597;c['augmentation_patients_without_synthetic']=263
    c['augmentation_mapping_frozen_utc']=now();c['last_prediction_token']='latest REAL historical Exam in both arms, including augmented timelines'
    backup=Path(config_path).with_name(Path(config_path).name+'.bak_'+datetime.utcnow().strftime('%Y%m%dT%H%M%SZ'));shutil.copy2(config_path,backup)
    Path(config_path).write_text(yaml.safe_dump(c,sort_keys=False,allow_unicode=True),encoding='utf-8')
    status=dict(status='PASS_MAPPING_FROZEN_BEFORE_FORMAL_VAL',mapping='BOUNDED_SYNTHETIC_QUERY_TOKEN',candidate_nodes=1545,candidate_images=9270,eligible_synthetic_nodes=686,eligible_synthetic_images=4116,
                excluded_target_year_nodes=758,excluded_after_last_REAL_nodes=101,patients_with_synthetic=597,patients_without_synthetic=263,same_train_patients=860,same_val_patients=189,
                synthetic_date_from_target_reference=False,synthetic_tokens_are_REAL_exams=False,latest_token_is_REAL=True,Formal_Val_metrics_observed_before_choice=0,source_query_matches_actual_REAL_dates=0,frozen_utc=now())
    jout(status_path,status)
    (root/'audit/ORIGINAL_PROGRESSION_LINEAGE_AND_TOKEN_MAPPING_V3.md').write_text('# 唯一增强映射\n\n完整候选1545节点/9270图来自原历史Original Progression CPFM ON正式清单；source/samepatient/index/sixseed/path相同，条件真实检查严格早于query且早于target_year，无报告/target标签/未来真值。\n\n源查询全部等于首个真实历史检查的月日及该年+冻结annual index，不读取目标真实日期间隔。没有一个query与实际历史Exam同日，所以不把合成图冒充同次实际检查。\n\n在正式Val前冻结唯一合法synthetic-token映射：686节点/4116图，其query严格位于首末真实历史Exam之间。每节点六图先均值为一个标明SYNTH的虚拟查询token，按冻结查询时间排序。保留全部真实Exam骨架、同Train860/Val189、同模型/初始化/优化/选模；最终token仍是最后真实历史Exam。\n\n排除758个target-year预测节点和101个最后真实之后的未知将来预测节点；不硬塞、不新生成、不从Val挑映射或提高AUC。597患者有合成内容，263无合成内容仍参与同一监督人群；这是协议限定复用，不是全部9270图或原CLIMAT固定K6协议的直接复制。\n',encoding='utf-8')
    postflight(root);print({k:v for k,v in status.items() if k not in ['frozen_utc']},flush=True);state(root,'AUGMENTATION_MAPPING_PRE_VAL_FROZEN',eligible_nodes=686,excluded_nodes=859)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--verify-only',action='store_true');a=p.parse_args();main(a.config,a.verify_only)
