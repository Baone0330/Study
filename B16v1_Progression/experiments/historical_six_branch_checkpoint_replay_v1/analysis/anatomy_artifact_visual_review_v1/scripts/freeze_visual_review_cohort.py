"""Before reading pixels: source inventory, immutable fixed12, flag-stratified48, blind36."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from review_common import setup,read_json,json_out,csv_out,parquet_out,metadata,postflight,state,unique_pairs,now,TASK


def select_strata(pairs,core_nodes,rng):
    p=pairs.copy();on=p.on_cross.to_numpy(bool);a=p.on_line.to_numpy(bool);b=p.off_line.to_numpy(bool)
    p['stratum']=np.select([on,~on & a & ~b,~on & a & b,~on & ~a & ~b],['S1','S2','S3','S4'],default='UNASSIGNED_OFF_ONLY')
    selected=[];coverage=[];core_patients=set(p[p.anonymous_node_key_on.isin(core_nodes)].anonymous_patient_key_on)
    used=set(core_patients);pre_state=rng.bit_generator.state
    for s in ['S1','S2','S3','S4']:
        all_s=p[p.stratum==s];pool=all_s[~all_s.anonymous_node_key_on.isin(core_nodes)]
        order=rng.permutation(len(pool));chosen=[];available_unused=pool[~pool.anonymous_patient_key_on.isin(used)].anonymous_patient_key_on.nunique()
        for rank,j in enumerate(order):
            row=pool.iloc[int(j)];pid=row.anonymous_patient_key_on
            if pid in used:continue
            used.add(pid);d=row.to_dict();d.update(random_permutation_rank=rank,selection_reason='first eligible globally unused patient in frozen seeded permutation');chosen.append(d)
            if len(chosen)==12:break
        selected.extend(chosen)
        coverage.append(dict(stratum=s,all_queue_pairs=len(all_s),eligible_noncore_pairs=len(pool),eligible_noncore_patients=pool.anonymous_patient_key_on.nunique(),
                             previously_unused_patients=available_unused,target_pairs=12,selected_pairs=len(chosen),selected_patients=len(chosen),
                             shortfall=12-len(chosen),nominal_pair_allocation_fraction=len(chosen)/len(pool) if len(pool) else None,
                             inclusion_probability='NOT_ESTABLISHED_NONUNIFORM_DISTINCT_PATIENT_PRIORITY',clinical_prevalence_estimation='NOT_ALLOWED_WITHOUT_VALID_WEIGHTING'))
    return pd.DataFrame(selected),pd.DataFrame(coverage),p,dict(before=pre_state,after=rng.bit_generator.state)


def main(output_root,quality_root):
    root=setup(output_root);old=Path(quality_root).resolve();dest=root/'protocol/FROZEN_VISUAL_AND_ARTIFACT_REVIEW_CONTRACT_V1.json'
    if dest.exists():
        assert read_json(dest)['quality_root_private']==str(old);postflight(root);print('Frozen same-source cohort retained');return
    s=read_json(old/'release/FINAL_QUALITY_EVALUATION_STATUS_V1.json');assert s['FINAL_STATUS']=='QUALITY_EVALUATION_COMPLETED_WITH_LIMITATIONS' and s['git_publication_status']=='PUSH_VERIFIED'
    assert s['git_commit']=='5cdb25688fc702da057ce4821db5d15e5c1f991e'
    qpath=old/'manifests/COMBINED_QUALITY_EVAL_FROZEN_QUEUE_V1.parquet';qa_path=old/'metrics/IMAGE_DECODE_AND_TECHNICAL_QA_V1.parquet'
    pair_path=old/'manifests/OP_ON_OFF_EXACT_IMAGE_PAIRS_V1.parquet';node_path=old/'manifests/PRECOMMITTED_VISUAL_PANEL_NODES_V1.csv'
    q=pd.read_parquet(qpath);qa=pd.read_parquet(qa_path);pairs=unique_pairs(q,pd.read_parquet(pair_path));nodes=pd.read_csv(node_path)
    assert len(q)==len(qa)==50280 and len(nodes)==12 and nodes.anonymous_node_key.nunique()==12
    assert qa.image_id.tolist()==q.image_id.tolist() and qa.decode_status.eq('PASS').all()
    flags=qa[['image_id','long_line_flag','cross_proxy_flag','long_line_count']]
    for side in ['on','off']:
        pairs=pairs.merge(flags,left_on='image_id_'+side,right_on='image_id',validate='1:1').drop(columns='image_id')
        pairs=pairs.rename(columns={'long_line_flag':side+'_line','cross_proxy_flag':side+'_cross','long_line_count':side+'_line_count'})
    core=pairs[pairs.anonymous_node_key_on.isin(nodes.anonymous_node_key)].copy()
    core=core.merge(nodes[['panel_node_code','anonymous_node_key','display_section']].rename(columns={'anonymous_node_key':'anonymous_node_key_on'}),on='anonymous_node_key_on',how='left')
    core=core.sort_values(['panel_node_code','image_index']);assert len(core)==72
    rng=np.random.default_rng(20261012);sample,coverage,allpairs,rngstate=select_strata(pairs,set(nodes.anonymous_node_key),rng)
    blind=core[core.image_index.isin([0,2,5])].copy();assert len(blind)==36
    assert not set(sample.image_id_on)&set(core.image_id_on)
    for name,d in [('ORIGINAL_FIXED12_NODE_LIST_V1.csv',nodes)]:csv_out(root/'manifests'/name,d)
    for name,d in [('CORE_ON_OFF_PAIRED_PANEL72_V1.parquet',core),('STRATIFIED_ARTIFACT_REVIEW48_V1.parquet',sample),
                   ('ALL9270_PAIRS_AND_FLAG_REFERENCES_V1.parquet',allpairs),('CLINICIAN_BLIND_SET_V1.parquet',blind)]:parquet_out(root/'manifests'/name,d)
    private_queue=q.copy();parquet_out(root/'manifests/PRIVATE_UPSTREAM_IMAGE_QUEUE_V1.parquet',private_queue)
    csv_out(root/'metrics/REVIEW_SAMPLE_COVERAGE_AND_SELECTION_V1.csv',coverage)
    json_out(root/'manifests/PRIVATE_SAMPLING_RNG_STATES_V1.json',rngstate)
    counts=[]
    for b,d in q.groupby('branch',sort=False):counts.append(dict(branch=b,existing_images=len(d),existing_patients=d.anonymous_patient_key.nunique(),
                   existing_nodes=d[d.anonymous_node_key!=''].anonymous_node_key.nunique(),existing_native_exams=d[d.real_exam_key_private!=''].real_exam_key_private.nunique(),source='PRIOR_COMPLETED_QA_REFERENCED_NOT_NEW_DECODING'))
    csv_out(root/'source_audit/EXISTING_BRANCH_COUNTS_V1.csv',counts)
    inventory=[]
    for role,path,n in [('image_queue',qpath,len(q)),('per_image_flag_QA',qa_path,len(qa)),('exact_ON_OFF_pairs',pair_path,len(pairs)),('fixed12_nodes',node_path,len(nodes))]:
        inventory.append(dict(logical_role=role,source_location_server_private=str(path),format=path.suffix,item_count=n,access_status='READONLY_VERIFIED',remark='no patient/path detail in stdout'))
    csv_out(root/'source_audit/QUALITY_V1_ASSET_INVENTORY_V1.csv',inventory)
    csv_out(root/'manifests/SOURCE_DEPENDENCY_INDEX_V1.csv',inventory)
    (root/'source_audit/IMAGE_AND_FLAG_SCHEMA_V1.md').write_text('# Actual upstream per-image schema\n\n'+'\n'.join('* '+c+': '+str(t) for c,t in zip(qa.columns,qa.dtypes))+'\n',encoding='utf-8')
    old_cfg=read_json(old/'protocol/FROZEN_QUALITY_EVALUATION_CONTRACT_V1.json');abl=Path(old_cfg['source_root']);identity=read_json(abl/'protocol/FROZEN_HISTORICAL_CPFM_ABLATION_CONFIG_V1.json')
    assert identity['reference_model']=='Original Progression 20260921 epoch120, not S2/Clean'
    (root/'source_audit/ORIGINAL_CHECKPOINT_AND_BRANCH_IDENTITY_V1.md').write_text('# Frozen branch identity\n\nHistorical Original Progression20260921 epoch120 and original B14 CPFM; old F16/B14 frozen, ON/OFF exact seed/index/node. OFF zeros the sole active Progression-to-UNet injection path. Full and Plain are historical auxiliary routes, not isolated module effects. Only36 prior fixed pilot ON resamples had proven pixel parity. No upstream/downstream training or sampling in this task.\n',encoding='utf-8')
    anchors=[dict(evidence='old_long_line_and_cross_definition',source_private=str(old/'scripts/check_decode_and_image_stats.py'),source_symbol='gray/texture/worker'),
             dict(evidence='old_distance_results_referenced_not_recomputed',source_private=str(old/'metrics/QUALITY_DISTRIBUTION_METRICS_V1.csv'),source_symbol='measured CSV'),
             dict(evidence='old_pair_cluster_KID_referenced_not_recomputed',source_private=str(old/'metrics/QUALITY_PATIENT_CLUSTER_BOOTSTRAP_V1.csv'),source_symbol='paired patient interval')]
    csv_out(root/'source_audit/PRIOR_METRIC_SOURCE_ANCHORS_V1.csv',anchors)
    # Recover the prior protection dependency set, then actually take this task's baseline.
    paths={r['path'] for r in read_json(old/'source_audit/PROTECTED_SOURCE_METADATA_V1.json')}
    paths.update(r['path'] for r in read_json(old/'source_audit/AUXILIARY_PROTECTED_SOURCE_METADATA_V1.json'))
    enc=read_json(old/'image_features/encoder_contract.json');paths.update(enc[k]['path'] for k in ['inception_weight_metadata','alexnet_weight_metadata','lpips_calibration_metadata'])
    for folder in ['protocol','source_audit','manifests','metrics','image_features','figures','writing','scripts','release']:
        paths.update(str(p) for p in (old/folder).rglob('*') if p.is_file())
    paths.add(str(old/'B16V1_HISTORICAL_ORIGINAL_PROGRESSION_GENERATION_QUALITY_RESULT_V1.md'))
    # Existing prior medical panel count is an inventory fact, not a new render count.
    previous_panels=pd.read_csv(old/'figures/private_medical_panels/PRIVATE_MEDICAL_PANEL_MANIFEST_V1.csv');assert len(previous_panels)==36
    json_out(root/'source_audit/PROTECTED_SOURCE_METADATA_THIS_TASK_V1.json',[metadata(p) for p in sorted(paths)])
    contract=dict(task=TASK,quality_root_private=str(old),started_utc=now(),cohorts_frozen_before_new_pixel_read=True,
                  prior_QA_images_referenced=50280,prior_medical_node_panels_referenced=36,primary_pairs=9270,primary_nodes=1545,primary_patients=860,
                  real_reference_views=13200,real_reference_patients=859,core_pairs=len(core),core_nodes=core.anonymous_node_key_on.nunique(),
                  core_patients=core.anonymous_patient_key_on.nunique(),stratified_pairs=len(sample),stratified_patients=sample.anonymous_patient_key_on.nunique(),
                  clinician_candidate_pairs=len(blind),clinician_candidate_images=2*len(blind),clinician_candidate_patients=blind.anonymous_patient_key_on.nunique(),
                  sample_core_pair_overlap=0,sample_core_patient_overlap=len(set(sample.anonymous_patient_key_on)&set(core.anonymous_patient_key_on)),
                  unassigned_full_queue_pairs=int(allpairs.stratum.eq('UNASSIGNED_OFF_ONLY').sum()),stratification_seed=20261012,
                  sampling_method='stable patient/node/slot/index/seed order; one seeded permutation per stratum; exclude core nodes; one globally unused patient per selected pair; no quota-filling duplicates',
                  population_prevalence_from48_allowed=False,localizer_version='LOCALIZATION_DIAGNOSTIC_V1',
                  original_flag_definition={'gray':'0.299R+0.587G+0.114B float64; round/clip uint8 for Canny','resize':'whole-image PIL bilinear512',
                     'canny':[50,150],'hough_rho':1,'hough_theta':'pi/180','hough_votes':50,'hough_min_length':128,'hough_max_gap':10,'orientation_tolerance_deg':10,
                     'cross':'horizontal/vertical average-coordinate intersection within both projected segment extents'},
                  regions={'coordinates':'common512 pixel grid, not mm or thyroid ROI','border':'outer26 pixels (ceil5 percent),[26:486) interior',
                           'central':'[102:410) x [102:410), approximately central60 percent','intermediate':'remaining pixels','multizone':'candidate spans more than one geometric zone'},
                  line_location_sampling='segment points at <=1px spacing, rounded to pixel coordinates; cross zones union both supporting segments; crossing-point zone saved separately',
                  display_center_crop=[192,320,192,320],difference_map_range=[0,255],doctor_image_indices=[0,2,5],
                  localizer_branches=['OP_CPFM_ON','OP_CPFM_OFF','REAL_HISTORY_REFERENCE'],decode_workers=6,max_resident_memory_plan_gib=8,
                  bootstrap={'unit':'patient','repeats':2000,'seed':20261012,'direction':'ON minus OFF','paired_same_patient_weights':True},
                  public_figure_contract={'groups':6,'backend':'Python Matplotlib','PNG':'RGB600dpi','SVG':'editable text','PDF':'embedded TrueType','min_font_pt':8},
                  actual_observers=0,actual_clinicians=0,medical_visual_review='NOT_RUN',real_visual_reference='WITHHELD',phi_review='PENDING',
                  medical_image_transfer=False,new_SD_sampling=0,new_training=0,new_FID_KID=0,new_HT_or_Test_evaluation=0)
    json_out(dest,contract)
    notes={
      'SCIENTIFIC_CLAIM_BOUNDARIES_V1.md':'Technical line/cross candidates and geometric positions are not anatomical ROI, clinical artifact labels or target-year truth. Natural-image distances are cited from the prior phase, not recomputed. No observations or clinician ratings are generated.',
      'MEDICAL_IMAGE_PRIVACY_AND_REVIEW_ROUTE_V1.md':'Generated medical pixels stay private on the server. REAL reference pixels remain WITHHELD until a genuine approved de-identification/local viewing route exists. No remote-model or chat image transfer. Private cards are preparation, not approval or completed review.',
      'PRECOMMITTED_REVIEW_SAMPLING_V1.md':'Inherited fixed12 nodes, all6 indices, first6 main/next6 supplementary; flag strata sampled once with seed20261012 and distinct-patient priority. Expert candidate subset indices0,2,5. Sampling fractions are not calibrated inclusion weights; do not generalize48-case observations to population.',
      'REVIEW_ROLE_AND_BLINDING_V1.md':'NONE/MACHINE_PROXY/NONCLINICAL_HUMAN/CLINICIAN/CODEX_VISUAL remain separate. No role is promoted automatically. Expert templates remain blank; raw forms empty. Hidden branch names do not prove observers cannot infer the route.',
      'IMAGE_DISPLAY_ORIGINAL_VS_DERIVATIVE_RULES_V1.md':'Original RGB512 never overwritten. Originals and position-overlay derivatives are paired. Difference range0..255 and center crop192:320 are fixed; no sharpening, histogram adjustment or invented physical-mm scale. All overlays are technical candidates.'}
    for name,text in notes.items():(root/'protocol'/name).write_text('# '+name.replace('_',' ')+'\n\n'+text+'\n',encoding='utf-8')
    csv_out(root/'writing/TERMINOLOGY_LEDGER_V1.csv',[
      dict(term_en='technical line/cross candidate',term_zh='技术直线/十字候选',boundary='not clinically confirmed artifact'),
      dict(term_en='geometric central field',term_zh='几何中心区域',boundary='not thyroid ROI'),
      dict(term_en='NOT_REVIEWED',term_zh='尚未实际观察',boundary='distinct from observed UNCERTAIN'),
      dict(term_en='clinician subjective rating',term_zh='医生主观评分',boundary='only actual qualified approved observers; not certification')])
    postflight(root);state(root,'SOURCE_AND_REVIEW_COHORTS_FROZEN',core_pairs=len(core),stratified_pairs=len(sample),blind_pairs=len(blind))
    print({k:contract[k] for k in ['core_pairs','core_nodes','core_patients','stratified_pairs','stratified_patients','clinician_candidate_pairs','clinician_candidate_images','unassigned_full_queue_pairs']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',required=True);p.add_argument('--quality-root',required=True);a=p.parse_args();main(a.output_root,a.quality_root)
