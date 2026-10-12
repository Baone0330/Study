"""Positive file allowlist plus actual private-value checks; no medical binaries are allowed."""
import re
from pathlib import Path
import pandas as pd

SCRIPTS={'review_common.py','freeze_visual_review_cohort.py','localize_artifact_candidates.py','aggregate_nonclinical_review.py',
         'build_private_blind_review_cards.py','verify_phi_display_route.py','aggregate_optional_expert_ratings.py','prepare_paper_materials.py',
         'draw_public_aggregate_figures.py','review_publication_scope.py','build_review_report.py','verify_export_and_protection.py','publish_review_package.py','test_review_contracts.py'}
WRITING={'QUALITATIVE_METHODS_CN_EN_V1.md','QUALITATIVE_RESULTS_CN_EN_V1.md','ANATOMY_ARTIFACT_DISCUSSION_CN_EN_V1.md','CAPTIONS_ALL_CN_EN_V1.md','PAPER_FIGURE_INTEGRATION_V1.md','TERMINOLOGY_LEDGER_V1.csv'}
PROTOCOL={'SCIENTIFIC_CLAIM_BOUNDARIES_V1.md','MEDICAL_IMAGE_PRIVACY_AND_REVIEW_ROUTE_V1.md','PRECOMMITTED_REVIEW_SAMPLING_V1.md',
          'REVIEW_ROLE_AND_BLINDING_V1.md','IMAGE_DISPLAY_ORIGINAL_VS_DERIVATIVE_RULES_V1.md','PRE_DRAW_FIGURE_EVIDENCE_PLAN_V1.json'}
TABLES={f'Table0{i}_{n}.csv' for i,n in enumerate(['Review_Cohort_and_Coverage','Flag_vs_Visual_Category','Region_Artifact_Distribution','CPFM_ON_OFF_Qualitative_Comparison','Sensitivity_and_Unknowns','Expert_Rating_Status','Claim_Evidence_Boundary'],1)}


def allowed_relative(relative):
    p=Path(relative);a=p.parts
    if p.is_absolute() or '..' in a:return False
    if len(a)==1:return p.name in {'README.md','PUBLIC_REVIEW_RESULT_V1.md','requirements-cpu.txt'}
    if len(a)==2:
        if a[0]=='scripts':return p.name in SCRIPTS
        if a[0]=='tables':return p.name in TABLES
        if a[0]=='writing':return p.name in WRITING
        if a[0]=='protocol':return p.name in PROTOCOL
        if a[0]=='tests':return p.name=='ANONYMOUS_TEST_RESULTS_V1.json'
        if a[0]=='review_templates':return p.name in {'BLANK_CLINICIAN_SCALE_HEADER_ONLY_V1.csv','REVIEW_GUIDELINE_CN_EN_V1.md'}
        if a[0]=='release':return p.name=='PUBLIC_ENGINEERING_STATUS_V1.json'
    if len(a)==3 and a[0]=='figures':
        if a[1]=='export_qa':return p.name=='NONMEDICAL_FIGURE_EXPORT_QA_V1.csv'
        if a[1]=='public_nonmedical':return bool(re.fullmatch(r'Fig0[1-6]_[A-Za-z0-9_]+\.(png|svg|pdf)',p.name))
        if a[1]=='source_data_public':return bool(re.fullmatch(r'Fig0[1-6]_[A-Za-z0-9_]+\.csv',p.name))
        if a[1]=='captions_cn_en':return bool(re.fullmatch(r'Fig0[1-6]_[A-Za-z0-9_]+_CN_EN.md',p.name))
    return False


def compile_values(values):
    """Literal prefix trie: same exact boundary rule, without compiling every value per file."""
    trie={}
    for value in set(values):
        if len(value)<5:continue
        n=trie
        for char in value:n=n.setdefault(char,{})
        n['']=None
    if not trie:return None
    def emit(n):
        parts=[re.escape(k)+emit(v) for k,v in sorted(n.items()) if k]
        s=parts[0] if len(parts)==1 else '(?:'+'|'.join(parts)+')' if parts else ''
        return '(?:'+s+')?' if '' in n and parts else s
    return re.compile(r'(?<![A-Za-z0-9])'+emit(trie)+r'(?![A-Za-z0-9])')


def text_issues(text,patient_values=(),clinical_dates=(),private_codes=(),compiled=None):
    issues=[]
    if re.search(r'/(?:data|home)/|[A-Z]:[/\\]Users[/\\]',text):issues.append('PRIVATE_ABSOLUTE_PATH')
    if re.search(r'gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{24,}|sk-[A-Za-z0-9_-]{24,}|-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----',text):issues.append('CREDENTIAL_PATTERN')
    collections=[(patient_values,'PRIVATE_PATIENT_VALUE'),(clinical_dates,'CLINICAL_DATE_VALUE'),(private_codes,'PRIVATE_CASE_CODE')]
    if compiled is not None:
        for regex,kind in compiled:
            if regex is not None and regex.search(text):issues.append(kind)
    else:
        for collection,kind in collections:
            regex=compile_values(collection)
            if regex is not None and regex.search(text):issues.append(kind)
    return issues


def scan(package,private_root=None):
    package=Path(package);patients=[];dates=[];codes=[]
    if private_root:
        r=Path(private_root);q=pd.read_parquet(r/'manifests/PRIVATE_UPSTREAM_IMAGE_QUEUE_V1.parquet',columns=['patient_key_private','source_date_private','anonymous_patient_key','anonymous_node_key','image_id'])
        patients=sorted(set(str(x) for x in q.patient_key_private));dates=sorted({str(x)[:10] for x in q.source_date_private if len(str(x))>=10})
        codes=sorted(set(str(x) for col in ['anonymous_patient_key','anonymous_node_key','image_id'] for x in q[col] if len(str(x))>=5))
        cards=pd.read_csv(r/'manifests/PRIVATE_RANDOMIZED_IMAGE_ID_MAP_V1.csv');codes.extend(cards.blind_card_id.tolist())
    compiled=[(compile_values(patients),'PRIVATE_PATIENT_VALUE'),(compile_values(dates),'CLINICAL_DATE_VALUE'),(compile_values(codes),'PRIVATE_CASE_CODE')]
    files=[];issues=[]
    for p in sorted(package.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(package).as_posix();files.append(rel)
        if not allowed_relative(rel):issues.append(dict(file=rel,reason='NOT_EXPLICITLY_ALLOWLISTED'));continue
        if p.suffix in {'.md','.py','.csv','.json','.txt','.svg'}:
            t=p.read_text(encoding='utf-8')
            for reason in text_issues(t,compiled=compiled):issues.append(dict(file=rel,reason=reason))
            if p.suffix=='.svg' and '<image' in t:issues.append(dict(file=rel,reason='EMBEDDED_RASTER'))
        if p.suffix in {'.png','.pdf'} and not rel.startswith('figures/public_nonmedical/'):
            issues.append(dict(file=rel,reason='UNAPPROVED_BINARY'))
    return dict(status='PASS' if not issues else 'FAIL',file_count=len(files),files=files,issues=issues,medical_images_public=0 if not issues else None,
                safeguards='positive allowlist; actual private patient/date/case values; no medical binary routes; no checksums')
