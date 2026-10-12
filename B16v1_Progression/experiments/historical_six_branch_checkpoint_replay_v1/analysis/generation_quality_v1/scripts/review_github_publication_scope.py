"""Explicit public allowlist and private-value scan; binary medical assets never allowed."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

SCRIPTS={'b16_quality_common.py','inspect_frozen_generation_assets.py','prepare_quality_models.py','freeze_quality_extensions.py',
         'check_decode_and_image_stats.py','extract_quality_features.py','calculate_paired_image_changes.py','calculate_fid_kid.py',
         'aggregate_patient_cluster_metrics.py','draw_public_quality_figures.py','draw_private_medical_panels.py',
         'prepare_blinded_expert_review_template.py','review_github_publication_scope.py','run_measurement_pipeline.py',
         'test_quality_contracts.py','build_quality_report.py'}
WRITING={'IMAGE_QUALITY_METHODS_CN_EN_V1.md','IMAGE_QUALITY_RESULTS_CN_EN_V1.md','IMAGE_QUALITY_DISCUSSION_AND_LIMITATIONS_CN_EN_V1.md',
         'FIGURES_AND_TABLES_CAPTIONS_CN_EN_V1.md','PAPER_EXPERIMENT_INTEGRATION_V1.md','TERMINOLOGY_LEDGER_V1.csv'}
PROTOCOL={'REAL_REFERENCE_SELECTION_POLICY_V1.md','QUALITY_METRIC_DEFINITIONS_V1.md','PRECOMMITTED_VISUAL_SAMPLE_SELECTION_V1.md',
          'COMPARISON_INTERPRETATION_RULES_V1.md'}


def allowed_relative(relative):
    p=Path(relative);parts=p.parts
    if p.is_absolute() or '..' in parts:return False
    if len(parts)==1:return p.name in {'README.md','PUBLIC_QUALITY_RESULT_V1.md','requirements-evaluation.txt'}
    if parts[0]=='scripts':return len(parts)==2 and p.name in SCRIPTS
    if parts[0]=='tests':return len(parts)==2 and p.name in {'test_quality_contracts.py','FID_KID_NUMERICAL_TOY_V1.json','ANONYMOUS_TEST_RESULTS_V1.json'}
    if parts[0]=='writing':return len(parts)==2 and p.name in WRITING
    if parts[0]=='protocol':return len(parts)==2 and p.name in PROTOCOL
    if parts[0]=='tables':return len(parts)==2 and bool(re.fullmatch(r'Table(?:0[1-9]|1[0-2])_[A-Za-z0-9_]+_V1.csv',p.name))
    if parts[0]=='figures' and len(parts)==3:
        if parts[1] in {'nonmedical','source_data','captions'}:
            return bool(re.fullmatch(r'Fig0[1-8]_[A-Za-z0-9_]+(?:_CN_EN)?\.(?:png|svg|pdf|csv|md)',p.name))
        if parts[1]=='export_qa':return p.name=='NONMEDICAL_FIGURE_EXPORT_QA_V1.csv'
    if parts[0]=='release':return len(parts)==2 and p.name in {'PUBLIC_FINAL_STATUS_V1.json','NEXT_STAGE_PAPER_HANDOFF_V1.md'}
    return False


def scan(package,private_root=None):
    package=Path(package);issues=[];files=[]
    patient_values=[];dates=[]
    if private_root:
        r=Path(private_root)
        patient_values=list(json.loads((r/'manifests/PRIVATE_ANONYMOUS_PATIENT_MAP_V1.json').read_text()))
        import pandas as pd
        q=pd.read_parquet(r/'manifests/COMBINED_QUALITY_EVAL_FROZEN_QUEUE_V1.parquet')
        dates=sorted({str(d)[:10] for d in q.source_date_private if len(str(d))>=10})
    for p in sorted(package.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(package).as_posix();files.append(rel)
        if not allowed_relative(rel):issues.append(dict(file=rel,reason='NOT_IN_EXPLICIT_PUBLIC_ALLOWLIST'));continue
        if p.suffix in {'.md','.py','.csv','.json','.txt','.svg'}:
            text=p.read_text(encoding='utf-8')
            if re.search(r'/(?:data|home)/|[A-Z]:[/\\]Users[/\\]',text):issues.append(dict(file=rel,reason='PRIVATE_ABSOLUTE_PATH'))
            if re.search(r'gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{24,}|sk-[A-Za-z0-9_-]{24,}|-----BEGIN (?:OPENSSH|RSA|EC) PRIVATE KEY-----',text):
                issues.append(dict(file=rel,reason='CREDENTIAL_OR_PRIVATE_KEY_PATTERN'))
            for value in patient_values:
                if len(value)>=5 and re.search(r'(?<![A-Za-z0-9])'+re.escape(value)+r'(?![A-Za-z0-9])',text):
                    issues.append(dict(file=rel,reason='PRIVATE_PATIENT_IDENTIFIER_VALUE'));break
            for date in dates:
                if date in text:issues.append(dict(file=rel,reason='PRIVATE_CLINICAL_DATE_VALUE'));break
            if p.suffix=='.svg' and '<image' in text:issues.append(dict(file=rel,reason='EMBEDDED_RASTER_NOT_ALLOWED_IN_AGGREGATE_SVG'))
        if parts_contains_private(rel):issues.append(dict(file=rel,reason='PRIVATE_ROLE_FILE'))
    return dict(status='PASS' if not issues else 'FAIL',file_count=len(files),issues=issues,files=files,
                metadata_only_public=True,clinical_images_public=0 if not issues else None,no_checksums=True)


def parts_contains_private(relative):
    return any(x in relative.split('/') for x in ['manifests','image_features','private_medical_panels','models','vendor','cache','logs'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',required=True);p.add_argument('--private-root');a=p.parse_args()
    result=scan(a.package,a.private_root);print(json.dumps({k:v for k,v in result.items() if k!='files'},ensure_ascii=False));raise SystemExit(0 if result['status']=='PASS' else 1)
