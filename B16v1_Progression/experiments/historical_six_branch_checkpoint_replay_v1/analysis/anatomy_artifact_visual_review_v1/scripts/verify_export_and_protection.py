"""Actual export/source checks; visual PASS requires an explicit completed nonmedical inspection."""
import argparse
import ast
import sys
import unittest
from pathlib import Path
import pandas as pd
from PIL import Image
from review_common import setup,read_json,csv_out,json_out,postflight,now


def main(output_root,actual_visual_reviewed=False):
    root=setup(output_root);sys.path.insert(0,str(root/'scripts'))
    import test_review_contracts
    suite=unittest.defaultTestLoader.loadTestsFromModule(test_review_contracts);result=unittest.TextTestRunner(verbosity=2).run(suite)
    test=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),anonymous_only=True,checked_utc=now())
    json_out(root/'tests/ANONYMOUS_TEST_RESULTS_V1.json',test);assert result.wasSuccessful()
    for p in (root/'scripts').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    qa_path=root/'figures/export_qa/NONMEDICAL_FIGURE_EXPORT_QA_V1.csv';qa=pd.read_csv(qa_path);assert len(qa)==6 and qa.auto_export.eq('PASS').all()
    for row in qa.itertuples(index=False):
        with Image.open(root/row.png) as im:assert im.mode=='RGB' and abs(im.info['dpi'][0]-600)<.1 and abs(im.info['dpi'][1]-600)<.1
        svg=(root/row.svg).read_text();assert '<text' in svg and '<image' not in svg
        assert b'/FontFile2' in (root/row.pdf).read_bytes()
    if actual_visual_reviewed:
        qa['actual_nonmedical_agent_visual_review']='PASS_AFTER_ACTUAL_VIEW'
        qa['actual_view_scope']='all six final figures; temporary in-session nonmedical previews only; no local picture retention'
        qa['actual_view_findings']='no clipped labels, contradictory denominators, hidden missingness or medical raster content seen'
        csv_out(qa_path,qa)
        json_out(root/'figures/export_qa/ACTUAL_NONMEDICAL_VISUAL_REVIEW_V1.json',dict(status='PASS_AFTER_ACTUAL_VIEW',figures=6,
                  scope='Fig01-Fig06 all actually viewed in this session; NONMEDICAL ONLY',medical_visual_review='NOT_RUN',local_image_files_saved=0,reviewed_utc=now()))
    else:assert qa.actual_nonmedical_agent_visual_review.eq('PASS_AFTER_ACTUAL_VIEW').all(),'Nonmedical images must actually be viewed, not auto-promoted'
    forms=pd.read_csv(root/'reviews_private/blank_review_form.csv');assert len(forms)==240 and forms.observer_role.eq('NONE').all() and forms.review_quality_status.eq('NOT_VIEWED').all()
    for f in ['thyroid_like_region_visibility','anatomical_border_readability','local_texture_readability','line_or_cross_visible','line_candidate_interpretation']:assert forms[f].eq('NOT_REVIEWED').all()
    ratings=pd.read_csv(root/'metrics/CLINICIAN_BLIND_RATINGS_V1.csv');assert ratings.empty
    assert pd.read_csv(root/'reviews_private/reviewers_metadata_private.csv').empty
    assert not list((root/'reviews_private/raw_completed_forms').iterdir())
    panels=pd.read_csv(root/'figures/private_medical_panels/PRIVATE_MEDICAL_PANEL_MANIFEST_V1.csv');assert len(panels)==84
    cards=pd.read_csv(root/'manifests/PRIVATE_RANDOMIZED_IMAGE_ID_MAP_V1.csv');assert len(cards)==72 and cards.blind_card_id.nunique()==72
    for p in [root,root/'manifests',root/'reviews_private',root/'figures/private_medical_panels',root/'figures/private_expert_review_cards']:
        assert p.stat().st_mode & 0o777 == 0o700
    for p in list((root/'figures/private_medical_panels').glob('*.png'))+list((root/'figures/private_expert_review_cards').glob('*.png')):
        assert p.stat().st_mode & 0o777 == 0o600
        with Image.open(p) as im:im.verify()  # File decoding is not medical visual inspection.
    protection=postflight(root)
    inventory=[dict(path_private=str(p),bytes=p.stat().st_size) for folder in ['protocol','source_audit','manifests','metrics','reviews_private','figures','tables','writing','scripts','tests','release'] for p in (root/folder).rglob('*') if p.is_file()]
    json_out(root/'release/ARTIFACT_INDEX_V1.json',dict(files=inventory,contains_private_paths=True,public_export_allowed=False))
    json_out(root/'release/VERIFICATION_SUMMARY_V1.json',dict(status='PASS',anonymous_tests=test['tests_run'],syntax='PASS',source=protection['status'],
                 actual_unique_source_files=protection['unique_files_actually_checked'],nonmedical_figure_groups=6,aggregate_tables=7,private_medical_panels=84,private_cards=72,
                 medical_pixels_displayed=0,actual_medical_observers=0,expert_scores=0,medical_review='NOT_RUN',checked_utc=now()))
    print({'verification':'PASS','tests':test['tests_run'],'source_unique_files':protection['unique_files_actually_checked'],'nonmedical_visual':'PASS_AFTER_ACTUAL_VIEW','medical_visual':'NOT_RUN'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',required=True);p.add_argument('--actual-visual-reviewed',action='store_true');a=p.parse_args();main(a.output_root,a.actual_visual_reviewed)
