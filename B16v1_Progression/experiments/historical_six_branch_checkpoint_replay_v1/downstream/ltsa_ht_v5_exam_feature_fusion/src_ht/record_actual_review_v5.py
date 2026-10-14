"""Only invoke after genuinely viewing every final nonmedical figure."""
import argparse
from common import config,setup,jread,jout,metadata,now,RESULT
NAMES=['Fig01_V5_ROC_PR','Fig02_V3_V4_V5_Descriptive_Comparison','Fig03_V5_Seeds_and_Ensemble','Fig04_V5_Exam_Fusion_Internal_Architecture']
def main(path,notes):
    c=config(path);root=setup(c['project_private']);final=jread(root/'releases/FINAL_V5_STATUS.json')
    assert final['RUNS_AND_METRICS_COMPLETE'] and jread(root/'figures/export_qa/V5_EXPORT_QA.json')['status']=='PASS'
    review=jread(notes);assert review['actual_images_viewed'] is True and set(review['figure_notes'])==set(NAMES)
    assert all(isinstance(v,str) and len(v)>=30 for v in review['figure_notes'].values())
    assert review['labels_legible'] and review['no_overlap'] and review['numeric_and_architecture_content_checked']
    a=dict(status='PASS_AGENT_ACTUAL_VIEW',reviewed_figure_groups=4,clinical_images=0,clinical_expert_review='NOT_RUN',review_notes=review,
        image_metadata_private=[metadata(root/'figures/nonmedical'/f'{n}.png') for n in NAMES],reviewed_utc=now())
    # Public review notes exclude absolute path metadata.
    jout(root/'audit/V5_ACTUAL_VISUAL_REVIEW_PRIVATE.json',a)
    del a['image_metadata_private'];jout(root/'figures/export_qa/V5_ACTUAL_VISUAL_REVIEW.json',a)
    assert len(list((root/'writing').glob('LTSA_V5_*_CN_EN.md')))==5
    assert jread(root/'audit/V5_FINAL_MATERIALS_QA.json')['status']=='PASS'
    result=(root/RESULT).read_text();assert '6/6' in result and result.count('## ')>=13
    final.update(DELIVERABLES_COMPLETE=True,visual_review='PASS_AGENT_ACTUAL_VIEW',deliverables_completed_utc=now());jout(root/'releases/FINAL_V5_STATUS.json',final)
    print(dict(DELIVERABLES_COMPLETE=True,GITHUB_PUSH_VERIFIED=final['GITHUB_PUSH_VERIFIED'],reviewed_groups=4))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--actual-review-notes',required=True);a=p.parse_args();main(a.config,a.actual_review_notes)
