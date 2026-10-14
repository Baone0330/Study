"""Check final documents against unchanged aggregates; no training/metric recomputation."""
import argparse
import re
import pandas as pd
from common import config,setup,jread,jout,metadata,postflight,now
def main(path):
    c=config(path);root=setup(c['project_private']);rows=jread(root/'audit/POSTPROCESS_NUMERICAL_ARTIFACTS_BASELINE_V5.json')
    assert all(metadata(row['path'])==row for row in rows),'Completed weights/predictions/statistics changed'
    assert jread(root/'figures/export_qa/V5_EXPORT_QA.json')['status']=='PASS'
    names=['METHODS','RESULTS','DISCUSSION_LIMITATIONS','TABLE_FIGURE_CAPTIONS','INTEGRATION_WITH_CLIMAT']
    counts={}
    for name in names:
        p=root/'writing'/f'LTSA_V5_{name}_CN_EN.md';text=p.read_text()
        blocks=[b.strip() for b in text.split('\n\n') if b.strip() and not b.lstrip().startswith('#')]
        assert len(blocks)%2==0
        for i in range(0,len(blocks),2):
            assert not re.search('[\u4e00-\u9fff]',blocks[i]) and re.search('[\u4e00-\u9fff]',blocks[i+1])
        counts[name]=len(blocks)//2
    results=(root/'writing/LTSA_V5_RESULTS_CN_EN.md').read_text()
    ens=pd.read_csv(root/'tables/V5_THREE_SEED_ENSEMBLE.csv');seeds=pd.read_csv(root/'tables/V5_SEED_LEVEL_METRICS.csv');boot=pd.read_csv(root/'tables/V5_PAIR_BOOTSTRAP.csv')
    for row in ens.itertuples(index=False):
        for metric in ['ROC_AUC','AP','Accuracy','Precision','Sensitivity','Specificity','F1','threshold']:assert f'{getattr(row,metric):.6f}' in results
    for row in seeds.itertuples(index=False):
        assert f'{row.arm}, seed {row.seed}: best epoch {row.best_epoch}, {row.actual_epochs} completed epochs' in results
        for metric in ['ROC_AUC','AP','F1']:assert f'{getattr(row,metric):.6f}' in results
    for row in boot.itertuples(index=False):
        for metric in ['delta_OP_minus_REAL','ci_low','ci_high']:assert f'{getattr(row,metric):.6f}' in results
    discussion=(root/'writing/LTSA_V5_DISCUSSION_LIMITATIONS_CN_EN.md').read_text()
    assert 'included zero' in discussion and 'adaptive' in discussion and 'Official Test' in discussion
    assert 'binary cross-entropy' in (root/'writing/LTSA_V5_METHODS_CN_EN.md').read_text()
    protection=postflight(root)
    a=dict(status='PASS',numerical_artifacts_checked=len(rows),numerical_metadata_changes=0,bilingual_paragraph_pairs=counts,
           ensemble_seed_and_CI_text_consistency='PASS',claim_scope='exploratory internal Val; intervals include zero; no unique causality or clinical generalization',
           protected_old_source_status=protection['status'],author_review='PENDING',checked_utc=now())
    jout(root/'audit/V5_FINAL_MATERIALS_QA.json',a)
    (root/'audit/V5_FINAL_MATERIALS_QA.md').write_text('# Material evidence review\n\nBoth AUC/AP contrasts are bounded exploratory internal-validation results; both intervals include zero. V3/V4 comparison is descriptive, REAL numerical parity is implementation evidence, and CLIMAT values remain historical references under a different fixedK6 protocol. No new experiment was added. Bilingual pairing and every seed/ensemble/CI number were checked against frozen aggregate tables. The final figure annotations and raw REAL residual skip were actually visually inspected. This is agent QA, not author approval, clinician certification or journal submission certification.\n')
    print(a,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);main(p.parse_args().config)
