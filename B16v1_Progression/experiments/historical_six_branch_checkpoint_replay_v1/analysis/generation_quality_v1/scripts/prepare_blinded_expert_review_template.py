"""Empty physician rating template; fixed selection and hidden randomized source map."""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
from b16_quality_common import setup, write_csv, atomic_json


def main(output_root):
    root=setup(output_root);dest=root/'figures/private_medical_panels';os.chmod(dest,0o700)
    q=pd.read_parquet(root/'manifests/COMBINED_QUALITY_EVAL_FROZEN_QUEUE_V1.parquet')
    n=pd.read_csv(root/'manifests/PRECOMMITTED_VISUAL_PANEL_NODES_V1.csv')
    s=q[q.anonymous_node_key.isin(n.anonymous_node_key)].copy()
    # Stable first native view for each precommitted patient, with no image-content or outcome selection.
    patients=s.anonymous_patient_key.drop_duplicates()
    real=q[(q.branch=='REAL_HISTORY_REFERENCE') & q.anonymous_patient_key.isin(patients)].sort_values('image_id').drop_duplicates('anonymous_patient_key')
    selected=pd.concat([s,real],ignore_index=True);rng=np.random.default_rng(20261012)
    shuffled=selected.iloc[rng.permutation(len(selected))].reset_index(drop=True);mapping=[];template=[]
    scorecols=['thyroid_structure_visibility_1to5','organ_boundary_plausibility_1to5','echo_texture_plausibility_1to5',
               'artifact_severity_1to5','overall_realism_1to5','reviewer_confidence_1to5']
    for i,r in enumerate(shuffled.itertuples()):
        code=f'Q{i+1:04d}'
        mapping.append(dict(blind_image_code=code,branch_private=r.branch,anonymous_patient_key_private=r.anonymous_patient_key,
                            anonymous_node_key_private=r.anonymous_node_key,source_path_private=r.png_abs_path,
                            real_deidentification_status='PENDING_VERIFIED_DEIDENTIFICATION' if r.branch=='REAL_HISTORY_REFERENCE' else 'RESTRICTED_NO_PUBLIC_RELEASE'))
        template.append(dict(blind_image_code=code,reviewer_code='',**{c:'' for c in scorecols},remarks='',review_date='',actual_review_status='NOT_RUN'))
    write_csv(dest/'EXPERT_BLINDED_RATING_TEMPLATE_V1.csv',template);write_csv(dest/'PRIVATE_EXPERT_BLIND_SOURCE_MAP_V1.csv',mapping)
    for p in [dest/'EXPERT_BLINDED_RATING_TEMPLATE_V1.csv',dest/'PRIVATE_EXPERT_BLIND_SOURCE_MAP_V1.csv']:os.chmod(p,0o600)
    atomic_json(dest/'EXPERT_TEMPLATE_STATUS_V1.json',dict(status='EXPERT_REVIEW_NOT_RUN',template_rows=len(template),reviewer_count=0,
                actual_scores=0,randomization_seed=20261012,real_images_require_deidentification_before_any_review=True,
                instructions='Before genuine independent review, define qualifications, scale interpretation, authorization and handling of correlated same-patient images. Never fill synthetic ratings.'))
    print({'blinded_template_rows':len(template),'actual_reviewers':0,'actual_scores':0},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',required=True);a=p.parse_args();main(a.output_root)
