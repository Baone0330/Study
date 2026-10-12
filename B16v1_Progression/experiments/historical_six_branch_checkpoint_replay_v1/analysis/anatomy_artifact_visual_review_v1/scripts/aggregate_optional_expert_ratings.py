"""Fail closed on actual scores without qualified approval. Current run has no expert data."""
import argparse
import pandas as pd
from review_common import setup,read_json,csv_out,json_out


def main(output_root):
    root=setup(output_root);ratings=pd.read_csv(root/'metrics/CLINICIAN_BLIND_RATINGS_V1.csv');metadata=pd.read_csv(root/'reviews_private/reviewers_metadata_private.csv')
    assert ratings.empty and metadata.empty,'Actual expert import is NOT authorized by this preparation-only route; validate original approved records in a separate stage'
    csv_out(root/'metrics/OPTIONAL_EXPERT_RATING_STATUS_V1.csv',[dict(status='NOT_RUN',actual_clinicians=0,actual_rated_images=0,
          candidate_pairs=36,candidate_images=72,candidate_nodes=12,candidate_patients=12,mean_score=None,paired_delta=None,ci_low=None,ci_high=None,
          kappa=None,icc=None,reason='No authentic qualified approved local clinician ratings; empty cells are missing, not zero')])
    json_out(root/'release/OPTIONAL_EXPERT_REVIEW_STATUS_V1.json',dict(status='NOT_RUN',actual_clinicians=0,actual_rated_images=0,kappa=None,icc=None,
                   full_original_view_confirmations=0,public_anatomy_certification=False))
    print({'expert_review':'NOT_RUN','actual_scores':0})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',required=True);a=p.parse_args();main(a.output_root)
