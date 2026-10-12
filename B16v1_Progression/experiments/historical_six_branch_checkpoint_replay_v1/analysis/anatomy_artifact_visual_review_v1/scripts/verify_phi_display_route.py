"""Record the actual absence of an approved display/de-identification/reviewer route."""
import argparse
import pandas as pd
from review_common import setup,csv_out,json_out,now


def main(output_root):
    root=setup(output_root)
    people=pd.read_csv(root/'reviews_private/reviewers_metadata_private.csv');assert people.empty
    assert not list((root/'reviews_private/raw_completed_forms').iterdir())
    json_out(root/'release/PHI_DISPLAY_ROUTE_STATUS_V1.json',dict(REAL_VISUAL_REFERENCE='WITHHELD',PHI_REVIEW='PENDING',
        GENERATED_IMAGE_PUBLIC_APPROVAL='NOT_APPROVED',CLINICIAN_REVIEW='NOT_RUN',NONCLINICAL_HUMAN_REVIEW='NOT_RUN',CODEX_MEDICAL_VISUAL_REVIEW='NOT_RUN',
        actual_approved_reviewers=0,external_medical_pixels_transferred=0,verified_deidentification_asset_count=0,
        reason='No authentic institutional approval, verified PHI display clearance or qualified local reviewer records supplied',checked_utc=now()))
    csv_out(root/'source_audit/REAL_VIEW_DISPLAY_ELIGIBILITY_V1.csv',[dict(branch='REAL_HISTORY_REFERENCE',views=13200,exams=3154,patients=859,
                 actual_public_or_chat_displayed=0,actual_private_REAL_panels_rendered=0,status='WITHHELD',PHI_REVIEW='PENDING',route='NOT_ESTABLISHED')])
    print({'real_display':'WITHHELD','approved_reviewers':0,'medical_pixels_transferred':0})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',required=True);a=p.parse_args();main(a.output_root)
