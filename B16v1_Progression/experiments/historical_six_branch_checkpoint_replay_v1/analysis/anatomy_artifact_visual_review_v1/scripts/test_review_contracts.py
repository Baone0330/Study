"""Anonymous regression only: identities, localization, human roles/NA and source protection."""
from __future__ import annotations
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from review_common import unique_pairs,metadata,json_out,postflight
from localize_artifact_candidates import diagnose_array,line_geometry
from aggregate_nonclinical_review import validate_observation,human_agreement_unperformed,paired_bootstrap
from review_publication_scope import allowed_relative,text_issues,compile_values


class ReviewContractTests(unittest.TestCase):
    def test_exact_pair_identity(self):
        q=[];p=[]
        for i in range(2):
            for b in ['OP_CPFM_ON','OP_CPFM_OFF']:q.append(dict(branch=b,patient_key_private='TOY_PAT',historical_node_key_private='TOY_NODE',
                 image_index=i,seed=7+i,source_date_private='TOY_DATE',anonymous_patient_key='PTOY',anonymous_node_key='NTOY',png_abs_path=b+str(i)))
            p.append(dict(patient_key_private='TOY_PAT',historical_node_key='TOY_NODE',image_index=i,master_seed=7+i,nominal_slot_id=2,
                          reference_ON_image_path='OP_CPFM_ON'+str(i),new_OFF_image_path='OP_CPFM_OFF'+str(i)))
        df=pd.DataFrame(q);other=pd.DataFrame(p);out=unique_pairs(df,other,2,1,1,2);self.assertEqual(len(out),2)
        df.loc[1,'seed']=55
        with self.assertRaises(AssertionError):unique_pairs(df,other,2,1,1,2)

    def test_geometric_localization_not_anatomy(self):
        a=np.zeros((512,512,3),np.uint8);self.assertFalse(diagnose_array(a)['reproduced_long_line_flag'])
        a[255:258,100:411]=255;a[100:411,255:258]=255;d=diagnose_array(a)
        self.assertTrue(d['reproduced_cross_proxy_flag']);self.assertFalse(d['anatomical_ROI'])
        line=line_geometry(np.array([[0,256,511,256]]))[0]
        self.assertEqual(line['candidate_location'],'MULTIZONE');self.assertIn('BORDER',line['zones']);self.assertIn('CENTRAL_FIELD',line['zones'])

    def test_reviewer_role_integrity(self):
        self.assertTrue(validate_observation(dict(observer_role='NONE'),None))
        with self.assertRaises(AssertionError):validate_observation(dict(observer_role='MACHINE_PROXY',thyroid_like_region_visibility='CLEAR'),None)
        with self.assertRaises(AssertionError):validate_observation(dict(observer_role='CLINICIAN',review_quality_status='FULL_IMAGE_VIEWED'),None)

    def test_unknown_handling_and_no_generated_scores(self):
        d=human_agreement_unperformed();self.assertEqual(d['status'],'NOT_RUN')
        for key in ['TP','FP','TN','FN','kappa','icc']:self.assertIsNone(d[key])
        a=paired_bootstrap([1,2,3],[2,3,4]);self.assertEqual(a['delta_on_minus_off'],-1);self.assertEqual(a['ci_low'],-1)

    def test_no_label_or_private_path_export(self):
        self.assertFalse(allowed_relative('figures/private_medical_panels/card.png'))
        self.assertFalse(allowed_relative('figures/public_nonmedical/patient.png'))
        self.assertTrue(allowed_relative('figures/public_nonmedical/Fig01_Review_Design_and_Cohorts.png'))
        self.assertIn('PRIVATE_ABSOLUTE_PATH',text_issues('/'+'data/toy/private'))
        self.assertIn('PRIVATE_PATIENT_VALUE',text_issues('patient=TOY_PRIVATE_PERSON',['TOY_PRIVATE_PERSON']))
        self.assertIn('PRIVATE_CASE_CODE',text_issues('card=C123ABC',private_codes=['C123ABC']))
        matcher=compile_values(['ABCDE','ABCDE6','ABCDG'])
        self.assertIsNotNone(matcher.search('x ABCDE6 y'));self.assertIsNone(matcher.search('xABCDE6x'))
        self.assertIsNotNone(matcher.search('ABCDG'));self.assertIsNone(matcher.search('ABCD'))

    def test_no_generated_clinician_score(self):
        row=dict(observer_role='MACHINE_PROXY',thyroid_like_region_visibility='CLEAR',review_quality_status='FULL_IMAGE_VIEWED')
        with self.assertRaises(AssertionError):validate_observation(row,None)

    def test_no_source_mutation(self):
        with tempfile.TemporaryDirectory() as t:
            r=Path(t);(r/'source_audit').mkdir();s=r/'toy.txt';s.write_text('immutable toy',encoding='utf-8')
            json_out(r/'source_audit/PROTECTED_SOURCE_METADATA_THIS_TASK_V1.json',[metadata(s)])
            self.assertEqual(postflight(r)['status'],'PASS');s.write_text('changed toy only',encoding='utf-8')
            with self.assertRaises(RuntimeError):postflight(r)


if __name__=='__main__':unittest.main(verbosity=2)
