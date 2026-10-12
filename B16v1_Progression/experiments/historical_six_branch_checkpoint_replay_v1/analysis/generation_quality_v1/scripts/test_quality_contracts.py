"""Anonymous tests only: identity, metric meaning, cluster units, missing inputs, publication."""
from __future__ import annotations
import unittest
import tempfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor'))
import numpy as np
import pandas as pd
from calculate_fid_kid import toy_tests, mmd_u
from calculate_paired_image_changes import basic
from aggregate_patient_cluster_metrics import patient_values, patient_ci


class ContractTests(unittest.TestCase):
    def test_pair_alignment_toy(self):
        a=pd.DataFrame({'node':['N1','N1'],'index':[0,1],'seed':[4,7]})
        b=a.iloc[::-1].copy();paired=a.merge(b,on=['node','index'],validate='1:1')
        self.assertTrue((paired.seed_x==paired.seed_y).all())
        b.loc[b['index']==1,'seed']=8;bad=a.merge(b,on=['node','index'],validate='1:1')
        self.assertFalse((bad.seed_x==bad.seed_y).all())

    def test_metric_units_toy(self):
        a=np.zeros((512,512,3),np.uint8);b=np.full_like(a,20)
        self.assertEqual(basic(a,a)['gray_ssim'],1.0)
        change=basic(a,b);self.assertEqual(change['rgb_mae'],20.0);self.assertEqual(change['rgb_rmse'],20.0)
        self.assertFalse(change['bit_exact']);self.assertEqual(toy_tests()['status'],'PASS')

    def test_missing_reference_is_error_not_fake_score(self):
        with self.assertRaises(AssertionError):mmd_u(np.ones((1,4)),np.ones((5,4)))

    def test_patient_cluster_weighting(self):
        d=pd.DataFrame(dict(branch=['OP_CPFM_ON']*3,anonymous_patient_key=['P1','P1','P2'],anonymous_node_key=['N1','N2','N3'],
                            real_exam_key_private=['']*3,value=[0.,10.,100.]))
        p=patient_values(d,['value']);self.assertEqual(p.value.mean(),52.5)
        self.assertEqual(patient_ci(p.value),patient_ci(p.value))

    def test_publication_privacy_toy(self):
        from review_github_publication_scope import allowed_relative
        self.assertTrue(allowed_relative('figures/nonmedical/Fig02_Real_vs_ON_OFF_FID_KID.svg'))
        self.assertFalse(allowed_relative('figures/private_medical_panels/plate.png'))
        self.assertFalse(allowed_relative('manifests/pair_map.parquet'))
        self.assertFalse(allowed_relative('image_features/on/features.npz'))


if __name__=='__main__':unittest.main(verbosity=2)
