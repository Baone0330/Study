"""Anonymous CPU regression only; real patient/source data are never used in public tests."""
import argparse
import json
import unittest
from datetime import date
from pathlib import Path
import numpy as np
import torch
from common import config,setup,jout
from model_ht import LTSAHT,masked_view_mean,transformer_features
from dataset_ht import HTLongitudinalDataset,collate_patients,model_inputs
from build_ht_manifest import allowed_history,relative_months
from runtime_ht import configure,metric_values,selection_key,lr_for_epoch,paired_bootstrap
from validate_original_progression_lineage import eligibility

CONFIG=None


class ToyEncoder(torch.nn.Module):
    def __init__(self):super().__init__();self.fc=torch.nn.Linear(3,512)
    def forward(self,x):return self.fc(x.mean((2,3)))


class LTSAContractTests(unittest.TestCase):
    def setUp(self):configure(42);self.model=LTSAHT(CONFIG,78,encoder_override=ToyEncoder()).eval()
    def test_target_year_exclusion(self):
        self.assertTrue(allowed_history(date(2000,12,31),2001));self.assertFalse(allowed_history(date(2001,1,1),2001))
    def test_patient_split_overlap(self):
        train={101,102};val={201};test={301};self.assertFalse(train&val or train&test or val&test)
    def test_label_alignment(self):
        p=metric_values([0,1],[.1,.9],.5);self.assertEqual(p['TP'],1);self.assertEqual(p['TN'],1);self.assertEqual(p['N'],2)
    def test_multiview_exam_pool(self):
        x=torch.arange(3*512,dtype=torch.float32).reshape(3,512);self.assertTrue(torch.equal(masked_view_mean(x),masked_view_mean(x[[2,0,1]])))
        self.assertTrue(torch.equal(masked_view_mean(x,torch.tensor([True,False,True])),(x[0]+x[2])/2))
    def test_time_order_mask(self):
        feat=torch.randn(3,512);times=torch.tensor([[0,2],[0,0]])
        with torch.no_grad():a,h,pad,causal=self.model.temporal_forward(feat,[2,1],times,True)
        self.assertTrue(bool(pad[1,1]));self.assertTrue(bool(causal[0,1]));self.assertFalse(bool(causal[1,0]))
        swapped=torch.cat([feat[2:],feat[:2]])
        with torch.no_grad():b=self.model.temporal_forward(swapped,[1,2],times[[1,0]])
        self.assertTrue(torch.allclose(a,b[[1,0]],atol=1e-6,rtol=0))
        self.assertEqual(relative_months([date(2000,1,1),date(2000,3,1)]),[0,1])
        with self.assertRaises(AssertionError):relative_months([date(2000,3,1),date(2000,1,1)])
    def test_padding_mutation_invariance(self):
        layer=self.model.transformer;tokens=torch.randn(2,3,512);padding=torch.tensor([[False,False,False],[False,True,True]]);mask=torch.ones(3,3,dtype=torch.bool).triu(1)
        with torch.no_grad():a=layer(tokens,mask=mask,src_key_padding_mask=padding,is_causal=True)
        tokens[1,1:]=torch.randn(2,512)*100
        with torch.no_grad():b=layer(tokens,mask=mask,src_key_padding_mask=padding,is_causal=True)
        self.assertTrue(torch.allclose(a[1,0],b[1,0],atol=1e-6,rtol=0))
    def test_observed_image_and_month_sensitivity(self):
        x=torch.randn(2,512)
        with torch.no_grad():a=self.model.temporal_forward(x,[2],torch.tensor([[0,1]]));b=self.model.temporal_forward(x,[2],torch.tensor([[0,20]]));d=self.model.temporal_forward(x+1,[2],torch.tensor([[0,1]]))
        self.assertGreater(float((a-b).abs().max()),1e-7);self.assertGreater(float((a-d).abs().max()),1e-7)
    def test_device_compatibility_cpu(self):
        x=torch.randn(1,512,requires_grad=True);z=self.model.temporal_forward(x,[1],torch.tensor([[0]]));self.assertEqual(z.device.type,'cpu')
        z.sum().backward();self.assertTrue(torch.isfinite(x.grad).all())
    def test_original_source_interface(self):
        x=torch.randn(2,3,512);self.assertIs(transformer_features(x),x);self.assertIs(transformer_features((x,None)),x)
        with self.assertRaises(TypeError):transformer_features((x,None,None))
    def test_no_target_info_in_step_ahead(self):
        self.assertFalse(hasattr(self.model,'step_ahead_predictor'));self.assertEqual(CONFIG['step_ahead'],False)
    def test_finite_forward_and_grad(self):
        self.model.train();x=torch.randn(3,3,224,224);z=self.model(x,[1,2],[2],torch.tensor([[0,1]]));loss=torch.nn.functional.binary_cross_entropy_with_logits(z,torch.tensor([1.]))
        loss.backward();grads=[p.grad for p in self.model.parameters() if p.requires_grad];self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        opt=torch.optim.AdamW(self.model.parameters(),lr=1e-4);opt.step();self.assertTrue(all(torch.isfinite(p).all() for p in self.model.parameters()))
    def test_no_test_access(self):
        with self.assertRaises(AssertionError):HTLongitudinalDataset(Path('ANONYMOUS_NO_FILES'),'test')
    def test_augmented_mapping(self):
        first=date(2000,1,1);last=date(2004,1,1)
        self.assertEqual(eligibility(first,last,date(2002,1,1),2005),'ELIGIBLE_BOUNDED_SYNTHETIC_QUERY_TOKEN')
        self.assertEqual(eligibility(first,last,date(2005,1,1),2005),'EXCLUDE_TARGET_YEAR_SYNTH_PREDICTION')
        self.assertEqual(eligibility(first,last,date(2004,6,1),2005),'EXCLUDE_UNKNOWN_FUTURE_AFTER_LAST_REAL')
        self.assertEqual(eligibility(first,last,first,2005),'EXCLUDE_NO_REAL_HISTORY_BEFORE_QUERY')
    def test_two_arm_and_scheduler_selection(self):
        self.assertEqual(CONFIG['arms'],['LTSA_REAL_ONLY','LTSA_ORIGINAL_PROGRESSION_HIST'])
        self.assertAlmostEqual(lr_for_epoch(5,CONFIG),1e-4);self.assertAlmostEqual(lr_for_epoch(60,CONFIG),0)
        self.assertGreater(selection_key(dict(ROC_AUC=.7,AP=.5),.6,2),selection_key(dict(ROC_AUC=.7,AP=.5),.6,3))
    def test_paired_bootstrap_and_ensemble_unit(self):
        y=np.array([0,0,1,1]);p=np.array([.1,.3,.6,.8]);out=paired_bootstrap(y,p,p,30,42)
        self.assertTrue(all(r['delta_OP_minus_REAL']==r['ci_low']==r['ci_high']==0 for r in out))
        e=np.stack([p,p,p]).mean(0);self.assertTrue(np.allclose(e,p,atol=1e-15,rtol=0))


def main(path):
    global CONFIG
    CONFIG=config(path);root=setup(CONFIG['project_private']);result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LTSAContractTests))
    jout(root/'tests/ANONYMOUS_CPU_TEST_RESULTS_V3.json',dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),actual_patient_data_used=False,framework='stdlib unittest; pytest not installed'))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();main(a.config)
