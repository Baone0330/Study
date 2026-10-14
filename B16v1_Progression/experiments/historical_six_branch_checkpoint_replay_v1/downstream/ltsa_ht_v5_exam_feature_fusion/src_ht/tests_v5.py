"""Actual anonymous tests; no private patient records or inherited PASS marks."""
import argparse
import json
import random
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import torch
from common import config,setup,jout,ARMS
from model_base import LTSAHT as Base
from model_ht import LTSAHT,ExamResidualFusion,active_fusion_gradient_evidence
from dataset_ht import HTLongitudinalDataset,collate_patients,model_inputs,next_real_index
from runtime_ht import configure,metric_values,paired_bootstrap,lr_for_epoch,selection_key
from v5_safety import validate_checkpoint_path
C=None
class ToyEncoder(torch.nn.Module):
    def __init__(self):super().__init__();self.fc=torch.nn.Linear(3,512);self.seen=0
    def forward(self,x):self.seen+=len(x);return self.fc(x.mean((2,3)))
def sample(index=0,synth=0):
    return dict(sample_index=index,real_images=torch.randn(6,3,224,224),real_exam_view_counts=[2,2,2],real_relative_months=[0,24,48],synthetic_images=torch.randn(6*synth,3,224,224),synthetic_node_image_counts=[6]*synth,node_to_real_exam_index=[1]*synth,has_synth_mask=[False,bool(synth),False],label=index%2)
class FusionTests(unittest.TestCase):
    def setUp(self):configure(42);self.model=LTSAHT(C,78,encoder_override=ToyEncoder()).eval()
    def test_next_real_attachment(self):
        self.assertEqual(next_real_index(['2020-01-01','2022-01-01','2024-01-01'],'2021-01-01'),1)
        for q in ['2020-01-01','2022-01-01','2025-01-01']:
            with self.assertRaises(AssertionError):next_real_index(['2020-01-01','2022-01-01','2024-01-01'],q)
    def test_no_synth_identity_and_no_norm_call(self):
        real=torch.randn(3,512)
        with patch.object(self.model.fusion.real_norm,'forward',side_effect=RuntimeError('MUST_BYPASS')):
            fused=self.model.fusion(real,real.new_empty((0,512)),[],[False]*3)
        self.assertIs(fused,real);self.assertTrue(torch.equal(fused,real))
        _,features,pad,causal=self.model.temporal_forward(fused,[3],torch.tensor([[0,24,48]]),True)
        self.assertEqual(features.shape,(1,3,512));self.assertEqual(causal.shape,(3,3))
    def test_equal_node_mean_order_and_formula(self):
        real=torch.randn(3,512);nodes=torch.randn(2,512);m=self.model.fusion
        a=m(real,nodes,[1,1],[False,True,False]);b=m(real,nodes.flip(0),[1,1],[False,True,False])
        hr=m.real_norm(real[1:2]);hs=m.synthetic_norm(nodes.mean(0,keepdim=True))
        expected=real[1]+.1*torch.sigmoid(m.gate(torch.cat([hr,hs],1)))[0]*m.projection(hs-hr)[0]
        self.assertTrue(torch.allclose(a,b,atol=1e-6,rtol=0));self.assertTrue(torch.equal(a[0],real[0]));self.assertTrue(torch.allclose(a[1],expected,atol=1e-6,rtol=0))
    def test_real_and_op_masks_no_new_tokens(self):
        s=sample();op=dict(s,synthetic_images=torch.randn(6,3,224,224),synthetic_node_image_counts=[6],node_to_real_exam_index=[1],has_synth_mask=[False,True,False])
        a=collate_patients([s]);b=collate_patients([op]);self.assertEqual(a['real_exam_lengths'],b['real_exam_lengths']);self.assertTrue(torch.equal(a['real_relative_months'],b['real_relative_months']))
        outputs=[]
        for batch in [a,b]:
            args=model_inputs(batch,'cpu');f=self.model.fused_exam_features(**{k:v for k,v in args.items() if k not in ['real_exam_lengths','real_relative_months']})
            outputs.append(self.model.temporal_forward(f,args['real_exam_lengths'],args['real_relative_months'],True))
        self.assertTrue(torch.equal(outputs[0][2],outputs[1][2]));self.assertTrue(torch.equal(outputs[0][3],outputs[1][3]))
    def test_causal_and_padding_invariance(self):
        tokens=torch.randn(2,3,512);pad=torch.tensor([[False]*3,[False,True,True]]);mask=torch.ones(3,3,dtype=torch.bool).triu(1)
        a=self.model.transformer(tokens,mask=mask,src_key_padding_mask=pad,is_causal=True)
        tokens[0,2]+=100;tokens[1,1:]+=100
        b=self.model.transformer(tokens,mask=mask,src_key_padding_mask=pad,is_causal=True)
        self.assertTrue(torch.allclose(a[0,0:2],b[0,0:2],atol=1e-6,rtol=0));self.assertTrue(torch.allclose(a[1,0],b[1,0],atol=1e-6,rtol=0))
    def test_complete_native_views(self):
        x=torch.randn(58,3,224,224);f=self.model.encode_exams(x,[2,6,50]);self.assertEqual(self.model.encoder.seen,58)
        with torch.no_grad():expected=torch.stack([self.model.encoder(x[:2]).mean(0),self.model.encoder(x[2:8]).mean(0),self.model.encoder(x[8:]).mean(0)])
        self.assertTrue(torch.allclose(f,expected,atol=1e-6,rtol=0))
    def test_finite_nonzero_synthetic_and_all_fusion_gradients(self):
        self.model.train();s=sample(1,2);s['synthetic_images'].requires_grad_();batch=collate_patients([s]);z=self.model(**model_inputs(batch,'cpu'))
        loss=torch.nn.functional.binary_cross_entropy_with_logits(z,batch['labels']);loss.backward();active_fusion_gradient_evidence(self.model)
        self.assertTrue(torch.isfinite(s['synthetic_images'].grad).all());self.assertGreater(float(s['synthetic_images'].grad.abs().sum()),0)
        self.assertTrue(torch.isfinite(z).all() and torch.isfinite(loss) and torch.isfinite(torch.sigmoid(z)).all())
    def test_unused_fusion_none_gradients_expected(self):
        self.model.train();self.model(**model_inputs(collate_patients([sample()]),'cpu')).sum().backward()
        self.assertTrue(all(p.grad is None for p in self.model.fusion.parameters()));self.assertTrue(all(p.grad is not None for p in self.model.encoder.parameters()))
    def test_patient_isolation_and_batch_reorder(self):
        s,t=sample(0,0),sample(1,2)
        with torch.no_grad():a=self.model(**model_inputs(collate_patients([s,t]),'cpu'));b=self.model(**model_inputs(collate_patients([t,s]),'cpu'));one=self.model(**model_inputs(collate_patients([s]),'cpu'))
        self.assertTrue(torch.allclose(a,b.flip(0),atol=1e-6,rtol=0));self.assertTrue(torch.allclose(a[:1],one,atol=1e-6,rtol=0))
    def test_rng_and_baseline_initial_state(self):
        configure(2026);baseline=Base(C,78,pretrained=True);cpu=torch.get_rng_state().clone();py=random.getstate();nps=np.random.get_state();gpu=[s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else []
        configure(2026);v5=LTSAHT(C,78,pretrained=True)
        self.assertTrue(torch.equal(cpu,torch.get_rng_state()));self.assertEqual(py,random.getstate());self.assertTrue(np.array_equal(nps[1],np.random.get_state()[1]))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(gpu,torch.cuda.get_rng_state_all())))
        for k,v in baseline.state_dict().items():self.assertTrue(torch.equal(v,v5.state_dict()[k]))
        self.assertEqual(sum(p.numel() for p in baseline.parameters()),12755009);self.assertEqual(sum(p.numel() for p in v5.parameters()),13020738)
    def test_fake_target_and_test_never_opened_and_other_patient_ignored(self):
        root=Path(C['project_private'])/'tests/anonymous';root.mkdir(exist_ok=True)
        target=root/'target_year.png';test=root/'Official_Test.png';target.write_bytes(b'FAKE');test.write_bytes(b'FAKE')
        exams=[dict(date='2020-01-01',views=2,paths=['ANONYMOUS_HISTORY_1','ANONYMOUS_HISTORY_2'],source_type='REAL')]
        record=dict(patient_index=0,split='val',eligibility_status='ELIGIBLE',target_year=2024,target_HT=0,exams_private_json=json.dumps(exams),native_cache_private=str(root/'native.npy'),relative_months_private_json='[0]')
        import pandas as pd
        frames=pd.DataFrame([dict(record,patient_index=i) for i in range(189)])
        opened=[]
        def read_pixels(p,**kw):
            opened.append(str(p));assert Path(p) not in [target,test];return np.zeros((2,224,224,3),dtype=np.uint8)
        with patch('dataset_ht.pd.read_parquet',return_value=frames),patch('dataset_ht.jread',side_effect=AssertionError('Val must not read synthetic ledger')),patch('dataset_ht.np.load',side_effect=read_pixels):
            d=HTLongitudinalDataset(C['project_private'],'val',ARMS[1]);d.augmentation={999:[dict(synthetic_cache_private=[str(test)])]};out=d[0]
        self.assertEqual(opened,[str(root/'native.npy')]);self.assertEqual(len(out['synthetic_images']),0);self.assertEqual(out['has_synth_mask'],[False])
        with self.assertRaises(AssertionError):HTLongitudinalDataset(root,'test',ARMS[0])
    def test_old_checkpoints_and_own_resume_guard(self):
        root=Path(C['project_private']);self.assertEqual(validate_checkpoint_path(root/'checkpoints/anonymous/best.pt',root),root/'checkpoints/anonymous/best.pt')
        for version in ['v3_local_source','v4_independent_retrain']:
            with self.assertRaises(PermissionError):validate_checkpoint_path(root.parent/('ltsa_ht_real_vs_original_progression_'+version)/'checkpoints/best.pt',root)
    def test_labels_scheduler_selection_bootstrap(self):
        self.assertEqual(metric_values([0,1],[.1,.9],.5)['TP'],1);self.assertAlmostEqual(lr_for_epoch(5,C),1e-4);self.assertAlmostEqual(lr_for_epoch(60,C),0)
        self.assertGreater(selection_key(dict(ROC_AUC=.7,AP=.5),.6,2),selection_key(dict(ROC_AUC=.7,AP=.5),.6,3))
        b=paired_bootstrap([0,0,1,1],[.1,.2,.8,.9],[.1,.2,.8,.9],30,42);self.assertTrue(all(r['delta_OP_minus_REAL']==0 for r in b))
def main(path):
    global C
    C=config(path);root=setup(C['project_private']);result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FusionTests))
    a=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),actual_patient_data_used=False,framework='stdlib unittest',new_V5_tests=True)
    jout(root/'tests/ANONYMOUS_CPU_TEST_RESULTS_V5.json',a)
    (root/'audit/V5_FEATURE_FUSION_FORWARD_TESTS.md').write_text('# Anonymous forward tests\n\n'+json.dumps(a,indent=2)+'\n\nActual tests cover identity, no extra tokens, next-real binding, equal-node averaging, causal/padding, all views, synthetic/gate/projection/LayerNorm gradients, isolation, fake target/Test non-access, RNG/base-state parity and own-checkpoint boundary. Actual maximum-window GPU and metadata protection are separate receipts.\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);main(p.parse_args().config)
