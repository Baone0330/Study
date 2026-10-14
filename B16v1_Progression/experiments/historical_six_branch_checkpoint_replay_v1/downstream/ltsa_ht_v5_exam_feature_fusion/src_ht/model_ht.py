"""One pre-registered next-real Exam fusion; unchanged inherited temporal forward."""
import random
import numpy as np
import torch
from model_base import LTSAHT as BaseLTSAHT

class ExamResidualFusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.real_norm=torch.nn.LayerNorm(512); self.synthetic_norm=torch.nn.LayerNorm(512)
        self.gate=torch.nn.Linear(1024,1); self.projection=torch.nn.Linear(512,512)
    def forward(self,real,node_features,node_to_real_exam_index,has_synth_mask):
        mask=torch.as_tensor(has_synth_mask,dtype=torch.bool,device=real.device)
        assert mask.shape==(len(real),)
        if len(node_features)==0:
            assert not bool(mask.any()) and len(node_to_real_exam_index)==0
            return real  # Strict identity: no normalization, dummy statistics or extra RNG.
        indices=torch.as_tensor(node_to_real_exam_index,dtype=torch.long,device=real.device)
        assert len(indices)==len(node_features) and int(indices.min())>=0 and int(indices.max())<len(real)
        actual=torch.zeros_like(mask);actual[indices]=True;assert torch.equal(actual,mask)
        # Mean of node means, independent of the number of nodes at an Exam.
        synthetic=torch.stack([node_features[indices==i].mean(0) for i in torch.where(mask)[0].tolist()])
        hr=self.real_norm(real[mask]);hs=self.synthetic_norm(synthetic)
        gate=torch.sigmoid(self.gate(torch.cat([hr,hs],dim=1)))
        delta=self.projection(hs-hr);fused=real.clone()
        fused[mask]=real[mask]+0.1*gate*delta
        return fused

class LTSAHT(BaseLTSAHT):
    def __init__(self,c,max_month,pretrained=True,encoder_override=None):
        # Construct baseline in exactly the V3 sequence, then isolate new RNG consumption.
        super().__init__(c,max_month,pretrained,encoder_override)
        py=random.getstate();np_state=np.random.get_state();cpu=torch.get_rng_state()
        gpu=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        try:self.fusion=ExamResidualFusion()
        finally:
            random.setstate(py);np.random.set_state(np_state);torch.set_rng_state(cpu)
            if gpu is not None:torch.cuda.set_rng_state_all(gpu)
    def fused_exam_features(self,real_images,real_exam_view_counts,synthetic_images,synthetic_node_image_counts,node_to_real_exam_index,has_synth_mask):
        real=self.encode_exams(real_images,real_exam_view_counts)
        assert len(synthetic_images)==sum(synthetic_node_image_counts)
        assert all(n==6 for n in synthetic_node_image_counts)
        nodes=self.encode_exams(synthetic_images,synthetic_node_image_counts) if len(synthetic_node_image_counts) else real.new_empty((0,512))
        return self.fusion(real,nodes,node_to_real_exam_index,has_synth_mask)
    def forward(self,real_images,real_exam_view_counts,real_exam_lengths,real_relative_months,synthetic_images,synthetic_node_image_counts,node_to_real_exam_index,has_synth_mask):
        fused=self.fused_exam_features(real_images,real_exam_view_counts,synthetic_images,synthetic_node_image_counts,node_to_real_exam_index,has_synth_mask)
        return self.temporal_forward(fused,real_exam_lengths,real_relative_months)

def assert_finite_gradients(model):
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        if name.startswith('fusion.') and p.grad is None:continue  # Both arms can have no-synthetic accumulation groups.
        assert p.grad is not None and bool(torch.isfinite(p.grad).all()),'Nonfinite/missing active-path gradient'

def active_fusion_gradient_evidence(model):
    out={}
    for name,p in model.named_parameters():
        if name.startswith('fusion.'):
            assert p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(p.grad.abs().sum()>0)
            out[name]=float(p.grad.detach().norm())
    encoder=[p.grad for p in model.encoder.parameters() if p.requires_grad]
    assert encoder and all(g is not None and bool(torch.isfinite(g).all()) for g in encoder)
    assert sum(float(g.detach().abs().sum()) for g in encoder)>0
    return out
