"""Original local LTSA TPE plus equivalent ResNet18/causal Transformer, with HT-only head."""
from __future__ import annotations
import ast
import math
from pathlib import Path
import torch
import torchvision


def load_original_tpe(snapshot):
    source=Path(snapshot)/'src/models.py';tree=ast.parse(source.read_text(encoding='utf-8'))
    node=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='PositionalEncoding')
    namespace={'torch':torch,'math':math,'__name__':'ltsa_local_tpe'}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    return namespace['PositionalEncoding']


def transformer_features(output):
    if isinstance(output,torch.Tensor):return output
    if isinstance(output,tuple) and len(output)==2 and isinstance(output[0],torch.Tensor):return output[0]
    raise TypeError('Unsupported local transformer return signature; never unpack a Tensor as a tuple')


def masked_view_mean(features,valid=None):
    if valid is not None:features=features[valid]
    assert len(features)>0;return features.mean(0)


class LTSAHT(torch.nn.Module):
    def __init__(self,c,max_month,pretrained=True,encoder_override=None):
        super().__init__();self.c=dict(c);self.view_chunk=int(c['encoder_view_chunk'])
        if encoder_override is None:
            backbone=torchvision.models.resnet18(weights=None)
            if pretrained:
                weights=torch.load(c['resnet18_weights_private'],map_location='cpu',weights_only=True)
                backbone.load_state_dict(weights,strict=True)
            backbone.fc=torch.nn.Identity();self.encoder=backbone
        else:self.encoder=encoder_override  # Anonymous unit tests only; never allowed in formal trainer.
        tpe=load_original_tpe(c['snapshot_private']);self.pos_encoder=tpe(512,dropout=0,max_len=int(max_month)+1)
        layer=torch.nn.TransformerEncoderLayer(d_model=512,nhead=4,dim_feedforward=512,dropout=.25,activation='relu',batch_first=True,norm_first=False)
        self.transformer=torch.nn.TransformerEncoder(layer,num_layers=1,enable_nested_tensor=False)
        self.classifier=torch.nn.Sequential(torch.nn.Dropout(.25),torch.nn.Linear(512,1))

    def encode_exams(self,images,exam_view_counts):
        assert images.ndim==4 and images.shape[1:]==(3,224,224) and sum(exam_view_counts)==len(images)
        assert all(int(n)>0 for n in exam_view_counts)
        chunks=[self.encoder(images[i:i+self.view_chunk]) for i in range(0,len(images),self.view_chunk)]
        features=torch.cat(chunks,dim=0);assert features.shape==(len(images),512)
        return torch.stack([masked_view_mean(x) for x in features.split(exam_view_counts,dim=0)])

    def temporal_forward(self,exam_features,lengths,relative_months,return_features=False):
        lengths=torch.as_tensor(lengths,device=exam_features.device,dtype=torch.long);b=len(lengths);t=int(lengths.max())
        assert int(lengths.sum())==len(exam_features) and int(lengths.min())>=1
        tokens=exam_features.new_zeros((b,t,512));offset=0
        for i,n in enumerate(lengths.tolist()):tokens[i,:n]=exam_features[offset:offset+n];offset+=n
        months=relative_months.to(device=tokens.device,dtype=torch.long)
        assert months.shape==(b,t) and int(months.min())>=0 and int(months.max())<self.pos_encoder.pe.shape[1]
        padding=torch.arange(t,device=tokens.device)[None,:]>=lengths[:,None]
        # Original TPE is intentionally used on a nonleaf clone; no target-year/query-gap input.
        encoded=self.pos_encoder(tokens.clone(),months)
        causal=torch.ones((t,t),dtype=torch.bool,device=tokens.device).triu(1)
        features=transformer_features(self.transformer(encoded,mask=causal,src_key_padding_mask=padding,is_causal=True))
        last=features[torch.arange(b,device=tokens.device),lengths-1]
        logits=self.classifier(last).squeeze(-1)
        return (logits,features,padding,causal) if return_features else logits

    def forward(self,images,exam_view_counts,lengths,relative_months):
        return self.temporal_forward(self.encode_exams(images,exam_view_counts),lengths,relative_months)
