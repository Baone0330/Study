"""Exact ON/OFF and auxiliary Full/ON changes, plus within-node six-seed diversity."""
from __future__ import annotations
import argparse
import itertools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from b16_quality_common import setup, read_json, atomic_parquet, atomic_json, state, now
from check_decode_and_image_stats import native_rgb, gray
from prepare_quality_models import offline_lpips


def basic(a,b):
    from skimage.metrics import structural_similarity
    assert a.shape==b.shape==(512,512,3)
    d=a.astype(np.float64)-b
    return dict(rgb_mae=float(np.abs(d).mean()),rgb_rmse=float(np.sqrt((d*d).mean())),
                rgb_max_abs=float(np.abs(d).max()),bit_exact=bool(np.array_equal(a,b)),
                gray_ssim=float(structural_similarity(gray(a),gray(b),data_range=255,gaussian_weights=True,
                                 sigma=1.5,win_size=11,use_sample_covariance=False)))


def cpu_node(entry):
    # The numerical cache preserves every image, including unusual or low-quality outputs.
    paths=entry['paths'];images={b:[native_rgb(x)[0] for x in pp] for b,pp in paths.items()}
    pairrows=[];divrows=[]
    for comparison,right in [('ON_OFF','OP_CPFM_OFF'),('ON_FULL_AUX','FULL_B16V1_HIST')]:
        for i in range(6):
            pairrows.append(dict(comparison=comparison,anonymous_patient_key=entry['patient'],anonymous_node_key=entry['node'],image_index=i,
                                 **basic(images['OP_CPFM_ON'][i],images[right][i])))
    for branch in ['OP_CPFM_ON','OP_CPFM_OFF']:
        for i,j in itertools.combinations(range(6),2):
            divrows.append(dict(branch=branch,anonymous_patient_key=entry['patient'],anonymous_node_key=entry['node'],
                                image_index_a=i,image_index_b=j,**basic(images[branch][i],images[branch][j])))
    return pairrows,divrows


def normalized_lpips_features(model,x):
    import lpips
    outs=model.net.forward(model.scaling_layer(x))
    return [lpips.normalize_tensor(z) for z in outs]


def lpips_from_features(model,fa,fb,ia,ib):
    import torch
    ia=torch.tensor(ia,device=fa[0].device);ib=torch.tensor(ib,device=fb[0].device)
    return sum(model.lins[k]((fa[k][ia]-fb[k][ib])**2).mean((2,3),keepdim=True) for k in range(model.L)).flatten()


def gpu_node(model,entry,device):
    import torch
    f={}
    for branch,pp in entry['paths'].items():
        a=np.stack([native_rgb(p)[0].transpose(2,0,1) for p in pp])
        x=torch.from_numpy(a).to(device,dtype=torch.float32)/127.5-1
        f[branch]=normalized_lpips_features(model,x)
    idx=list(range(6));both=list(itertools.combinations(idx,2));ia=[x[0] for x in both];ib=[x[1] for x in both]
    pairs=[]
    for right in ['OP_CPFM_OFF','FULL_B16V1_HIST']:
        pairs.extend(lpips_from_features(model,f['OP_CPFM_ON'],f[right],idx,idx).cpu().numpy().tolist())
    div=[]
    for b in ['OP_CPFM_ON','OP_CPFM_OFF']:
        div.extend(lpips_from_features(model,f[b],f[b],ia,ib).cpu().numpy().tolist())
    return pairs,div


def main(output_root,device='cuda',workers=4,smoke_only=False):
    root=setup(output_root)
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,_,_=offline_lpips(root,device)
    rng=np.random.default_rng(42)
    x=torch.from_numpy(rng.uniform(-1,1,(2,3,64,64)).astype(np.float32)).to(device)
    y=x.flip(-1)
    with torch.inference_mode():
        direct=model(x,y).flatten(); cached=lpips_from_features(model,normalized_lpips_features(model,x),normalized_lpips_features(model,y),[0,1],[0,1])
        error=float((direct-cached).abs().max());assert torch.allclose(direct,cached,rtol=1e-5,atol=1e-6)
    atomic_json(root/'metrics/LPIPS_CACHED_FEATURE_NUMERICAL_TOY_V1.json',dict(status='PASS',max_abs_diff=error,clinical_pixels=0))
    if smoke_only:print({'LPIPS_toy_pass':True,'max_abs_diff':error},flush=True);return
    q=pd.read_parquet(root/'manifests/COMBINED_QUALITY_EVAL_FROZEN_QUEUE_V1.parquet')
    on=q[q.branch=='OP_CPFM_ON'];nodes=list(on.anonymous_node_key.drop_duplicates());cache=root/'cache/paired_chunks';cache.mkdir(exist_ok=True)
    entries=[]
    for node in nodes:
        d=q[q.anonymous_node_key==node];patient=d.anonymous_patient_key.iloc[0]
        paths={b:d[d.branch==b].sort_values('image_index').png_abs_path.tolist() for b in ['OP_CPFM_ON','OP_CPFM_OFF','FULL_B16V1_HIST']}
        assert all(len(p)==6 for p in paths.values());entries.append(dict(node=node,patient=patient,paths=paths))
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for start in range(0,len(entries),32):
            pfile=cache/f'{start:05d}_pairs.parquet';dfile=cache/f'{start:05d}_diversity.parquet';batch=entries[start:start+32]
            if pfile.exists() and dfile.exists():
                prior=pd.read_parquet(pfile);assert list(prior.anonymous_node_key.drop_duplicates())==[e['node'] for e in batch];continue
            futures=[pool.submit(cpu_node,e) for e in batch]
            ps=[];ds=[]
            for e,future in zip(batch,futures):
                with torch.inference_mode(): pp,dp=gpu_node(model,e,device)
                p,d=future.result();assert len(p)==len(pp) and len(d)==len(dp)
                for r,v in zip(p,pp):r['lpips']=float(v)
                for r,v in zip(d,dp):r['lpips']=float(v)
                ps.extend(p);ds.extend(d)
            pdf,ddf=pd.DataFrame(ps),pd.DataFrame(ds)
            assert np.isfinite(pdf[['rgb_mae','rgb_rmse','gray_ssim','lpips']].to_numpy()).all()
            atomic_parquet(pfile,pdf);atomic_parquet(dfile,ddf)
            print(f'PAIRED_AND_DIVERSITY {min(start+32,len(entries))}/{len(entries)} nodes',flush=True)
    p=pd.concat([pd.read_parquet(f) for f in sorted(cache.glob('*_pairs.parquet'))],ignore_index=True)
    d=pd.concat([pd.read_parquet(f) for f in sorted(cache.glob('*_diversity.parquet'))],ignore_index=True)
    assert len(p)==18540 and len(d)==46350
    atomic_parquet(root/'metrics/PAIRED_ON_OFF_PIXEL_FEATURE_STATS_V1.parquet',p[p.comparison=='ON_OFF'])
    atomic_parquet(root/'metrics/PAIRED_FULL_ON_AUXILIARY_ROUTE_STATS_V1.parquet',p[p.comparison=='ON_FULL_AUX'])
    atomic_parquet(root/'metrics/NODE_SIX_SEED_DIVERSITY_V1.parquet',d)
    state(root,'PAIRED_CHANGES_AND_DIVERSITY_COMPLETED',primary_pairs=9270,auxiliary_pairs=9270,diversity_pairs=46350)
    print({'primary_pairs':9270,'auxiliary_pairs':9270,'diversity_pairs':46350},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',required=True);p.add_argument('--device',default='cuda');p.add_argument('--workers',type=int,default=4);p.add_argument('--smoke-only',action='store_true')
    a=p.parse_args();main(a.output_root,a.device,a.workers,a.smoke_only)
