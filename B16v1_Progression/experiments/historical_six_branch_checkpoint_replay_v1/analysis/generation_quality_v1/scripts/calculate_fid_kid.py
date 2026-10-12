"""Fixed standard-feature FID/KID, common-support sensitivity, paired patient bootstrap."""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4';os.environ['MKL_NUM_THREADS']='4'
import argparse
import numpy as np
import pandas as pd
from scipy import linalg
from b16_quality_common import setup, read_json, atomic_json, write_csv, state
from extract_quality_features import load_features


def fid(x,y):
    x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    assert x.ndim==y.ndim==2 and x.shape[1]==y.shape[1] and min(len(x),len(y))>=2
    if np.array_equal(x,y):return 0.0
    mx,my=x.mean(0),y.mean(0);cx=np.cov(x,rowvar=False);cy=np.cov(y,rowvar=False)
    cx=np.atleast_2d(cx);cy=np.atleast_2d(cy)
    w,v=linalg.eigh(cx,check_finite=True)
    assert w.min()>=-1e-8*max(1,abs(w).max())
    sx=(v*np.sqrt(np.maximum(w,0)))@v.T
    product=sx@cy@sx;product=(product+product.T)*.5
    z=linalg.eigvalsh(product,check_finite=True)
    assert z.min()>=-1e-8*max(1,abs(z).max())
    value=float((mx-my)@(mx-my)+np.trace(cx)+np.trace(cy)-2*np.sqrt(np.maximum(z,0)).sum())
    if value<0 and value>=-1e-6:return 0.0
    assert value>=0 and np.isfinite(value), value
    return value


def polynomial(x,y):
    assert x.shape[1]==y.shape[1]
    return (x.astype(np.float64)@y.astype(np.float64).T/x.shape[1]+1)**3


def mmd_u(x,y):
    kxx=polynomial(x,x);kyy=polynomial(y,y);kxy=polynomial(x,y);n,m=len(x),len(y)
    assert min(n,m)>1
    return float((kxx.sum()-np.trace(kxx))/(n*(n-1))+(kyy.sum()-np.trace(kyy))/(m*(m-1))-2*kxy.mean())


def kid_mc(x,y,subsets=50,size=512,seed=20261012):
    assert min(len(x),len(y))>=size
    rng=np.random.default_rng(seed);values=[]
    for _ in range(subsets):
        a=x[rng.choice(len(x),size,replace=False)];b=y[rng.choice(len(y),size,replace=False)]
        values.append(mmd_u(a,b))
    return float(np.mean(values)),float(np.std(values,ddof=1)),values


def grouped_kernel(x,y,p,m):
    return polynomial(x,y).reshape(p,m,p,m).sum((1,3))


def paired_balanced_kid_bootstrap(real,on,off,p,m,repeats=2000,seed=20261012):
    assert len(real)==len(on)==len(off)==p*m
    dxx=grouped_kernel(off,off,p,m)-grouped_kernel(on,on,p,m)
    diag=(np.diag(polynomial(off,off))-np.diag(polynomial(on,on))).reshape(p,m).sum(1)
    dxy=grouped_kernel(real,off,p,m)-grouped_kernel(real,on,p,m)
    n=p*m
    def eval_counts(c):
        return (np.einsum('bi,ij,bj->b',c,dxx,c,optimize=True)-c@diag)/(n*(n-1))-2*np.einsum('bi,ij,bj->b',c,dxy,c,optimize=True)/n**2
    point=float(eval_counts(np.ones((1,p)))[0])
    rng=np.random.default_rng(seed);counts=rng.multinomial(p,np.full(p,1/p),size=repeats).astype(np.float64)
    vals=eval_counts(counts);assert np.isfinite(vals).all()
    lo,hi=np.quantile(vals,[.025,.975])
    return dict(comparison='KID_U_OFF_minus_ON_vs_REAL',scope='patient_balanced_common_support',point_estimate=point,
                ci_low=float(lo),ci_high=float(hi),bootstrap_sd=float(vals.std(ddof=1)),patients=p,images_per_patient=m,
                repeats=repeats,seed=seed,method='paired patient-cluster percentile bootstrap of full unbiased polynomial MMD squared',
                monte_carlo_subset_variance=False)


def toy_tests():
    rng=np.random.default_rng(5);x=rng.normal(size=(14,4));y=rng.normal(size=(12,4));z=x+2
    assert fid(x,x)==0 and abs(fid(x,z)-16)<1e-8
    assert abs(fid(x,y)-fid(x[::-1],y[::-1]))<1e-8
    assert abs(mmd_u(x,y)-mmd_u(x[::-1],y[::-1]))<1e-12
    # A finite unbiased estimator is not constrained to be nonnegative.
    assert mmd_u(x,x)<0
    r=rng.normal(size=(8,4));o=rng.normal(size=(8,4));f=rng.normal(size=(8,4))
    out=paired_balanced_kid_bootstrap(r,o,f,4,2,repeats=12)
    assert abs(out['point_estimate']-(mmd_u(r,f)-mmd_u(r,o)))<1e-12
    return {'status':'PASS','permutation_invariant_exact_estimators':True,'negative_unbiased_KID_retained':True,'paired_kernel_bootstrap_formula_matches_direct':True}


def main(output_root,toy_only=False):
    root=setup(output_root);atomic_json(root/'tests/FID_KID_NUMERICAL_TOY_V1.json',toy_tests())
    if toy_only:print(toy_tests());return
    cfg=read_json(root/'protocol/FROZEN_QUALITY_EVALUATION_CONTRACT_V1.json');common=cfg['balanced_policy']['common_patients']
    branches=['REAL_HISTORY_REFERENCE','OP_CPFM_ON','OP_CPFM_OFF','FULL_B16V1_HIST','PLAIN_SD_HIST']
    features={b:load_features(root,b) for b in branches};rows=[];mcrows=[]
    target=root/'metrics/QUALITY_DISTRIBUTION_METRICS_V1.csv'
    existing=pd.read_csv(target).to_dict('records') if target.exists() else []
    for scope in ['all_available_common_support','patient_balanced_common_support','all_available_native_support_sensitivity']:
        subsets={}
        for b,(q,z) in features.items():
            select=q.anonymous_patient_key.isin(common).to_numpy() if scope!='all_available_native_support_sensitivity' else np.ones(len(q),bool)
            if scope=='patient_balanced_common_support':select &= q.patient_balanced_selected.to_numpy()
            subsets[b]=(q[select],z[select])
        qr,x=subsets['REAL_HISTORY_REFERENCE']
        for branch in branches[1:]:
            if scope=='all_available_native_support_sensitivity' and branch not in ['OP_CPFM_ON','OP_CPFM_OFF']:continue
            prior=[r for r in existing if r['scope']==scope and r['branch']==branch]
            if prior:rows.append(prior[0]);continue
            qg,y=subsets[branch]
            print(f'FID_KID {scope} {branch} N_REAL={len(x)} N_GEN={len(y)}',flush=True)
            f=fid(x,y);k,sd,vs=kid_mc(x,y)
            row=dict(scope=scope,branch=branch,reference='REAL_HISTORY_REFERENCE',fid=f,kid_mc_mean=k,kid_mc_sd=sd,
                     real_images=len(x),generated_images=len(y),real_patients=qr.anonymous_patient_key.nunique(),generated_patients=qg.anonymous_patient_key.nunique(),
                     feature_dim=2048,subsets=50,subset_size=512,kernel='degree3 gamma1/2048 coef1 unbiased',
                     covariance_rank_limited=min(len(x),len(y))<=2048,fid_ci_status='NOT_RUN_RESOURCE_BOUNDED_DECLARATION',
                     uncertainty='KID_MC_SD_NOT_PATIENT_CI',role='PRIMARY_DESCRIPTIVE' if branch.startswith('OP_') else 'AUXILIARY_NOT_SINGLE_VARIABLE',status='MEASURED')
            rows.append(row);write_csv(target,rows)
            atomic_json(root/'metrics'/f'KID_MC_{scope}_{branch}_V1.json',dict(values=vs,seed=20261012))
    # Reorder patient-balanced features by common patient then frozen image ID.
    balanced={};m=cfg['balanced_policy']['m']
    for b in ['REAL_HISTORY_REFERENCE','OP_CPFM_ON','OP_CPFM_OFF']:
        q,z=features[b];q=q.copy();q['position']=np.arange(len(q));g=q[q.patient_balanced_selected].sort_values(['anonymous_patient_key','image_id'])
        assert g.groupby('anonymous_patient_key').size().eq(m).all() and sorted(g.anonymous_patient_key.unique())==common
        balanced[b]=z[g.position.to_numpy()]
    ci=paired_balanced_kid_bootstrap(balanced['REAL_HISTORY_REFERENCE'],balanced['OP_CPFM_ON'],balanced['OP_CPFM_OFF'],len(common),m)
    write_csv(root/'metrics/QUALITY_PATIENT_CLUSTER_BOOTSTRAP_V1.csv',[ci])
    state(root,'FID_KID_AND_PATIENT_CLUSTER_DISTANCE_COMPLETED',comparisons=len(rows),bootstrap_repeats=2000)
    print({'comparisons':len(rows),'balanced_KID_delta':ci},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',required=True);p.add_argument('--toy-only',action='store_true')
    a=p.parse_args();main(a.output_root,a.toy_only)
