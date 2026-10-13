"""Deterministic FP32 support and small auditable evaluation primitives."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,confusion_matrix,accuracy_score,precision_score,recall_score,f1_score


def configure(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    torch.backends.mha.set_fastpath_enabled(False);torch.use_deterministic_algorithms(True)


def worker_init(_):torch.set_num_threads(1)


def metric_values(y,p,threshold=None):
    y=np.asarray(y,int);p=np.asarray(p,float);assert np.isfinite(p).all() and len(y)==len(p)
    if threshold is None:
        fpr,tpr,t=roc_curve(y,p);valid=np.isfinite(t);j=tpr[valid]-fpr[valid];threshold=float(t[valid][int(np.argmax(j))])
    pred=p>=threshold;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(N=len(y),ROC_AUC=float(roc_auc_score(y,p)),AP=float(average_precision_score(y,p)),threshold=float(threshold),
          Accuracy=float(accuracy_score(y,pred)),Precision=float(precision_score(y,pred,zero_division=0)),Sensitivity=float(recall_score(y,pred,zero_division=0)),
          Specificity=float(tn/(tn+fp)),F1=float(f1_score(y,pred,zero_division=0)),TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp))


def lr_for_epoch(epoch,c):
    return c['lr']*epoch/c['warmup_epoch'] if epoch<=c['warmup_epoch'] else c['lr']*.5*(1+np.cos(np.pi*(epoch-c['warmup_epoch'])/(c['max_epoch']-c['warmup_epoch'])))


def selection_key(metrics,loss,epoch):return (metrics['ROC_AUC'],metrics['AP'],-float(loss),-int(epoch))


def rng_state(shuffle_generator):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all(),shuffle=shuffle_generator.get_state())


def restore_rng(state,shuffle_generator):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch_cpu']);torch.cuda.set_rng_state_all(state['torch_cuda']);shuffle_generator.set_state(state['shuffle'])


def save_checkpoint(path,value):
    temp=path.with_name(path.name+'.tmp');torch.save(value,temp);os.replace(temp,path)


def paired_bootstrap(y,real,op,repeats=2000,seed=20261013):
    y=np.asarray(y);real=np.asarray(real);op=np.asarray(op);assert len(y)==len(real)==len(op)
    rng=np.random.default_rng(seed);rows=[];skipped=0
    for _ in range(repeats):
        i=rng.integers(0,len(y),len(y))
        if len(np.unique(y[i]))<2:skipped+=1;continue
        rows.append([roc_auc_score(y[i],op[i])-roc_auc_score(y[i],real[i]),average_precision_score(y[i],op[i])-average_precision_score(y[i],real[i])])
    a=np.asarray(rows);out=[]
    for j,name in enumerate(['ROC_AUC','AP']):
        original=(roc_auc_score(y,op)-roc_auc_score(y,real)) if j==0 else (average_precision_score(y,op)-average_precision_score(y,real))
        lo,hi=np.quantile(a[:,j],[.025,.975]);out.append(dict(metric=name,delta_OP_minus_REAL=float(original),ci_low=float(lo),ci_high=float(hi),
                   paired_patients=len(y),bootstrap_requested=repeats,bootstrap_valid=len(a),bootstrap_skipped_one_class=skipped,bootstrap_seed=seed))
    return out
