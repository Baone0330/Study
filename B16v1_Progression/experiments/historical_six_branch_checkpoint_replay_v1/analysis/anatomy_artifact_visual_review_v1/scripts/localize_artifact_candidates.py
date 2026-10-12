"""Reproduce prior per-image flags; append CPU-only geometric localization, not clinical truth."""
from __future__ import annotations
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from review_common import setup,read_json,parquet_out,csv_out,json_out,state,now


def rgb_read(path):
    with Image.open(path) as im:
        im.load();mode=im.mode;a=np.asarray(im.convert('RGB')).copy()
    return a,mode


def gray(a):return a[...,0].astype(np.float64)*.299+a[...,1]*.587+a[...,2]*.114


def zone_points(x,y):
    x=np.asarray(x);y=np.asarray(y);border=(x<26)|(x>=486)|(y<26)|(y>=486)
    central=(x>=102)&(x<410)&(y>=102)&(y<410)
    codes=np.where(border,'BORDER',np.where(central,'CENTRAL_FIELD','INTERMEDIATE'))
    return sorted(set(codes.tolist()))


def line_geometry(raw):
    out=[]
    for i,(x1,y1,x2,y2) in enumerate(raw):
        dx=int(x2-x1);dy=int(y2-y1);length=float(np.hypot(dx,dy));n=int(np.ceil(length))+1
        t=np.linspace(0,1,n);x=np.rint(x1+t*dx).astype(int);y=np.rint(y1+t*dy).astype(int);zones=zone_points(x,y)
        out.append(dict(candidate_id=i,x1=int(x1),y1=int(y1),x2=int(x2),y2=int(y2),length_px=length,
                        angle_deg_0to180=float(np.degrees(np.arctan2(dy,dx))%180),
                        bbox=[int(min(x1,x2)),int(min(y1,y2)),int(max(x1,x2)),int(max(y1,y2))],zones=zones,
                        candidate_location='MULTIZONE' if len(zones)>1 else zones[0]))
    return out


def original_crosses(raw,lines):
    hs=[];vs=[];tol=np.tan(np.deg2rad(10))
    for i,(x1,y1,x2,y2) in enumerate(raw):
        dx,dy=abs(x2-x1),abs(y2-y1)
        if dy<=tol*dx:hs.append((i,min(x1,x2),max(x1,x2),(y1+y2)/2))
        if dx<=tol*dy:vs.append((i,min(y1,y2),max(y1,y2),(x1+x2)/2))
    out=[]
    for hi,xlo,xhi,y in hs:
        for vi,ylo,yhi,x in vs:
            if xlo<=x<=xhi and ylo<=y<=yhi:
                a,b=lines[hi],lines[vi];zones=sorted(set(a['zones'])|set(b['zones']))
                bbox=[min(a['bbox'][0],b['bbox'][0]),min(a['bbox'][1],b['bbox'][1]),max(a['bbox'][2],b['bbox'][2]),max(a['bbox'][3],b['bbox'][3])]
                out.append(dict(candidate_id=len(out),horizontal_line_id=hi,vertical_line_id=vi,x=float(x),y=float(y),bbox=bbox,
                                zones=zones,crossing_point_zone=zone_points([x],[y])[0],candidate_location='MULTIZONE' if len(zones)>1 else zones[0]))
    return out


def diagnose_array(rgb):
    a=np.asarray(Image.fromarray(rgb).resize((512,512),Image.Resampling.BILINEAR));g=gray(a);u=np.clip(np.rint(g),0,255).astype(np.uint8)
    edges=cv2.Canny(u,50,150);found=cv2.HoughLinesP(edges,1,np.pi/180,threshold=50,minLineLength=128,maxLineGap=10)
    raw=np.empty((0,4),dtype=int) if found is None else found[:,0]
    lines=line_geometry(raw);crosses=original_crosses(raw,lines)
    result=dict(reproduced_long_line_flag=found is not None,reproduced_cross_proxy_flag=bool(crosses),line_candidates=len(lines),
                cross_candidates=len(crosses),line_total_length_px=float(sum(d['length_px'] for d in lines)),
                line_max_length_px=float(max([d['length_px'] for d in lines],default=0)),
                line_candidates_private_json=json.dumps(lines),cross_candidates_private_json=json.dumps(crosses),
                observer_role='MACHINE_PROXY',localizer_version='LOCALIZATION_DIAGNOSTIC_V1',anatomical_ROI=False)
    for kind,objects in [('line',lines),('cross',crosses)]:
        for zone,name in [('BORDER','border'),('CENTRAL_FIELD','central'),('INTERMEDIATE','intermediate')]:
            result[kind+'_'+name+'_candidate_count']=sum(zone in d['zones'] for d in objects)
            result[kind+'_'+name+'_any']=any(zone in d['zones'] for d in objects)
        result[kind+'_multizone_candidate_count']=sum(d['candidate_location']=='MULTIZONE' for d in objects)
        result[kind+'_multizone_any']=any(d['candidate_location']=='MULTIZONE' for d in objects)
    return result


def worker(r):
    try:
        a,mode=rgb_read(r['png_abs_path']);d=diagnose_array(a)
        keys=['image_id','branch','anonymous_patient_key','anonymous_node_key','real_exam_key_private']
        out={k:r[k] for k in keys};out.update(d,decode_status='PASS',native_width=a.shape[1],native_height=a.shape[0],native_mode=mode,
                             old_long_line_flag=bool(r['long_line_flag']),old_cross_proxy_flag=bool(r['cross_proxy_flag']),old_line_count=int(r['long_line_count']))
        out['exact_old_flag_match']=out['old_long_line_flag']==d['reproduced_long_line_flag'] and out['old_cross_proxy_flag']==d['reproduced_cross_proxy_flag'] and out['old_line_count']==d['line_candidates']
        return out
    except Exception as error:return dict(image_id=r['image_id'],branch=r['branch'],decode_status='FAILED',error_type=type(error).__name__,error_detail_private=repr(error))


def initializer():cv2.setNumThreads(1)


def main(output_root,workers=6):
    root=setup(output_root);c=read_json(root/'protocol/FROZEN_VISUAL_AND_ARTIFACT_REVIEW_CONTRACT_V1.json');old=c['quality_root_private']
    q=pd.read_parquet(root/'manifests/PRIVATE_UPSTREAM_IMAGE_QUEUE_V1.parquet');q=q[q.branch.isin(c['localizer_branches'])].copy()
    qa=pd.read_parquet(old+'/metrics/IMAGE_DECODE_AND_TECHNICAL_QA_V1.parquet',columns=['image_id','long_line_flag','cross_proxy_flag','long_line_count'])
    q=q.merge(qa,on='image_id',validate='1:1');assert len(q)==31740
    csv_out(root/'metrics/EXISTING_FLAG_REUSE_AND_DEFINITION_V1.csv',[dict(parameter=k,definition=v,version='UPSTREAM_DEFINITION_REUSED_UNCHANGED') for k,v in c['original_flag_definition'].items()])
    cache=root/'cache/localization';paths=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=initializer) as pool:
        for start in range(0,len(q),256):
            path=cache/f'{start:06d}.parquet';ledger=cache/f'{start:06d}.ledger.json';batch=q.iloc[start:start+256]
            if path.exists():
                d=pd.read_parquet(path);prior=read_json(ledger);assert d.image_id.tolist()==batch.image_id.tolist()
                assert prior['cv2_version']==cv2.__version__ and prior['contract_freeze_utc']==c['started_utc'] and prior['status']=='COMPLETE'
            else:
                d=pd.DataFrame(pool.map(worker,batch.to_dict('records'),chunksize=8));parquet_out(path,d)
                valid=d.decode_status.eq('PASS').all() and 'exact_old_flag_match' in d and d.exact_old_flag_match.all()
                json_out(ledger,dict(status='COMPLETE' if valid else 'FAILED_PRESERVED',images=len(d),contract_freeze_utc=c['started_utc'],cv2_version=cv2.__version__,completed_utc=now()))
                if not valid:
                    state(root,'LOCATION_OR_FLAG_REPRODUCTION_FAILED',chunk_start=start,failed_rows=int((d.decode_status!='PASS').sum()))
                    raise RuntimeError('Private diagnostic chunk contains source failure or original-flag mismatch; no sample/threshold changes allowed')
                print(f'LOCALIZATION {min(start+256,len(q))}/{len(q)} images',flush=True)
            paths.append(path)
    out=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True);assert out.image_id.tolist()==q.image_id.tolist() and out.exact_old_flag_match.all()
    parquet_out(root/'metrics/FULL_QUEUE_FLAG_LOCATION_DIAGNOSTICS_V1.parquet',out)
    counts=[]
    for b,g in out.groupby('branch',sort=False):counts.append(dict(branch=b,images_read_and_decoded_this_stage=len(g),old_long_line_hits=int(g.old_long_line_flag.sum()),
                      old_cross_hits=int(g.old_cross_proxy_flag.sum()),old_flag_mismatches=int((~g.exact_old_flag_match).sum()),actual_line_candidates=int(g.line_candidates.sum()),
                      actual_cross_candidates=int(g.cross_candidates.sum()),patients=g.anonymous_patient_key.nunique()))
    csv_out(root/'metrics/ORIGINAL_FLAG_FULL_QUEUE_REPRODUCTION_V1.csv',counts)
    json_out(root/'release/LOCALIZATION_COMPLETE_V1.json',dict(status='COMPLETE',actual_images_read_and_decoded=len(out),old_flag_mismatches=0,workers=workers,
                       gpu_used=False,new_FID_KID=0,medical_visual_review='NOT_RUN',completed_utc=now()))
    state(root,'FULL_QUEUE_TECHNICAL_LOCALIZATION_COMPLETE',images=len(out),old_flag_mismatches=0);print(counts,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',required=True);p.add_argument('--workers',type=int,default=6);a=p.parse_args();main(a.output_root,a.workers)
