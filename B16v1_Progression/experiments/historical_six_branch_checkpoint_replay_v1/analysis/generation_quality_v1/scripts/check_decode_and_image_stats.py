"""Full frozen-queue decode, prespecified whole-image proxies, and pixel duplicates."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from b16_quality_common import setup, read_json, atomic_parquet, atomic_json, state, now


def gray(rgb):
    return rgb[...,0].astype(np.float64)*0.299 + rgb[...,1]*0.587 + rgb[...,2]*0.114


def native_rgb(path):
    with Image.open(path) as im:
        im.load()
        mode=im.mode
        rgb=np.asarray(im.convert('RGB')).copy()
    return rgb, mode


def texture(rgb):
    g=gray(rgb); u=np.clip(np.rint(g),0,255).astype(np.uint8)
    sx=cv2.Sobel(g,cv2.CV_64F,1,0,ksize=3); sy=cv2.Sobel(g,cv2.CV_64F,0,1,ksize=3)
    lap=cv2.Laplacian(g,cv2.CV_64F,ksize=3)
    blocks=g.reshape(16,32,16,32).transpose(0,2,1,3).reshape(256,1024)
    local=blocks.std(axis=1)
    ps=np.abs(np.fft.fft2((g-g.mean())*np.outer(np.hanning(512),np.hanning(512))))**2
    fr=np.fft.fftfreq(512); radius=np.sqrt(fr[:,None]**2+fr[None,:]**2); ps[0,0]=0
    total=ps.sum(); hf=float(ps[radius>=.25].sum()/total) if total>0 else 0.0
    border=np.ones((512,512),dtype=bool); border[26:-26,26:-26]=False
    edges=cv2.Canny(u,50,150)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=50,minLineLength=128,maxLineGap=10)
    horizontal=[]; vertical=[]
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            dx,dy=abs(x2-x1),abs(y2-y1)
            if dy <= np.tan(np.deg2rad(10))*dx: horizontal.append((min(x1,x2),max(x1,x2),(y1+y2)/2))
            if dx <= np.tan(np.deg2rad(10))*dy: vertical.append((min(y1,y2),max(y1,y2),(x1+x2)/2))
    cross=any(xlo<=x<=xhi and ylo<=y<=yhi for xlo,xhi,y in horizontal for ylo,yhi,x in vertical)
    p1,p99=np.percentile(g,[1,99]); black=float((g<5).mean()); white=float((g>250).mean())
    return dict(gray_mean=float(g.mean()),gray_std=float(g.std()),gray_p01=float(p1),gray_p99=float(p99),
                dynamic_range_p99_p01=float(p99-p1),local_std_median=float(np.median(local)),
                local_std_q25=float(np.quantile(local,.25)),local_std_q75=float(np.quantile(local,.75)),
                sobel_gradient_mean=float(np.hypot(sx,sy).mean()),laplacian_variance=float(lap.var()),
                high_frequency_fraction=hf,black_fraction=black,white_fraction=white,
                border_black_fraction=float((g[border]<5).mean()),border_edge_fraction=float((edges[border]>0).mean()),
                edge_fraction=float((edges>0).mean()),long_line_count=0 if lines is None else len(lines),
                low_contrast_flag=bool(g.std()<10),very_smooth_flag=bool(np.median(local)<3),
                extreme_black_flag=black>.5,extreme_white_flag=white>.1,high_frequency_flag=hf>.5,
                long_line_flag=lines is not None,cross_proxy_flag=bool(cross))


def worker(r):
    try:
        rgb,mode=native_rgb(r['png_abs_path']); h,w=rgb.shape[:2]; g=gray(rgb)
        sample=rgb[np.linspace(0,h-1,9,dtype=int)[:,None],np.linspace(0,w-1,9,dtype=int)[None,:]].tobytes()
        resized=np.asarray(Image.fromarray(rgb).resize((512,512),Image.Resampling.BILINEAR))
        out={k:r[k] for k in ['image_id','branch','anonymous_patient_key','anonymous_node_key','real_exam_key_private']}
        out.update(decode_status='PASS',native_width=w,native_height=h,native_mode=mode,native_dtype=str(rgb.dtype),
                   finite=bool(np.isfinite(rgb).all()),native_gray_mean=float(g.mean()),native_gray_std=float(g.std()),
                   raw_sampled_pixel_array=sample,**texture(resized))
        return out
    except Exception as e:
        return {'image_id':r['image_id'],'branch':r['branch'],'decode_status':'FAILED','error':repr(e)}


def init_worker(): cv2.setNumThreads(1)


def duplicate_check(root,q,stats):
    out=root/'metrics/EXACT_PIXEL_DUPLICATE_PROOFS_V1.parquet'
    if out.exists(): return pd.read_parquet(out)
    paths=dict(zip(q.image_id,q.png_abs_path)); proofs=[]; candidates=0
    for _,group in stats.groupby(['native_width','native_height','raw_sampled_pixel_array'],sort=False):
        if len(group)<2: continue
        # Sample arrays can only miss no exact-equal pair; equality needs all native pixels.
        representatives=[]
        for r in group.itertuples():
            a,_=native_rgb(paths[r.image_id]); found=False
            for key,branch,b in representatives:
                candidates+=1
                if np.array_equal(a,b):
                    proofs.append(dict(image_id=r.image_id,equal_to_image_id=key,branch=r.branch,equal_to_branch=branch,
                                       proof='FULL_NATIVE_RGB_ARRAY_EQUAL',retained=True));found=True;break
            if not found: representatives.append((r.image_id,r.branch,a))
    d=pd.DataFrame(proofs,columns=['image_id','equal_to_image_id','branch','equal_to_branch','proof','retained'])
    atomic_parquet(out,d)
    atomic_json(root/'metrics/EXACT_PIXEL_DUPLICATE_AUDIT_V1.json',dict(status='FULL_QUEUE_CHECKED',images=len(q),
                candidate_full_array_checks=candidates,duplicate_images_beyond_first=len(d),no_hashes=True,no_image_removal=True))
    return d


def main(output_root,workers=6):
    root=setup(output_root); read_json(root/'protocol/PREMEASUREMENT_EXTENSION_V1.json')
    q=pd.read_parquet(root/'manifests/COMBINED_QUALITY_EVAL_FROZEN_QUEUE_V1.parquet')
    target=root/'metrics/IMAGE_DECODE_AND_TECHNICAL_QA_V1.parquet'; cache=root/'cache/decode_chunks';cache.mkdir(exist_ok=True)
    if target.exists(): stats=pd.read_parquet(target)
    else:
        with ProcessPoolExecutor(max_workers=workers,initializer=init_worker) as pool:
            for i in range(0,len(q),500):
                dest=cache/f'{i:06d}.parquet';part=q.iloc[i:i+500]
                if dest.exists():
                    old=pd.read_parquet(dest);assert old.image_id.tolist()==part.image_id.tolist();continue
                d=pd.DataFrame(pool.map(worker,part.to_dict('records'),chunksize=8));atomic_parquet(dest,d)
                if not (d.decode_status=='PASS').all():
                    state(root,'DECODE_FAILED',failed=int((d.decode_status!='PASS').sum()),chunk=i)
                    raise RuntimeError('Source decoding failure preserved in private chunk; no image exclusion')
                print(f'DECODE {min(i+500,len(q))}/{len(q)}',flush=True)
        stats=pd.concat([pd.read_parquet(x) for x in sorted(cache.glob('*.parquet'))],ignore_index=True)
        assert stats.image_id.tolist()==q.image_id.tolist() and (stats.decode_status=='PASS').all()
        gen=stats[stats.branch!='REAL_HISTORY_REFERENCE'];assert (gen.native_width==512).all() and (gen.native_height==512).all()
        atomic_parquet(target,stats)
    d=duplicate_check(root,q,stats)
    state(root,'DECODE_AND_TECHNICAL_QA_COMPLETED',images=len(stats),decode_failed=0,duplicate_images=len(d))
    print({'decoded':len(stats),'duplicates_beyond_first':len(d)},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-root',required=True);p.add_argument('--workers',type=int,default=6)
    a=p.parse_args();main(a.output_root,a.workers)
