#!/usr/bin/env python3
"""Stateless pairwise ZipMap pose evaluation: mono pair or stereo pair."""
from __future__ import annotations
import argparse, csv, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import torch

EXT={'.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp'}
def b(v):
    if isinstance(v,bool): return v
    s=v.lower();
    if s in {'1','true','yes','y'}: return True
    if s in {'0','false','no','n'}: return False
    raise argparse.ArgumentTypeError(v)
def ap(x): return Path(x).expanduser().resolve()
def mod(name,path):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); sys.modules[name]=m; s.loader.exec_module(m); return m
def js(path,obj): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def csvout(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: path.write_text('',encoding='utf-8'); return
    keys=[]
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
def imgs(root,recursive=False):
    it=root.rglob('*') if recursive else root.iterdir(); x=sorted(p for p in it if p.is_file() and p.suffix.lower() in EXT)
    if not x: raise FileNotFoundError(root)
    return x
def sync(d):
    if d.type=='cuda': torch.cuda.synchronize(d)
def timed(d,fn):
    sync(d); t=time.perf_counter(); y=fn(); sync(d); return y,time.perf_counter()-t
def np0(x):
    a=x.detach().cpu().float().numpy(); return a[0] if a.ndim and a.shape[0]==1 else a

def load_model(a,ex,d):
    sys.path.insert(0,str(ap(a.zipmap_repo))); from zipmap.models.ZipMap import ZipMap
    cfg=json.loads(json.dumps(ex.MODEL_CONFIG)); cfg['other_config']['affine_invariant']=a.affine_invariant
    if a.pose_only_heads: cfg['enable_local_point']=False; cfg['enable_depth']=False
    model=ZipMap(**cfg); ck=torch.load(ap(a.zipmap_ckpt),map_location='cpu')
    sd=ck['ema'] if a.ema and isinstance(ck,dict) and 'ema' in ck else ck.get('model',ck) if isinstance(ck,dict) else ck
    inc=model.load_state_dict(sd,strict=False); model.eval().to(d)
    return model,{'config':cfg,'missing':list(inc.missing_keys),'unexpected':list(inc.unexpected_keys)}

def infer(model,paths,a,d,ex):
    from zipmap.utils.load_fn import load_and_preprocess_images
    from zipmap.utils.pose_enc import pose_encoding_to_extri_intri
    x,tp=timed(d,lambda:load_and_preprocess_images([str(p) for p in paths],target_size=a.target_size,mode=a.preprocess_mode).to(d))
    dtype=torch.bfloat16 if torch.cuda.get_device_capability()[0]>=8 else torch.float16
    def f():
        with torch.no_grad(),torch.amp.autocast('cuda',dtype=dtype): return model(x,store_state=False)
    pred,ti=timed(d,f)
    (e,k),td=timed(d,lambda:pose_encoding_to_extri_intri(pred['pose_enc'],x.shape[-2:]))
    w=ex.to_homogeneous_4x4(np0(e).astype(np.float32)); w=ex.align_w2c_to_first_camera(w)
    return ex.invert_se3_np(w).astype(np.float32),np0(k).astype(np.float32),{'preprocess':tp,'infer':ti,'decode':td,'total':tp+ti+td}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--zipmap_repo',required=True); p.add_argument('--zipmap_ckpt',required=True)
    p.add_argument('--left_dir',required=True); p.add_argument('--right_dir'); p.add_argument('--gt_pose_file',required=True); p.add_argument('--output_dir',required=True)
    p.add_argument('--mode',choices=['mono_pair','stereo_pair'],required=True); p.add_argument('--start_index',type=int,default=0); p.add_argument('--end_index',type=int)
    p.add_argument('--stride',type=int,default=1); p.add_argument('--max_frames',type=int); p.add_argument('--recursive',action='store_true'); p.add_argument('--device',default='cuda:0')
    p.add_argument('--ema',type=b,default=False); p.add_argument('--affine_invariant',type=b,default=True); p.add_argument('--pose_only_heads',type=b,default=True)
    p.add_argument('--target_size',type=int,default=518); p.add_argument('--preprocess_mode',choices=['crop','pad'],default='crop')
    p.add_argument('--stereo_baseline',type=float,default=0.25000006); p.add_argument('--scale_aggregation',choices=['mean','median','first','second'],default='mean')
    p.add_argument('--gt_convention',choices=['opencv_c2w','resplat_tartanair_pose'],default='resplat_tartanair_pose'); p.add_argument('--gt_quat_order',choices=['xyzw','wxyz'],default='xyzw')
    p.add_argument('--gt_matrix_convention',choices=['c2w','w2c'],default='c2w'); p.add_argument('--rpe_delta',type=int,default=1)
    a=p.parse_args(); out=ap(a.output_dir); out.mkdir(parents=True,exist_ok=True); zr=ap(a.zipmap_repo)
    ex=mod('pw_export',zr/'tools/export_zipmap_predictions.py'); ev=mod('pw_eval',zr/'tools/evaluate_zipmap_pose.py')
    L=imgs(ap(a.left_dir),a.recursive); R=imgs(ap(a.right_dir),a.recursive) if a.mode=='stereo_pair' else None
    if R is not None and len(L)!=len(R): raise ValueError('left/right count mismatch')
    end=len(L) if a.end_index is None else min(len(L),a.end_index); ids=list(range(max(0,a.start_index),end,a.stride)); ids=ids[:a.max_frames] if a.max_frames else ids
    if len(ids)<2: raise ValueError('need >=2 frames')
    d=torch.device(a.device)
    if d.type!='cuda' or not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    model,load=load_model(a,ex,d); T=np.repeat(np.eye(4)[None],len(ids),axis=0); rows=[]; locals=[]; Ks=[]; scales=[]; total={'preprocess':0.,'infer':0.,'decode':0.,'total':0.}
    for i in range(len(ids)-1):
        paths=[L[ids[i]],L[ids[i+1]]] if a.mode=='mono_pair' else [L[ids[i]],R[ids[i]],L[ids[i+1]],R[ids[i+1]]]
        P,K,tm=infer(model,paths,a,d,ex); delta=P[1 if a.mode=='mono_pair' else 2].astype(np.float64); s=1.; b0=b1=s0=s1=None
        if a.mode=='stereo_pair':
            b0=float(np.linalg.norm(P[1,:3,3])); rel=np.linalg.inv(P[2].astype(np.float64))@P[3].astype(np.float64); b1=float(np.linalg.norm(rel[:3,3]))
            if min(b0,b1)<=1e-6: raise RuntimeError(f'near-zero predicted baseline at pair {i}')
            s0=a.stereo_baseline/b0; s1=a.stereo_baseline/b1
            s={'mean':(s0+s1)/2,'median':float(np.median([s0,s1])),'first':s0,'second':s1}[a.scale_aggregation]
        delta[:3,3]*=s; T[i+1]=T[i]@delta; scales.append(s); locals.append(P); Ks.append(K)
        for k in total: total[k]+=tm[k]
        rows.append({'pair':i,'frame_t':ids[i],'frame_t1':ids[i+1],'scale':s,'baseline_t':b0,'baseline_t1':b1,'scale_t':s0,'scale_t1':s1,**tm})
        print(f'[{i+1}/{len(ids)-1}] {ids[i]}->{ids[i+1]} scale={s:.6f} time={tm["total"]:.4f}s',flush=True)
    GTall=ev.load_gt_trajectory(ap(a.gt_pose_file),a.gt_convention,a.gt_quat_order,a.gt_matrix_convention); GT=GTall[np.asarray(ids)]
    ev.write_pose_tum(out/'trajectory_raw_c2w_opencv.txt',T,names=[L[j].name for j in ids]); ev.write_pose_tum(out/'trajectory_gt_matched_c2w_opencv.txt',GT,names=[L[j].name for j in ids])
    E={}; arrays={'T_raw_accumulated_c2w_opencv':T.astype(np.float32),'T_gt_matched_c2w_opencv':GT.astype(np.float32),'T_pair_local_c2w_opencv':np.stack(locals),'pair_intrinsics':np.stack(Ks),'selected_original_indices':np.asarray(ids),'scale_per_pair':np.asarray(scales,dtype=np.float32)}
    for name,ws in [('se3',False),('sim3',True)]:
        s,Q,t=ev.umeyama_alignment(T[:,:3,3],GT[:,:3,3],with_scale=ws); A=ev.apply_similarity_to_poses(T,s,Q,t); ate=ev.compute_ate(A,GT); rr,rpe=ev.compute_rpe(A,GT,a.rpe_delta)
        E[name]={'alignment_scale':s,'alignment_rotation':Q.tolist(),'alignment_translation':t.tolist(),'ate':ate,'rpe':rpe}; arrays[f'T_{name}_aligned_c2w_opencv']=A.astype(np.float32)
        ev.write_pose_tum(out/f'trajectory_{name}_aligned_c2w_opencv.txt',A,names=[L[j].name for j in ids]); csvout(out/f'rpe_errors_{name}.csv',rr)
    np.savez_compressed(out/'pairwise_pose_results.npz',**arrays); csvout(out/'pairwise_steps.csv',rows)
    ss=np.asarray(scales); summary={'mode':a.mode,'pairwise_state_policy':'independent forward per adjacent pair; returned TTT state discarded','num_frames':len(ids),'num_pairs':len(ids)-1,'selected_original_indices':ids,'timing':{**total,'total_per_pair':total['total']/(len(ids)-1)},'scale_statistics':{'mean':float(ss.mean()),'median':float(np.median(ss)),'std':float(ss.std()),'min':float(ss.min()),'max':float(ss.max())},'evaluation':E,'model':load,'plots_generated':False}
    js(out/'summary.json',summary); print(f"SE3 ATE={E['se3']['ate']['rmse']:.6f} Sim3 ATE={E['sim3']['ate']['rmse']:.6f}")
if __name__=='__main__': main()
