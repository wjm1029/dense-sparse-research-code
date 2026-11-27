"""Dense6 120k + original joint Sparse13k real-video inference, server only."""
import argparse,gc,hashlib,json,os,shutil,subprocess,sys,time,traceback,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from safetensors import safe_open
from safetensors.torch import load_file
from infer_wan22_dense4 import encode_video,decode_video,load_official_vae,read_video,rgb_frames,task_frames
from train_wan22_dense6_ddp import Dense6LoRAModule
from wan22_dense6_task_router import TASK_TO_ID
from wan22_dense6_checkpoint import checkpoint_metadata,build_provenance_metadata,load_dense6_checkpoint
from train_wan22_sparse_full_ddp import SparseModule,PROMPT
from diffsynth.pipelines.wan_video import WanVideoUnit_PromptEmbedder
BASE=Path('/pfs/mayuema/dense/wan'); REPO=BASE/'DiffSynth-Studio'; HERE=Path(__file__).resolve().parent
INPUT=BASE/'inputs/eightmodal_tianj300_20261018'; OUTPUT=BASE/'outputs/eightmodal_tianj300_dense120k_sparse13k_20261018'
DENSE=BASE/'outputs/wan22_dense6_flow32_dispersion_100k_to120k_20261012/step-120000.safetensors'
SPARSE=BASE/'outputs/wan22_sparse_lora_bidir_full_4gpu_overnight_20261012/step-13000.recovered.safetensors'
MODEL=Path('/dev/shm/eightmodal_tianj300_20261018/model')
SCRATCH=Path('/dev/shm/eightmodal_tianj300_20261018/results')
TASK_PROMPTS={'depth':'output: depth','normal':'output: surface normal','raymap':'output: camera raymap','albedo':'output: albedo','optical_flow':'output: optical flow'}
MODES=['rgb','depth','normal','matte','raymap','albedo','optical_flow','keypoints_2d','keypoints_3d']
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()
def save_json(p,value):
    p=Path(p); cache=SCRATCH/p.relative_to(OUTPUT); cache.parent.mkdir(parents=True,exist_ok=True)
    temp=cache.with_name(cache.name+'.pending'); temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)); os.replace(temp,cache); publish(cache,p)
def save_tensor(value,path):
    cache=SCRATCH/path.relative_to(OUTPUT); cache.parent.mkdir(parents=True,exist_ok=True)
    torch.save(value,cache); publish(cache,path)
def event(**data): print(json.dumps(data,ensure_ascii=False),flush=True)
def run(cmd,data=None):
    r=subprocess.run(cmd,input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    if r.returncode: raise RuntimeError(r.stderr.decode(errors='replace')[-6000:])
    return r.stdout

PUBLISHER=ThreadPoolExecutor(max_workers=2)
PUBLISH_FUTURES=[]
CASE_FUTURES={}
DIRECTORIES=set()
DIRECTORY_LOCK=threading.Lock()
def _publish_sync(source,destination,dependencies):
    for future in dependencies: future.result()
    with DIRECTORY_LOCK:
        if destination.parent not in DIRECTORIES:
            destination.parent.mkdir(parents=True,exist_ok=True);DIRECTORIES.add(destination.parent)
    pending=destination.with_name(destination.name+'.pending')
    shutil.copyfile(source,pending);os.replace(pending,destination)
    if destination.name.endswith('_COMPLETE.json'):
        published=source.with_name(source.name.replace('_COMPLETE.json','_PUBLISHED.json'));published.write_text(source.read_text())
def publish(source,destination):
    relative=destination.relative_to(OUTPUT);case=relative.parts[0] if len(relative.parts)>1 else None
    previous=CASE_FUTURES.setdefault(case,[])
    dependencies=list(previous) if destination.name.endswith('_COMPLETE.json') else []
    future=PUBLISHER.submit(_publish_sync,source,destination,dependencies)
    previous.append(future);PUBLISH_FUTURES.append(future)
def wait_case(case):
    for future in list(CASE_FUTURES.get(case,[])):future.result()
def wait_publications():
    for future in list(PUBLISH_FUTURES):future.result()
def cached_artifact(path):
    cached=SCRATCH/path.relative_to(OUTPUT); cached.parent.mkdir(parents=True,exist_ok=True)
    if not cached.exists(): shutil.copyfile(path,cached)
    return cached
def cached_source(c):
    cached=Path('/dev/shm/eightmodal_tianj300_20261018/preparation')/c['case_name']/'prepared.mp4'
    if not cached.exists():
        cached=SCRATCH/c['case_name']/'rgb.mp4'; cached.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(c['path'],cached)
    assert sha(cached)==c['preprocessed_sha256']
    return cached

def write_video(path,frames):
    assert frames.shape==(81,480,832,3) and frames.dtype==np.uint8,(path,frames.shape,frames.dtype)
    cache=SCRATCH/path.relative_to(OUTPUT); cache.parent.mkdir(parents=True,exist_ok=True)
    temp=cache.with_name(cache.stem+'.partial.mp4')
    run(['ffmpeg','-nostdin','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','832x480','-framerate','16','-i','-','-frames:v','81','-an','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(temp)],np.ascontiguousarray(frames).tobytes())
    os.replace(temp,cache); publish(cache,path)

def dense_init(device,cases):
    assert (MODEL/'STAGED_MODEL_VERIFIED.json').is_file()
    meta=checkpoint_metadata(DENSE); assert int(meta['step'])==120000
    native=json.loads(meta['output_prompt_texts']); extras=sorted({c['matte_prompt'] for c in cases}-set(native))
    module=Dense6LoRAModule(str(MODEL),int(meta['lora_rank']),meta['lora_targets'],int(meta['router_hidden_dim']),meta['router_mode'],0.)
    module.task_router.to(device=device,dtype=torch.float32)
    contexts=[]
    if extras:
        module.pipe.device=device; module.pipe.text_encoder.to(device)
        encoder=WanVideoUnit_PromptEmbedder()
        with torch.no_grad():
            for prompt in extras: contexts.append(encoder.encode_prompt(module.pipe,prompt).detach().cpu())
    module.prepare_condition_tables(native,device)
    provenance=build_provenance_metadata(module,model_dir=MODEL,prompt_texts=native,prompt_dir='/pfs/mayuema/dense/prompts',depth_cache_dir=BASE/'cache/wan22_depth_latents_bf16_bright40',target_cache_dir=BASE/'cache/wan22_dense4_targets_bf16_bright40',code_patch_path=BASE/'docs/wan22_dense6_bidir_flow32_runtime_source.patch',repo_dir=REPO,auxiliary_cache_dirs={'albedo_delayed_flow':BASE/'cache/wan22_albedo_delayed_flow32_targets_v1_bf16_bright40'})
    if provenance['diffsynth_git_head']!=meta['diffsynth_git_head']:
        subprocess.run(['git','merge-base','--is-ancestor',meta['diffsynth_git_head'],provenance['diffsynth_git_head']],cwd=REPO,check=True)
        for name in ('train_wan22_dense6_ddp.py','wan22_dense6_checkpoint.py','wan22_dense6_task_router.py','wan22_masked_loss.py','wan22_dense6_task_sampler.py'):
            subprocess.run(['git','diff','--exit-code',meta['diffsynth_git_head'],'--',str((HERE/name).relative_to(REPO))],cwd=REPO,check=True)
        provenance['diffsynth_git_head']=meta['diffsynth_git_head']
    module.pipe.dit.to(device).eval()
    loaded=load_dense6_checkpoint(module,DENSE,expected_provenance=provenance,lora_rank=int(meta['lora_rank']),lora_targets=meta['lora_targets'],router_hidden_dim=int(meta['router_hidden_dim']),expected_router_mode=meta['router_mode'])
    if contexts: module.context_table=torch.cat([module.context_table,torch.cat(contexts,dim=0).to(device=device,dtype=torch.bfloat16)],dim=0)
    del contexts; module.eval()
    vae=load_official_vae(BASE/'Wan2.2',MODEL,device)
    info={'checkpoint':str(DENSE),'sha256':sha(DENSE),'step':120000,'strict_provenance_checked':True,'provenance':provenance,'native_prompts':native,'extra_evaluation_prompts':extras,'dispersion_contract':json.loads(meta['dispersion_contract'])}
    return module,vae,{p:i for i,p in enumerate(native+extras)},info

def dense_case(c,module,vae,pids,info,device):
    d=OUTPUT/c['case_name']
    if (SCRATCH/c['case_name']/'DENSE_PUBLISHED.json').exists():
        assert json.loads((d/'dense_manifest.json').read_text())['checkpoint']['sha256']==info['sha256']; return
    started=time.monotonic(); path=cached_source(c); assert sha(path)==c['preprocessed_sha256']
    frames,fps=read_video(path); assert fps==16
    source=encode_video(vae,frames,device); assert tuple(source.shape)==(48,21,30,52)
    save_tensor({'source_latent':source,'source_video_sha256':c['preprocessed_sha256']},d/'source_latent.pt')
    rgb_cache=SCRATCH/c['case_name']/'rgb.mp4'; rgb_cache.parent.mkdir(parents=True,exist_ok=True)
    if path!=rgb_cache: shutil.copyfile(path,rgb_cache)
    publish(rgb_cache,d/'rgb.mp4'); rows=[]
    with torch.inference_mode():
        for task,tid in TASK_TO_ID.items():
            prompt=c['matte_prompt'] if task=='matte' else TASK_PROMPTS[task]
            try:
                velocity,gates=module.predict_velocity(source[None].to(device),torch.tensor([tid],device=device),torch.tensor([pids[prompt]],device=device))
            finally: module.task_router.clear()
            latent=(-velocity[0]).detach().cpu().to(torch.bfloat16).contiguous(); assert tuple(latent.shape)==(48,21,30,52) and torch.isfinite(latent).all()
            save_tensor(latent,d/(task+'_latent.pt'))
            video=decode_video(vae,latent,device)
            visual={}
            if task=='depth':
                from matplotlib import colormaps
                values=video.mean(0).clamp(-1,1).add(1).mul(.5).numpy()
                result=np.rint(colormaps['Spectral_r'](values)[...,:3]*255).astype(np.uint8)
                visual={'colormap':'Spectral_r','range':[-1,1],'near':'warm','far':'cool','normalization':'fixed, no per-frame/per-clip stretch','units':'relative inverse depth, not meters'}
            elif task=='optical_flow':
                import cv2
                arr=video.numpy(); u=np.clip(arr[0],-1,1)*32.; v=np.clip(arr[1],-1,1)*32.; valid=arr[2]>0
                hue=np.clip((np.arctan2(v,u)+np.pi)*90/np.pi,0,179).astype(np.uint8)
                mag=np.sqrt(u*u+v*v); hsv=np.stack([hue,np.full_like(hue,255),np.clip(mag/20*255,0,255).astype(np.uint8)],axis=-1)
                result=np.stack([cv2.cvtColor(f,cv2.COLOR_HSV2RGB) for f in hsv]); result[~valid]=0; result[0]=0
                visual={'scale_px':32,'components':'u=decoded[0]*32, v=decoded[1]*32; each clipped to +/-32px','valid':'decoded[2]>0','display':'HSV direction; magnitude brightness saturated at20px; invalid black','temporal':'delayed: frame t stores flow(t-1 -> t); frame0 has no predecessor and displays black','raw_prediction':'optical_flow_latent.pt retains complete unmodified predicted latent'}
                del arr,u,v,valid,hue,mag,hsv
            else: result=task_frames(video,task)
            write_video(d/(task+'.mp4'),result)
            row={'task':task,'prompt':prompt,'prompt_in_training_vocabulary':prompt in info['native_prompts'],'gate_min':float(gates.min()),'gate_max':float(gates.max()),'gate_mean':float(gates.mean()),'visualization':visual}
            rows.append(row); event(case=c['case_name'],stage='dense',task=task,elapsed_seconds=round(time.monotonic()-started,2))
            del velocity,gates,latent,video,result; torch.cuda.empty_cache()
    save_json(d/'dense_manifest.json',{'case':c,'checkpoint':info,'tasks':rows,'source_rgb_mean':float(frames.mean()),'below_training_brightness_threshold':float(frames.mean())<40,'frames':81,'fps':16,'driver_sha256':DRIVER_SHA,'elapsed_seconds':time.monotonic()-started})
    save_json(d/'DENSE_COMPLETE.json',{'complete':True,'tasks':list(TASK_TO_ID),'step':120000})

def sparse_init(device):
    with safe_open(str(SPARSE),framework='pt',device='cpu') as f: meta=f.metadata()
    assert int(meta['step'])==13000; contract=json.loads(meta['contract']); assert contract['prompt']==PROMPT and contract['attention']=='bidirectional'
    for file,key in [(HERE/'train_wan22_sparse_full_ddp.py','script_sha256'),(REPO/'diffsynth/pipelines/wan_video.py','pipeline_sha256'),(REPO/'diffsynth/models/wan_video_dit.py','dit_sha256')]: assert sha(file)==contract[key],file
    index=BASE/'cache/wan22_sparse_coordinates_v1_20261011/manifest.json'; assert sha(index)==contract['index_sha256']
    module=SparseModule(str(MODEL)); module.prepare(device)
    weights=load_file(str(SPARSE),device='cpu'); trainable={n:p for n,p in module.named_parameters() if p.requires_grad}
    assert set(weights)==set(trainable) and len(weights)==617
    with torch.no_grad():
        for name,value in weights.items():
            assert value.shape==trainable[name].shape and torch.isfinite(value).all(),name
            trainable[name].copy_(value.to(trainable[name]))
    del weights; module.eval()
    manifest=json.loads(index.read_text()); cid=manifest['ids'][0]
    data=Path(manifest['dataset_root'])/cid/'genception/keypoints_2d/keypoints_2d.npz'
    with np.load(data,allow_pickle=False) as z: parents=z['parents'].tolist()
    assert len(parents)==24
    info={'checkpoint':str(SPARSE),'sha256':sha(SPARSE),'step':13000,'contract':contract,'contract_and_all_617_tensors_checked':True,'joint_names':manifest['joint_names'],'parents':parents,'target_2d':'image_xy_01; multiply by [832,480] for pixels','target_3d':'root_relative_camera_per_height; camera X right,Y down,Z forward; not meters','prompt':PROMPT}
    return module,info

def skeleton_views(frames,xy,xyz,parents):
    colors=[(255,90,170),(60,220,250),(255,210,80),(135,240,110)]
    output2=[]; output3=[]; font=ImageFont.truetype(FONT,16)
    bounds=max(float(np.abs(xyz).max())*1.08,.6)
    for t in range(81):
        image=Image.fromarray(frames[t]); draw=ImageDraw.Draw(image); pts=xy[t]*np.array([832,480]); valid=((xy[t]>=0)&(xy[t]<=1)).all(-1)
        for j,p in enumerate(parents):
            if p>=0 and valid[j] and valid[p]: draw.line([tuple(pts[p]),tuple(pts[j])],fill=colors[j%4],width=3)
        for j,(x,y) in enumerate(pts):
            if valid[j]: draw.ellipse((x-3,y-3,x+3,y+3),fill=colors[j%4])
        draw.rectangle((0,0,832,22),fill='black'); draw.text((6,2),'2D human24 | predicted in-frame joints; no confidence head',font=font,fill='white'); output2.append(np.asarray(image))
        image=Image.new('RGB',(832,480),(22,27,36)); draw=ImageDraw.Draw(image)
        for view,axis in enumerate([0,2]):
            x0=view*416
            draw.text((x0+12,10),'3D '+('front X/Y' if axis==0 else 'side Z/Y')+' | / actor height',font=font,fill='white')
            draw.line((x0+208,48,x0+208,452),fill=(80,90,110),width=1); draw.line((x0+16,250,x0+400,250),fill=(80,90,110),width=1)
            points=np.stack([x0+208+xyz[t,:,axis]/bounds*185,250+xyz[t,:,1]/bounds*185],axis=-1)
            for j,p in enumerate(parents):
                if p>=0: draw.line([tuple(points[p]),tuple(points[j])],fill=colors[j%4],width=3)
            for j,(x,y) in enumerate(points): draw.ellipse((x-3,y-3,x+3,y+3),fill=colors[j%4])
        output3.append(np.asarray(image))
    return np.stack(output2),np.stack(output3),bounds

def sparse_case(c,module,info,device):
    d=OUTPUT/c['case_name']; assert (d/'DENSE_COMPLETE.json').exists()
    if (d/'SPARSE_COMPLETE.json').exists():
        assert json.loads((d/'sparse_manifest.json').read_text())['checkpoint']['sha256']==info['sha256']; return
    started=time.monotonic(); payload=torch.load(cached_artifact(d/'source_latent.pt'),map_location='cpu',weights_only=False); assert payload['source_video_sha256']==c['preprocessed_sha256']
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16): xy,xyz=module(payload['source_latent'][None].to(device),False)
    xy=xy[0].detach().float().cpu().numpy(); xyz=xyz[0].detach().float().cpu().numpy()
    assert xy.shape==(81,24,2) and xyz.shape==(81,24,3) and np.isfinite(xy).all() and np.isfinite(xyz).all()
    coordinates=SCRATCH/c['case_name']/'keypoints.npz'; coordinates.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(coordinates,prediction_2d=xy,prediction_3d=xyz,joint_names=np.array(info['joint_names']),parents=np.array(info['parents']),predicted_in_frame=((xy>=0)&(xy<=1)).all(-1))
    publish(coordinates,d/'keypoints.npz')
    frames,fps=read_video(cached_source(c)); v2,v3,bounds=skeleton_views(frames,xy,xyz,info['parents'])
    write_video(d/'keypoints_2d.mp4',v2); write_video(d/'keypoints_3d.mp4',v3)
    save_json(d/'sparse_manifest.json',{'checkpoint':info,'frames':81,'fps':16,'human_only_training_domain':True,'nonhuman_case':c['category']!='people','qualitative_only_no_ground_truth':True,'multiple_people':'single fixed 24-joint output, no multi-person detector','display_bounds_actor_height':bounds,'raw_coordinates_unclipped':True,'elapsed_seconds':time.monotonic()-started,'driver_sha256':DRIVER_SHA})
    save_json(d/'SPARSE_COMPLETE.json',{'complete':True,'tasks':['keypoints_2d','keypoints_3d'],'step':13000})
    event(case=c['case_name'],stage='sparse',complete=True,elapsed_seconds=round(time.monotonic()-started,2)); torch.cuda.empty_cache()

def final_case(c):
    d=OUTPUT/c['case_name']; assert (d/'DENSE_COMPLETE.json').exists() and (d/'SPARSE_COMPLETE.json').exists()
    if (d/'CASE_COMPLETE.json').exists(): return
    header=Image.new('RGBA',(1248,768),(0,0,0,0)); draw=ImageDraw.Draw(header); font=ImageFont.truetype(FONT,13)
    for i,label in enumerate(['RGB','Depth (near warm / far cool)','Normal','Matte: '+c['matte_subject'],'Raymap','Albedo','Optical flow (32 px scale)','2D human keypoints','3D human keypoints']): draw.text(((i%3)*416+6,(i//3)*256),label,font=font,fill='white')
    header_path=SCRATCH/c['case_name']/'panel_header.png'; header_path.parent.mkdir(parents=True,exist_ok=True); header.save(header_path); publish(header_path,d/'panel_header.png')
    cmd=['ffmpeg','-nostdin','-v','error','-y']
    for task in MODES: cmd+=['-threads','1','-i',str(cached_artifact(d/(task+'.mp4')))]
    cmd+=['-loop','1','-framerate','16','-i',str(header_path)]
    filters=[f'[{i}:v]scale=416:240,pad=416:256:0:16,setsar=1[v{i}]' for i in range(9)]
    layout='|'.join(f'{i%3*416}_{i//3*256}' for i in range(9))
    filters+=[''.join(f'[v{i}]' for i in range(9))+f'xstack=inputs=9:layout={layout}:shortest=1[grid]','[grid][9:v]overlay=shortest=1[out]']
    temp=SCRATCH/c['case_name']/'nine_panel.partial.mp4'
    cmd+=['-filter_complex_threads','1','-filter_complex',';'.join(filters),'-map','[out]','-r','16','-frames:v','81','-an','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(temp)]
    run(cmd); os.replace(temp,temp.with_name('nine_panel.mp4')); publish(temp.with_name('nine_panel.mp4'),d/'nine_panel.mp4')
    run(['ffmpeg','-nostdin','-v','error','-y','-threads','1','-i',str(cached_artifact(d/'nine_panel.mp4')),'-filter_threads','1','-vf','select=eq(n\\,40)','-frames:v','1',str(d/'preview.jpg')])
    wait_case(c['case_name'])
    checks=[]
    for task in MODES+['nine_panel']:
        p=d/(task+'.mp4'); s=json.loads(run(['ffprobe','-v','error','-threads','1','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate,nb_read_frames,duration,codec_name,pix_fmt','-of','json',str(cached_artifact(p))]))['streams'][0]
        assert int(s['nb_read_frames'])==81 and s['avg_frame_rate']=='16/1' and abs(float(s['duration'])-81/16)<.001
        assert (s['width'],s['height'])==((1248,768) if task=='nine_panel' else (832,480))
        assert s['codec_name']=='h264' and s['pix_fmt']=='yuv420p'
        checksum=sha(p); assert checksum==sha(cached_artifact(p)),p
        checks.append({'task':task,'file':p.name,'bytes':p.stat().st_size,'sha256':checksum,'probe':s})
    with np.load(d/'keypoints.npz',allow_pickle=False) as z:
        assert z['prediction_2d'].shape==(81,24,2) and z['prediction_3d'].shape==(81,24,3)
    save_json(d/'CASE_COMPLETE.json',{'complete':True,'case_name':c['case_name'],'predicted_modalities':8,'video_files':10,'checks':checks,'coordinates_sha256':sha(d/'keypoints.npz')})
    event(case=c['case_name'],stage='finalize',complete=True,videos=10)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['dense','sparse','finalize'],required=True); p.add_argument('--rank',type=int,default=0); p.add_argument('--world_size',type=int,default=4); p.add_argument('--case_name'); p.add_argument('--max_cases',type=int)
    global DRIVER_SHA
    DRIVER_SHA=sha(__file__)
    a=p.parse_args(); cases=json.loads((INPUT/'cases_manifest.json').read_text())['cases']; cases=cases[a.rank::a.world_size]
    if a.case_name: cases=[c for c in cases if c['case_name']==a.case_name]
    if a.max_cases: cases=cases[:a.max_cases]
    assert cases and 0<=a.rank<a.world_size
    OUTPUT.mkdir(exist_ok=True); started=time.monotonic(); module=vae=ids=info=None
    if a.stage!='finalize':
        device=torch.device('cuda:0'); torch.cuda.set_device(device); torch.manual_seed(42)
        if a.stage=='dense': module,vae,ids,info=dense_init(device,cases)
        else: module,info=sparse_init(device)
        save_json(OUTPUT/f'{a.stage}_worker{a.rank}_model.json',info)
        event(stage=a.stage,resident_model_ready=True,rank=a.rank,cases=len(cases))
    failures=[]
    for c in cases:
        try:
            if a.stage=='dense': dense_case(c,module,vae,ids,info,device)
            elif a.stage=='sparse': sparse_case(c,module,info,device)
            else: final_case(c)
        except Exception:
            error=traceback.format_exc(); failures.append({'case_name':c['case_name'],'error':error})
            d=OUTPUT/c['case_name']; d.mkdir(exist_ok=True); (d/(a.stage+'_FAILED.txt')).write_text(error); event(case=c['case_name'],stage=a.stage,failed=True,error=error)
    try:wait_publications()
    except Exception:failures.append({'case_name':'PUBLISHING','error':traceback.format_exc()})
    status={'stage':a.stage,'rank':a.rank,'complete':not failures,'requested':len(cases),'failed':failures,'elapsed_seconds':time.monotonic()-started}
    save_json(OUTPUT/f'{a.stage}_worker{a.rank}_status.json',status)
    wait_publications();PUBLISHER.shutdown(wait=True)
    if failures: raise RuntimeError(f'{len(failures)} {a.stage} failures')
    event(**status)
if __name__=='__main__': main()
