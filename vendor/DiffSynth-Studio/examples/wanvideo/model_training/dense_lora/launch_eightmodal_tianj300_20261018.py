"""Persistent four-GPU inference runner with CPU exports and honest completion markers."""
import hashlib,json,os,subprocess,time,shutil
from pathlib import Path
BASE=Path('/pfs/mayuema/dense/wan'); REPO=BASE/'DiffSynth-Studio'; HERE=REPO/'examples/wanvideo/model_training/dense_lora'
INPUT=BASE/'inputs/eightmodal_tianj300_20261018'; OUT=BASE/'outputs/eightmodal_tianj300_dense120k_sparse13k_20261018'; LOG=BASE/'logs/eightmodal_tianj300_20261018'
PYTHON=BASE/'envs/cog/bin/python'; DRIVER=HERE/'infer_eightmodal_tianj300_20261018.py'; BOOT=HERE/'cog_runtime_bootstrap.py'
RAM=Path('/dev/shm/eightmodal_tianj300_20261018/results'); RAMLOG=Path('/dev/shm/eightmodal_tianj300_20261018/logs'); RAMLOG.mkdir(parents=True,exist_ok=True)
LAST_SYNC=0
MODES=['rgb','depth','normal','matte','raymap','albedo','optical_flow','keypoints_2d','keypoints_3d','nine_panel']
def save(p,d):
    temp=p.with_name(p.name+'.pending'); temp.write_text(json.dumps(d,ensure_ascii=False,indent=2)); os.replace(temp,p)
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()
def sync_logs(force=False):
    global LAST_SYNC
    if force or time.monotonic()-LAST_SYNC>=60:
        for p in RAMLOG.glob('*.log'):
            shutil.copyfile(p,LOG/p.name)
        LAST_SYNC=time.monotonic()
def progress(stage,started,workers):
    row={'stage':stage,'total':300,'dense_complete':len(list(RAM.glob('*/DENSE_PUBLISHED.json'))),'sparse_complete':len(list(RAM.glob('*/SPARSE_PUBLISHED.json'))),'case_complete':len(list(RAM.glob('*/CASE_PUBLISHED.json'))),'dense_computed':len(list(RAM.glob('*/DENSE_COMPLETE.json'))),'sparse_computed':len(list(RAM.glob('*/SPARSE_COMPLETE.json'))),'elapsed_seconds':round(time.monotonic()-started,2),'workers':[{k:v for k,v in w.items() if k!='process'}|{'returncode':w['process'].poll()} for w in workers]}
    save(RAM/'progress.json',row)
    if time.monotonic()-LAST_SYNC>=60 or stage=='complete':
        sync_logs(True); save(OUT/'progress.json',row)
    return row

def main():
    OUT.mkdir(exist_ok=True); LOG.mkdir(exist_ok=True)
    manifest=json.loads((INPUT/'cases_manifest.json').read_text()); cases=manifest['cases']; assert len(cases)==300 and manifest['subject_review_complete']
    assert not (OUT/'RUN_COMPLETE').exists()
    env=os.environ.copy(); env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',DIFFSYNTH_SKIP_DOWNLOAD='True',OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    for marker in ('DENSE_COMPLETE.json','SPARSE_COMPLETE.json','CASE_COMPLETE.json'):
        for old in OUT.glob('*/'+marker):
            cached=RAM/old.relative_to(OUT); cached.parent.mkdir(parents=True,exist_ok=True)
            if not cached.exists(): shutil.copyfile(old,cached)
            published=cached.with_name(cached.name.replace('_COMPLETE.json','_PUBLISHED.json'));shutil.copyfile(old,published)
    if (OUT/'run_config.json').exists() and not (OUT/'run_config.first_full_launch.json').exists(): shutil.copyfile(OUT/'run_config.json',OUT/'run_config.first_full_launch.json')
    started=time.monotonic(); save(OUT/'run_config.json',{'input_manifest':str(INPUT/'cases_manifest.json'),'input_manifest_sha256':digest(INPUT/'cases_manifest.json'),'driver_sha256':digest(DRIVER),'launcher_sha256':digest(Path(__file__)),'host':'sz6671','cases':300,'frames':81,'fps':16,'eight_predicted_modalities':MODES[1:-1],'rgb_is_input':True,'weights':{'dense':str(BASE/'outputs/wan22_dense6_flow32_dispersion_100k_to120k_20261012/step-120000.safetensors'),'sparse':str(BASE/'outputs/wan22_sparse_lora_bidir_full_4gpu_overnight_20261012/step-13000.recovered.safetensors')}})
    for stage,count in [('dense',4),('sparse',4),('finalize',8)]:
        workers=[]; handles=[]
        for rank in range(count):
            worker_env=env.copy(); worker_env['CUDA_VISIBLE_DEVICES']=str(rank) if stage!='finalize' else ''
            old_log=LOG/f'{stage}_{rank}.log'
            if old_log.exists() and not old_log.with_name(old_log.name+'.before_io_restart').exists(): shutil.copyfile(old_log,old_log.with_name(old_log.name+'.before_io_restart'))
            handle=(RAMLOG/f'{stage}_{rank}.log').open('w'); handles.append(handle)
            command=[str(PYTHON),'-u',str(BOOT),str(DRIVER),'--stage',stage,'--rank',str(rank),'--world_size',str(count)]
            proc=subprocess.Popen(command,cwd=REPO,env=worker_env,stdout=handle,stderr=subprocess.STDOUT)
            workers.append({'rank':rank,'pid':proc.pid,'log':str(RAMLOG/f'{stage}_{rank}.log'),'persistent_log':str(LOG/f'{stage}_{rank}.log'),'command':command,'process':proc})
        last=None
        while True:
            row=progress(stage,started,workers); compact={k:row[k] for k in ['stage','dense_complete','sparse_complete','case_complete']}
            if compact!=last: print(json.dumps(compact),flush=True); last=compact
            if all(w['process'].poll() is not None for w in workers): break
            time.sleep(10)
        codes=[w['process'].wait() for w in workers]
        sync_logs(True); save(OUT/'progress.json',progress(stage,started,workers))
        for h in handles: h.close()
        if any(codes): raise RuntimeError(f'{stage} worker return codes: {codes}; inspect {LOG}')
    rows=[]; video_count=0
    for c in cases:
        d=OUT/c['case_name']; v=json.loads((d/'CASE_COMPLETE.json').read_text()); assert v['complete'] and v['predicted_modalities']==8 and len(v['checks'])==10
        for item in v['checks']:
            assert (d/item['file']).stat().st_size==item['bytes']
        assert digest(d/'keypoints.npz')==v['coordinates_sha256']
        video_count+=10; rows.append({'name':c['case_name'],'category':c['category'],'subject':c['matte_subject']})
    assert len(rows)==300 and video_count==3000
    (OUT/'cases_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    readme='''# Dense6 120k + Sparse13k · 300 real cases

Eight predicted modalities: depth, normal, matte, raymap, albedo, optical flow, 2D human keypoints, 3D human keypoints. RGB is the input. Each case has nine separate videos plus nine_panel.mp4 (RGB + eight predictions). All preserve 81 consecutive frames at 16 FPS (5.0625 sec). Single views 832x480; nine-panel 1248x768. Full video decode/frame/FPS checks and SHA256 are recorded in CASE_COMPLETE.json.

Dense checkpoint: wan22_dense6_flow32_dispersion_100k_to120k_20261012/step-120000.safetensors. Original strict checkpoint loader and provenance checks retained. Extra matte words are user-authorized unseen-subject evaluation contexts from the same frozen UMT5; native prompt and task feature tables are validated unchanged before appending those contexts. No production model source or weights modified.

Sparse checkpoint: wan22_sparse_lora_bidir_full_4gpu_overnight_20261012/step-13000.recovered.safetensors. This is the original joint 2D/3D head (not the newer two-prompt model). Its contract/source hashes and all 617 tensors are verified. Coordinates are in keypoints.npz: 2D image_xy_01; 3D root-relative camera coordinates per actor height, not meters. Fixed24 human joints; no person detector, confidence/occlusion head, or multi-person association. Animal/no-person clips are cross-domain qualitative tests; predictions do not establish a valid animal skeleton.

Depth: fixed [-1,1] to Spectral_r, near warm/far cool, no per-frame or per-clip rescaling; relative inverse depth, not metric. Optical flow: delayed frame t contains t-1 to t, +/-32px encoding; third channel validity, black invalid and first frame. HSV hue direction, magnitude brightness saturates20px; optical_flow_latent.pt preserves the unmodified predicted latent. Normal and raymap retain the checkpoint's training coordinate conventions without post-hoc flips.

Each case stores source_latent.pt and six predicted latent tensors for reproducibility. No GT, no accuracy metrics. Original ZIPs, extracted videos, source/license records and centered-crop preprocessing provenance remain on server. All computation on sz6671; local transfers and previews only.

Extract or serve this directory and open index.html to browse all cases.
'''
    (OUT/'README.md').write_text(readme)
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Dense6 + Sparse · 300 cases</title><style>body{font:15px system-ui;background:#101722;color:#e8eef7;margin:24px}main{display:grid;grid-template-columns:300px 1fr;gap:20px}aside{height:82vh;overflow:auto}input{width:95%;padding:10px}button{background:#25374b;color:white;border:1px solid #526177;border-radius:5px;padding:8px;margin:4px;cursor:pointer}aside button{display:block;text-align:left;width:95%}video,img{width:100%;max-height:70vh;object-fit:contain;background:black}a{color:#95bfff}@media(max-width:800px){main{display:block}aside{height:180px}}</style><h1>Dense6 120k + Sparse13k · 300 cases</h1><p>8 predictions + RGB · 81 frames · 16 FPS · 5.0625 sec</p><main><aside><input id="q" placeholder="Search case or subject"><div id="list"></div></aside><article><h2 id="name"></h2><p id="subject"></p><div id="tabs"></div><video id="video" controls muted loop></video><p><a href="README.md">Readme / coordinate conventions</a></p><img id="preview"></article></main><script>const cases=__CASES__,modes=__MODES__;let c=cases[0],m='nine_panel';const el=x=>document.getElementById(x);function show(){el('name').textContent=c.name;el('subject').textContent=c.category+' | matte: '+c.subject;el('video').src=c.name+'/'+m+'.mp4';el('preview').src=c.name+'/preview.jpg'}function list(){el('list').replaceChildren();let q=el('q').value.toLowerCase();for(let x of cases.filter(x=>JSON.stringify(x).toLowerCase().includes(q))){let b=document.createElement('button');b.textContent=x.name+' | '+x.subject;b.onclick=()=>{c=x;show()};el('list').append(b)}}for(let x of ['nine_panel',...modes.filter(x=>x!=='nine_panel')]){let b=document.createElement('button');b.textContent=x;b.onclick=()=>{m=x;show()};el('tabs').append(b)}el('q').oninput=list;list();show();</script></html>'''
    (OUT/'index.html').write_text(page.replace('__CASES__',json.dumps(rows,ensure_ascii=False)).replace('__MODES__',json.dumps(MODES)))
    summary={'complete':True,'cases':300,'predicted_modalities':8,'video_files':3000,'frames':81,'fps':16,'duration_seconds':81/16,'dense_step':120000,'sparse_step':13000,'elapsed_seconds':round(time.monotonic()-started,2),'failures':0,'output':str(OUT),'host':'sz6671'}
    save(OUT/'RUN_FINISHED.json',summary); (OUT/'RUN_COMPLETE').touch(); progress('complete',started,[]); print(json.dumps(summary),flush=True)
if __name__=='__main__':
    try: main()
    except Exception:
        import traceback
        OUT.mkdir(exist_ok=True); (OUT/'RUN_FAILED.txt').write_text(traceback.format_exc()); raise
