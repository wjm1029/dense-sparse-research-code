"""Prepare the explicitly requested real-video batch on sz6671 only."""
import concurrent.futures as cf
import hashlib,json,re,subprocess,zipfile,shutil,time
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('/pfs/mayuema/dense/wan/inputs/eightmodal_tianj300_20261018')
DELIVERY=Path('/pfs/mayuema/tianj_video300_20261012/delivery')
SCRATCH=Path('/dev/shm/eightmodal_tianj300_20261018/preparation'); SCRATCH.mkdir(parents=True,exist_ok=True)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()
def command(args): return subprocess.check_output(args,stderr=subprocess.PIPE)
def probe(p,count=False):
    cmd=['ffprobe','-v','error','-threads','1','-select_streams','v:0','-show_entries','stream=width,height,avg_frame_rate,nb_frames,nb_read_frames,duration','-of','json']
    if count: cmd+=['-count_frames']
    return json.loads(command(cmd+[str(p)]))['streams'][0]
def guess(r):
    if r['group']=='people': return 'person'
    t=(r.get('query','')+' '+r.get('title','')+' '+r['id']).lower().replace('_',' ').replace('-',' ')
    words=['sea lion','sea otter','tortoise','turtle','caterpillar','chrysalis','chameleon','chamaleon','gorilla','monkey','camel','koala','goat','bear','dog','cat','horse','lion','deer','bison','antelope','chamois','fox','wolf','coyote','raccoon','squirrel','rabbit','penguin','pelican','heron','crane','crow','eagle','owl','bird','duck','goose','geese','swan','sheep','cow','cattle','seal','whale','shark','fish','octopus','crab','lobster','jellyfish','coral','butterfly','bee','insect','flower','waterfall','fountain','shipwreck','ship','boat','train','car','truck','bus','bicycle','motorcycle','bridge','building','church','statue','windmill','robot','tractor','tree','plant','wetland','river','lake','ocean','mountain','forest','water','fire']
    for w in words:
        if re.search(r'\b'+re.escape(w)+r'(?:s|es)?\b',t): return {'chamaleon':'chameleon','geese':'goose','cattle':'cow'}.get(w,w)
    return 'animal' if r['group']=='animals' else 'object'
def prepare(item):
    part,n,r=item; name=f'p{part}_{Path(n).stem}'; src=ROOT/'originals'/name/Path(n).name
    src.parent.mkdir(parents=True,exist_ok=True); work=SCRATCH/name; work.mkdir(exist_ok=True)
    local_src=work/Path(n).name
    if not local_src.exists():
        with zipfile.ZipFile(SCRATCH/f'video300_part{part}_150.zip') as z: local_src.write_bytes(z.read(n))
    digest=sha(local_src); assert digest==r['sha256'],(name,'source hash mismatch')
    if not src.exists() or src.stat().st_size!=local_src.stat().st_size or sha(src)!=digest:
        if src.exists(): src.rename(src.with_name(src.name+f'.interrupted_{time.time_ns()}'))
        shutil.copyfile(local_src,src)
    info=probe(local_src); count=int(info.get('nb_frames') or probe(local_src,True)['nb_read_frames']); assert count>=81
    start=(count-81)//2; w,h=int(info['width']),int(info['height']); k=min(w//26,h//15); k-=k%2; cw,ch=26*k,15*k
    out=ROOT/'preprocessed'/(name+'.mp4'); out.parent.mkdir(exist_ok=True); local_out=work/'prepared.mp4'
    valid=False
    if out.exists():
        shutil.copyfile(out,local_out)
        try:
            check=probe(local_out,True); valid=(check['width'],check['height'],int(check['nb_read_frames']),check['avg_frame_rate'])==(832,480,81,'16/1')
        except Exception: valid=False
        if not valid: out.rename(out.with_name(out.name+f'.interrupted_{time.time_ns()}'))
    if not valid:
        filters=f'trim=start_frame={start}:end_frame={start+81},setpts=N/(16*TB),crop={cw}:{ch},scale=832:480:flags=lanczos,setsar=1'
        command(['ffmpeg','-nostdin','-v','error','-y','-threads','1','-i',str(local_src),'-filter_threads','1','-vf',filters,'-r','16','-frames:v','81','-an','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(local_out)])
        check=probe(local_out,True); assert (check['width'],check['height'],int(check['nb_read_frames']),check['avg_frame_rate'])==(832,480,81,'16/1')
        shutil.copyfile(local_out,out)
    thumb=ROOT/'thumbnails'/(name+'.jpg'); thumb.parent.mkdir(exist_ok=True); local_thumb=work/'thumbnail.jpg'
    command(['ffmpeg','-nostdin','-v','error','-y','-threads','1','-i',str(local_out),'-filter_threads','1','-vf','select=eq(n\\,40),scale=256:148','-frames:v','1',str(local_thumb)])
    shutil.copyfile(local_thumb,thumb); subject=guess(r)
    return {'case_name':name,'source_id':r['id'],'category':r['group'],'matte_subject':subject,'matte_prompt':f'output: {subject} alpha matte','subject_review':'provisional_metadata_requires_visual_review','original_source':str(src),'original_sha256':digest,'path':str(out),'preprocessed_sha256':sha(local_out),'thumbnail':str(thumb),'record':r,'preprocessing':{'original_frames':count,'original_fps':info['avg_frame_rate'],'first_original_frame_index_zero_based':start,'frames':81,'spatial':'largest centered 26:15 crop, then Lanczos resize; no aspect stretch','crop_size':[cw,ch],'output_size':[832,480],'output_fps':16,'temporal_resampling':False,'duration_seconds':81/16}}
def main():
    records=json.loads((ROOT/'source_records.json').read_text()); by_id={r['id']:r for r in records}; assert len(by_id)==300
    report=json.loads(Path('/pfs/mayuema/tianj_video300_20261012/records/video_only_delivery_report.json').read_text()); jobs=[]
    for part,p in enumerate(sorted(DELIVERY.glob('*.zip')),1):
        expected=next(x['sha256'] for x in report['archives'] if Path(x['path']).name==p.name); assert sha(p)==expected
        cached=SCRATCH/p.name
        if not cached.exists(): shutil.copyfile(p,cached)
        assert sha(cached)==expected
        with zipfile.ZipFile(cached) as z:
            names=sorted(n for n in z.namelist() if n.lower().endswith('.mp4')); assert len(names)==150
            for n in names:
                assert Path(n).name==n
                jobs.append((part,n,by_id[Path(n).stem]))
    cases=[]
    with cf.ThreadPoolExecutor(max_workers=6) as pool:
        for c in pool.map(prepare,jobs):
            cases.append(c)
            if len(cases)%10==0: print(json.dumps({'prepared':len(cases),'total':300}),flush=True)
    assert len(cases)==300 and len({c['case_name'] for c in cases})==300
    (ROOT/'cases_manifest.provisional.json').write_text(json.dumps({'cases':cases},ensure_ascii=False,indent=2))
    sheets=ROOT/'review_sheets'; sheets.mkdir(exist_ok=True); font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    for offset in range(0,300,30):
        sheet=Image.new('RGB',(256*6,188*5),'white'); draw=ImageDraw.Draw(sheet)
        for j,c in enumerate(cases[offset:offset+30]):
            x=(j%6)*256; y=(j//6)*188
            sheet.paste(Image.open(c['thumbnail']),(x,y))
            draw.text((x+2,y+148),f"{offset+j+1:03d} {c['source_id'][:30]}",font=font,fill='black')
            draw.text((x+2,y+166),f"{c['category']} | {c['matte_subject']}",font=font,fill='black')
        sheet.save(sheets/f'review_{offset+1:03d}_{min(offset+30,300):03d}.jpg',quality=90)
    (ROOT/'PREPARATION_COMPLETE').touch(); print('PREPARATION_COMPLETE',flush=True)
if __name__=='__main__': main()
