"""显式选择操作才加载主干；--help 不加载模型。单步训练仅用于接口调试。"""
import argparse
import gc
from pathlib import Path
import sys
import numpy as np
import torch


def read_rgb(path, device):
    rgb=np.load(path,allow_pickle=False)
    if rgb.shape!=(81,480,832,3) or rgb.dtype!=np.uint8:
        raise ValueError('Expected uint8 RGB [81,480,832,3], no resizing in this example')
    return torch.from_numpy(rgb.copy()).permute(3,0,1,2).to(device=device,dtype=torch.bfloat16)/127.5-1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['dense-predict','sparse-one-step'])
    p.add_argument('--root',required=True,help='video_dense project root')
    p.add_argument('--rgb',required=True,help='lossless RGB .npy')
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--task',required=True,choices=['depth','matte','normal','2d','3d'])
    p.add_argument('--decoder',choices=['direct','vae'],default='direct')
    p.add_argument('--targets',help='JTA targets .npz, required for sparse-one-step')
    a=p.parse_args();root=Path(a.root);device=torch.device(a.device)
    output=Path(a.output)
    if output.exists():raise FileExistsError(f'Refuse to overwrite {output}')
    if device.type!='cuda':raise ValueError('These full-model examples require CUDA')
    torch.cuda.set_device(device);rgb=read_rgb(a.rgb,device)
    if a.mode=='dense-predict':
        if a.task not in ('depth','matte','normal'):p.error('Dense task must be depth/matte/normal')
        from dense_native import load_existing_dense,predict_existing_dense
        from dense_decode import load_direct_head
        module,vae,contract=load_existing_dense(root,device)
        head=None
        if a.decoder=='direct':
            exp=root/'experiments/pixvid_decode_probe_20261122'
            head=load_direct_head(exp/('pixvid_detailer_v12' if a.task=='depth' else 'pixvid_modalities_v13')/a.task/'head_30000.pt',a.task,device)
        pred=predict_existing_dense(module,vae,contract,rgb,a.task,head)
        payload=dict(prediction=pred.cpu(),task=a.task,decoder=a.decoder,layout='B,T,C,H,W',range='signed training units')
    else:
        if a.task not in ('2d','3d') or not a.targets:p.error('Sparse needs --task 2d/3d and --targets')
        repo=root/'code/DiffSynth-Studio';scripts=repo/'examples/wanvideo/model_training/dense_lora'
        sys.path[:0]=[str(repo),str(scripts)]
        from infer_wan22_dense4 import load_official_vae
        from sparse_model import build_fresh_jta_model,make_optimizer,training_step
        from sparse_loss import load_target_npz
        models=root/'models/Wan2.2-TI2V-5B'
        vae=load_official_vae(root/'code/Wan2.2',models,device)
        with torch.no_grad():latent=vae.encode([rgb])[0][None]
        del vae,rgb;gc.collect();torch.cuda.empty_cache()
        torch.manual_seed(20261127)
        model=build_fresh_jta_model(str(models),device)
        target=load_target_npz(a.targets,device)
        optimizer=make_optimizer(model)
        loss,metrics=training_step(model,optimizer,latent,[target],0 if a.task=='2d' else 1)
        trainable={n for n,p in model.named_parameters() if p.requires_grad}
        # head buffers + context + router feature table are part of the contract too.
        names=trainable|{n for n,_ in model.named_buffers() if n=='context' or n.startswith(('task_router.','sparse_head.'))}
        state={n:v.detach().cpu() for n,v in model.state_dict().items() if n in names}
        if set(state)!=names:raise RuntimeError('Incomplete adapter state')
        payload=dict(schema='jta_multi_person_bidir_prototype_v1',step=1,task=a.task,
            weights=state,loss=loss,metrics=metrics,
            note='One debug update only; no optimizer/RNG saved, not a resume checkpoint or trained model')
    output.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,output)
    print(f'Saved {output}')


if __name__=='__main__':main()
