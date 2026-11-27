"""现有 Dense6 OPSD 5250 的加载/前向接线，提取自已验证的原生推理代码。

依赖项目根目录的完整 DiffSynth/Wan 源码与权重；不使用 Sparse 的双向前向。
本次代码包只运行了核心模块测试，没有再次跑整个 5B 主干。
"""
import gc
import hashlib
import json
from pathlib import Path
import sys
import torch
from task_gating import DENSE_TASKS
from dense_decode import decode_native_features

CHECKPOINT_SHA='51ee931e662cf59347f96b2f9b29136f2c62e8db77c958af6a2e225ec57d6d97'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def load_existing_dense(project_root, device='cuda'):
    root=Path(project_root);device=torch.device(device)
    if device.type!='cuda':raise ValueError('Full native model entry requires CUDA')
    ckpt=root/'weights/current/dense6/long477_opsd5250_FINAL/step-5250.safetensors'
    if sha(ckpt)!=CHECKPOINT_SHA:raise ValueError('Wrong Dense OPSD checkpoint')
    contract=json.loads((ckpt.parent/'config.json').read_text())['contract']
    repo=root/'code/DiffSynth-Studio';scripts=repo/'examples/wanvideo/model_training/dense_lora'
    for name,checksum in contract['scripts'].items():
        if sha(scripts/name)!=checksum:raise ValueError(f'Native source mismatch: {name}')
        loaded=sys.modules.get(Path(name).stem)
        if loaded is not None and getattr(loaded,'__file__',None) and sha(loaded.__file__)!=checksum:
            raise RuntimeError(f'A different {name} is already imported; use a fresh process')
    sys.path[:0]=[str(scripts),str(repo)]
    from safetensors.torch import load_file
    from wan22_dense6_long477_opsd import Dense6OPSDModule
    from infer_wan22_dense4 import load_official_vae  # 仅复用 VAE loader，不调用 Dense4 forward
    models=root/'models/Wan2.2-TI2V-5B'
    module=Dense6OPSDModule(str(models))
    weights=load_file(str(ckpt),device='cpu');contexts=weights.pop('condition_contexts')
    expected={n for n,p in module.named_parameters() if p.requires_grad}|{
        n for n in module.state_dict() if n.startswith('task_router.')}
    if set(weights)!=expected:raise ValueError('Native checkpoint key set mismatch')
    loaded=module.load_state_dict(weights,strict=False)
    if loaded.unexpected_keys or set(loaded.missing_keys)&expected:raise ValueError('Incomplete native load')
    module.pipe.text_encoder=None;module.pipe.tokenizer=None
    module.context_table=contexts.to(device=device,dtype=torch.bfloat16)
    module.requires_grad_(False).eval()
    module.pipe.dit.to(device);module.task_router.to(device);module.pipe.device=device
    del weights,contexts;gc.collect();torch.cuda.empty_cache()
    vae=load_official_vae(root/'code/Wan2.2',models,device)
    return module,vae,contract


@torch.no_grad()
def native_opsd_rollout(module, source_latent, task_id, prompt_id, capture_features=True):
    """batch1, 81 raw -> 21 latent; chunk3, sink3/local12, predicted history。"""
    if tuple(source_latent.shape)!=(48,21,30,52):raise ValueError('Expected one 81-frame source latent')
    z=source_latent;dit=module.pipe.dit;features=[]
    hook=dit.head.head.register_forward_pre_hook(lambda layer,args:features.append(args[0].detach())) if capture_features else None
    caches=module.causal_controller.new_kv_cache();history=torch.zeros_like(z[None,:,:3]);predictions=[]
    tasks=torch.tensor([task_id],device=z.device);prompts=torch.tensor([prompt_id],device=z.device)
    dit.train()  # 原 streaming helper 要求 train；外层 no_grad，Wan 此处无 dropout。
    try:
        with torch.autocast(z.device.type,dtype=torch.bfloat16):
            for offset in range(0,21,3):
                pred,_,_=module(z[None,:,offset:offset+3],history,tasks,prompts,offset,caches,False)
                predictions.append(pred);history=pred.detach();module.task_router.clear()
    finally:
        if hook is not None:hook.remove()
        module.task_router.clear();dit.eval()
    native_features=torch.cat(features,1).view(1,21,15,26,3072) if capture_features else None
    return torch.cat(predictions,2),native_features


@torch.no_grad()
def predict_existing_dense(module, vae, contract, rgb, task, head=None, prompt=None):
    """rgb [3,81,480,832] float [-1,1]；返回 [1,81,C,480,832]。

    head=None 使用原 VAE decoder；head=训练好的 DensePixelDecode 使用直接像素头。
    同一 task 的 latent AR history 在两条 decode 分支中完全保留。
    """
    if task not in DENSE_TASKS:raise ValueError('Unknown dense task')
    if tuple(rgb.shape)!=(3,81,480,832) or not torch.isfinite(rgb).all() or rgb.abs().max()>1.0001:
        raise ValueError('Expected RGB [3,81,480,832] normalized to [-1,1]')
    prompts=dict(depth='output: depth',normal='output: surface normal',matte='output: person alpha matte',
                 raymap='output: camera raymap',albedo='output: albedo',optical_flow='output: optical flow')
    pid=contract['prompts'].index(prompt or prompts[task])
    if head is not None and head.task!=task:raise ValueError('Decode head task mismatch')
    device=module.pipe.device;rgb=rgb.to(device=device,dtype=torch.bfloat16)
    source=vae.encode([rgb])[0]
    latent,features=native_opsd_rollout(module,source,DENSE_TASKS.index(task),pid,head is not None)
    if head is not None:
        with torch.autocast(device.type,dtype=torch.bfloat16):return decode_native_features(head,features)
    pixels=vae.decode([latent[0]])[0].float().permute(1,0,2,3)[None]
    return pixels.mean(2,keepdim=True) if task in ('depth','matte') else pixels
