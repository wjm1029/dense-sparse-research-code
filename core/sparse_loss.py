"""NEW / UNTRAINED: clip-level Hungarian matching and masked JTA objectives.

默认系数是新实验的初始配置，未经全量训练调优。2D 是 crop-normalized，3D 是米。
每个 batch 元素可有不同人数；person_ids 仅构造 GT 轨迹，不直接作为预测 slot。
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class LossConfig:
    xy: float = 5.
    xyz: float = 1.
    presence: float = 1.
    joint_state: float = .2
    router: float = .001
    occluded_joint_weight: float = .25
    unmatched_presence_weight: float = .1
    matching_xy: float = 5.
    matching_xyz: float = 1.
    matching_presence: float = 2.


def load_target_npz(path, device='cpu'):
    with np.load(path,allow_pickle=False) as data:
        return {k:torch.from_numpy(data[k].copy()).to(device) for k in data.files}


def joint_weights(gt, config, three_d=False):
    mask=gt['valid_3d'] if three_d else gt['in_crop']
    mask=mask.bool() & gt['present'].bool()[...,None]
    visible=gt['visible'].bool()
    return mask.float()*torch.where(visible,1.,config.occluded_joint_weight)


def weighted_mse(pred, target, weight):
    # 把无效值先替换，再做减法；padding/无效标签即便是 NaN 也不传播到 loss。
    active=weight>0
    if not torch.isfinite(target[active]).all():raise ValueError('Nonfinite supervised target')
    delta=torch.where(active[...,None],pred,0)-torch.where(active[...,None],target,0)
    return (delta.square().sum(-1)*weight).sum()/(weight.sum()*pred.shape[-1]).clamp_min(1)


def validate(prediction, targets, task):
    b,t,q,j,d=prediction['xy'].shape
    if j!=22 or d!=2 or len(targets)!=b:raise ValueError('Bad prediction/target dimensions')
    if task not in (0,1) or (task==1 and 'xyz' not in prediction):raise ValueError('Missing 3D branch')
    if not all(torch.isfinite(x).all() for x in prediction.values()):raise ValueError('Nonfinite prediction')
    for gt in targets:
        n=gt['xy'].shape[1]
        if n>q:raise ValueError(f'{n} tracks exceed {q} slots; cannot drop people')
        if gt['xy'].shape!=(t,n,22,2) or gt['xyz'].shape!=(t,n,22,3):raise ValueError('Target shape mismatch')
        if gt['present'].shape!=(t,n):raise ValueError('Presence shape mismatch')
        if n and not gt['present'].any(0).all():raise ValueError('Every GT track must be observed at least once')
        if gt['present'].any() and not gt['visible'].any(-1)[gt['present'].bool()].all():
            raise ValueError('Positive GT must have a visible in-crop joint')


@torch.no_grad()
def match_tracks(prediction, targets, task, config=LossConfig()):
    validate(prediction,targets,task)
    assignments=[]
    for bi,gt in enumerate(targets):
        q=prediction['xy'].shape[2];n=gt['xy'].shape[1]
        if n==0:
            empty=torch.empty(0,dtype=torch.long,device=prediction['xy'].device)
            assignments.append((empty,empty));continue
        weights2=joint_weights(gt,config);weights3=joint_weights(gt,config,True)
        logits=prediction['presence_logits'][bi].transpose(0,1)  # [Q,T]
        cost=logits.new_zeros((q,n))
        # 单轨迹构造 cost，避免分配 [Q,N,T,22,D] 的巨大临时张量。
        for ni in range(n):
            w=weights2[:,ni];valid=w>0
            target=torch.where(valid[...,None],gt['xy'][:,ni],0)
            p=torch.where(valid[:,None,:,None],prediction['xy'][bi],0)
            error=(p-target[:,None]).abs().sum(-1)
            cost[:,ni]=config.matching_xy*(error*w[:,None]).sum((0,2))/(2*w.sum()).clamp_min(1)
            expected=gt['present'][:,ni].float()[None].expand_as(logits)
            cost[:,ni]+=config.matching_presence*F.binary_cross_entropy_with_logits(logits,expected,reduction='none').mean(1)
            if task==1:
                w=weights3[:,ni];valid=w>0
                target=torch.where(valid[...,None],gt['xyz'][:,ni],0)
                p=torch.where(valid[:,None,:,None],prediction['xyz'][bi],0)
                error=(p-target[:,None]).abs().sum(-1)
                cost[:,ni]+=config.matching_xyz*(error*w[:,None]).sum((0,2))/(3*w.sum()).clamp_min(1)
        if not torch.isfinite(cost).all():raise ValueError('Nonfinite assignment cost')
        qi,ni=linear_sum_assignment(cost.cpu().numpy())
        assignments.append((torch.as_tensor(qi,device=logits.device),torch.as_tensor(ni,device=logits.device)))
    return assignments


def sparse_objective(prediction, targets, task, gates, config=LossConfig()):
    assignments=match_tracks(prediction,targets,task,config)
    zero=prediction['xy'].sum()*0
    terms={k:zero for k in ('xy','xyz','presence','in_crop','visible')}
    for bi,(gt,(qi,ni)) in enumerate(zip(targets,assignments)):
        logits=prediction['presence_logits'][bi]
        expected=torch.zeros_like(logits)
        weight=torch.full_like(logits,config.unmatched_presence_weight)
        if len(qi):expected[:,qi]=gt['present'][:,ni].float();weight[:,qi]=1
        terms['presence']=terms['presence']+(F.binary_cross_entropy_with_logits(logits,expected,reduction='none')*weight).mean()
        if not len(qi):
            # 空图仍可 backward；所有已执行读出连接到零，便于 DDP。
            for name,key in [('xy','xy'),('in_crop','in_crop_logits'),('visible','visible_logits')]:
                terms[name]=terms[name]+prediction[key][bi].sum()*0
            if task==1:terms['xyz']=terms['xyz']+prediction['xyz'][bi].sum()*0
            continue
        terms['xy']=terms['xy']+weighted_mse(prediction['xy'][bi][:,qi],gt['xy'][:,ni],joint_weights(gt,config)[:,ni])
        if task==1:
            terms['xyz']=terms['xyz']+weighted_mse(prediction['xyz'][bi][:,qi],gt['xyz'][:,ni],joint_weights(gt,config,True)[:,ni])
        observed=gt['present'][:,ni].float()[...,None].expand(-1,-1,22)
        for name,key in [('in_crop','in_crop_logits'),('visible','visible_logits')]:
            target=gt[name][:,ni].float()
            error=F.binary_cross_entropy_with_logits(prediction[key][bi][:,qi],target,reduction='none')
            terms[name]=terms[name]+(error*observed).sum()/observed.sum().clamp_min(1)
    terms={k:v/len(targets) for k,v in terms.items()}
    terms['router']=(gates.float()-1).square().mean()
    total=(config.xy*terms['xy']+config.xyz*terms['xyz']+config.presence*terms['presence']
           +config.joint_state*(terms['in_crop']+terms['visible'])+config.router*terms['router'])
    return total,{k:v.detach() for k,v in terms.items()},assignments
