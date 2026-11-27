"""现有 Dense loss 的可读入口；保留原权重和归一化语义。"""
import math
import torch
import torch.distributed as dist
from native_reference.wan22_masked_loss import masked_latent_mse


def dispersion(features, prompt_ids, tau=1.):
    if tau<=0:raise ValueError('tau must be positive')
    if dist.is_available() and dist.is_initialized():
        from native_reference.wan22_dense6_dispersion import feature_dispersion
        return feature_dispersion(features,prompt_ids,tau)[0]
    # 单进程等价实现；同 prompt 的成对距离，包含对角线，不做时空池化。
    z=features.flatten(1).float();logits=[]
    for i in range(len(z)):
        logits.append(z[i].sum()*0)
        for j in range(i+1,len(z)):
            if int(prompt_ids[i])==int(prompt_ids[j]):
                value=-(z[i]-z[j]).square().mean()/tau
                logits.extend([value,value])
    return torch.logsumexp(torch.stack(logits),0)-math.log(len(logits))


def dense_bidirectional_loss(velocity, target_latent, mask, gates, ninth_block_features, prompt_ids):
    """现有 t=0 任务：预测 target latent = -velocity，不是噪声扩散采样 loss。"""
    latent=masked_latent_mse(-velocity,target_latent,mask)
    router=(gates.float()-1).square().mean()
    spread=dispersion(ninth_block_features,prompt_ids)
    return latent+.001*router+.01*spread,dict(latent=latent,router=router,dispersion=spread)


def opsd_block_average(supervised_losses, distillation_losses):
    """原 Dense OPSD: 40 个监督块与 33 个蒸馏块分别求平均，再以系数 1 相加。

    supervised_losses 每块已含 latent/router/dispersion。
    distillation_losses 是 student 与 detach 的 EMA teacher 的 latent MSE。
    teacher rollout、延迟 clean-prefix replay 和分离 KV 必须用 native OPSD 实现，
    不能仅调用这个求和函数就声称实现了 OPSD。
    """
    if not supervised_losses:raise ValueError('No supervised blocks')
    supervised=torch.stack(supervised_losses).mean()
    distilled=torch.stack(distillation_losses).mean() if distillation_losses else supervised*0
    return supervised+distilled


def pixel_decode_loss(pred, target, lpips_model, frame_indices):
    """已训练 depth/matte/normal 输出头：MSE + .2 temporal Charb + .2 VGG LPIPS。

    输入 [B,T,C,H,W]，scalar C=1 / normal C=3，训练值域沿用 [-1,1]。
    lpips_model 必须是冻结 VGG LPIPS；调用方指定通常 4 个抽样帧。
    这项 loss 用在冻结 DiT 的输出头训练，不混入现有 Dense 主干 latent loss。
    """
    if pred.shape!=target.shape or pred.ndim!=5 or not frame_indices:
        raise ValueError('Expected matching [B,T,C,H,W] and nonempty frame indices')
    p,g=pred.float(),target.float()
    mse=(p-g).square().mean()
    delta=p.diff(dim=1)-g.diff(dim=1)
    temporal=(delta.square()+1e-6).sqrt().mean()-.001 if p.shape[1]>1 else mse*0
    ps=p[:,frame_indices].flatten(0,1);gs=g[:,frame_indices].flatten(0,1)
    if ps.shape[1]==1:ps=ps.expand(-1,3,-1,-1);gs=gs.expand(-1,3,-1,-1)
    perceptual=lpips_model(ps,gs).mean()
    return mse+.2*temporal+.2*perceptual,dict(mse=mse,temporal=temporal,lpips=perceptual)
