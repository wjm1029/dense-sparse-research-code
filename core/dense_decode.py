"""现有 Dense 直接像素输出头，保留 checkpoint state_dict 参数名和结构。"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from native_reference.pixvid_detailer_v12 import Detailer as DepthDetailer, CausalConv


class DensePixelDecode(DepthDetailer):
    def __init__(self, task, dim=3072, widths=None, checkpoint_blocks=True):
        if task not in ('depth','matte','normal'):raise ValueError('Only these three direct heads were trained')
        super().__init__(dim,widths,checkpoint_blocks)
        self.task=task;self.channels=3 if task=='normal' else 1
        if self.channels==3:
            channels=(self.widths if widths is None else widths)[-1]
            self.output=CausalConv(channels,3)

    def forward(self, features, frame_ids, state=None):
        # 与原 v12/v13 相同；这里统一返回 [B,T,C,H,W]。
        phase=torch.where(frame_ids==0,4,(frame_ids-1).remainder(4))
        x=features.permute(0,4,1,2,3)
        e=self.phase(phase).permute(1,0)[None,:,:,None,None]
        x=self.bottleneck(F.silu(x+e),state,'bottleneck')
        for i,block in enumerate(self.blocks):
            x=checkpoint(block,x,use_reentrant=False) if self.training and self.checkpoint_blocks and state is None else block(x,state,f'up{i}')
        # CPU smoke 与 GPU 正式推理都支持；输出层保留 FP32。
        with torch.autocast(device_type=x.device.type,enabled=False):
            return self.output(x.float(),state,'output').permute(0,2,1,3,4)


@torch.no_grad()
def decode_native_features(head, features, chunk=4):
    """输入是 dit.head.head 的 PRE-hook 特征（已完成 norm+modulation）。

    features [B,21,15,26,3072]，不是 VAE latent，也不是 dit.head 的输入。
    每个视频/任务独立 state；81 帧下沿用原 0→0、1→1..4 映射。
    """
    if features.ndim!=5 or chunk<1:raise ValueError('Bad features/chunk')
    if head.training:raise ValueError('Call head.eval() before decoding')
    frames=1+4*(features.shape[1]-1);state={};output=[]
    for start in range(0,frames,chunk):
        ids=torch.arange(start,min(start+chunk,frames),device=features.device)
        index=torch.where(ids==0,0,(ids-1)//4+1)
        output.append(head(features[:,index],ids,state))
    return torch.cat(output,1)


def load_direct_head(path, task, device='cuda'):
    head=DensePixelDecode(task).to(device).eval()
    payload=torch.load(path,map_location='cpu',weights_only=False)
    request=payload.get('request',{})
    if 'task' in request and request['task']!=task:raise ValueError('Checkpoint task mismatch')
    head.load_state_dict(payload['model'],strict=True)
    return head


@torch.no_grad()
def official_vae_decode(vae, target_latents):
    """输入 [B,48,T,30,52]，返回 [B,Traw,3,H,W]；latent 已经是 -velocity。"""
    return torch.stack(vae.decode(list(target_latents))).permute(0,2,1,3,4)

# Dense OPSD 的旧 latent readout 仍生成 AR history，即使最终像素用直接输出头。
# 不得删掉 history readout，也不得把旧 bidirectional 前向用于 OPSD checkpoint。
# 只有 depth/matte/normal 有此直接头；raymap/albedo/flow 沿用官方 VAE 输出。
