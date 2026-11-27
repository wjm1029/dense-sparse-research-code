"""NEW / UNTRAINED: JTA 22-joint multi-person bidirectional coordinate decoder.

不是旧的单人 24-joint checkpoint 兼容模块。一个 slot 对应一个 clip 内的轨迹，
不是固定的标注 person ID；Hungarian matching 在整段 clip 上只做一次。
"""
import torch
from torch import nn


class PersonBlock(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.spatial = nn.TransformerDecoderLayer(hidden, heads, hidden*4,
            dropout=0., batch_first=True, norm_first=True)
        self.temporal_norm = nn.LayerNorm(hidden)
        self.temporal = nn.MultiheadAttention(hidden, heads, dropout=0., batch_first=True)

    def forward(self, queries, memory):
        b,t,q,h = queries.shape
        queries = self.spatial(queries.reshape(b*t,q,h), memory.flatten(0,1)).view(b,t,q,h)
        tracks = queries.permute(0,2,1,3).reshape(b*q,t,h)
        z = self.temporal_norm(tracks)
        # 双向时间 attention，无 causal mask，无历史回灌。
        tracks = tracks + self.temporal(z,z,z,need_weights=False)[0]
        return tracks.view(b,q,t,h).permute(0,2,1,3)


class MultiPersonDecode(nn.Module):
    def __init__(self, dim=3072, hidden=256, queries=128, layers=3, heads=8,
                 grid=(15,26), latent_frames=21, tile_size=(832,480)):
        super().__init__()
        self.grid, self.latent_frames, self.tile_size = tuple(grid), latent_frames, tuple(tile_size)
        self.num_queries, self.joints = queries, 22
        self.norm = nn.LayerNorm(dim)
        self.project = nn.Linear(dim,hidden)
        self.position = nn.Linear(4,hidden)  # tile xy + calibrated camera ray xy
        self.person_queries = nn.Parameter(torch.randn(queries,hidden)*.02)
        self.time = nn.Parameter(torch.randn(latent_frames,hidden)*.02)
        self.phase = nn.Embedding(5,hidden)
        self.blocks = nn.ModuleList([PersonBlock(hidden,heads) for _ in range(layers)])
        self.output = nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden*2),nn.GELU())
        self.xy = nn.Linear(hidden*2,22*2)
        self.xyz = nn.Linear(hidden*2,22*3)
        self.presence = nn.Linear(hidden*2,1)
        self.in_crop = nn.Linear(hidden*2,22)
        self.visible = nn.Linear(hidden*2,22)

    def forward(self, features, intrinsic, task):
        if task not in (0,1): raise ValueError('task must be 0=2D or 1=3D')
        b,n,d=features.shape;h,w=self.grid;t=self.latent_frames
        if n!=t*h*w or intrinsic.shape!=(b,3,3):
            raise ValueError('Feature grid or crop intrinsic shape mismatch')
        memory=self.project(self.norm(features)).view(b,t,h*w,-1)
        k=intrinsic.to(device=features.device,dtype=memory.dtype)
        if not torch.isfinite(k).all() or (k[:,[0,1],[0,1]]<=0).any():
            raise ValueError('Invalid crop camera intrinsics')
        yy,xx=torch.meshgrid(torch.arange(h,device=features.device,dtype=memory.dtype),
                             torch.arange(w,device=features.device,dtype=memory.dtype),indexing='ij')
        width,height=self.tile_size
        u=(xx.flatten()+.5)*(width/w);v=(yy.flatten()+.5)*(height/h)
        xy=torch.stack((u/width*2-1,v/height*2-1),-1)[None].expand(b,-1,-1)
        rays=torch.stack(((u[None]-k[:,0,2,None])/k[:,0,0,None],
                          (v[None]-k[:,1,2,None])/k[:,1,1,None]),-1)
        memory=memory+self.position(torch.cat((xy,rays),-1))[:,None]+self.time[None,:,None]
        q=self.person_queries[None,None].expand(b,t,-1,-1)+self.time[None,:,None]
        for block in self.blocks:q=block(q,memory)
        # 原始帧 0 对应 latent0；其余每个 latent 产生 4 帧，21 -> 81。
        raw=torch.arange(1+4*(t-1),device=features.device)
        index=torch.where(raw==0,0,(raw-1)//4+1)
        phase=torch.where(raw==0,4,(raw-1)%4)
        q=q[:,index]+self.phase(phase)[None,:,None]
        z=self.output(q);length=len(raw)
        result=dict(xy=self.xy(z).view(b,length,self.num_queries,22,2).float(),
            presence_logits=self.presence(z).squeeze(-1).float(),
            in_crop_logits=self.in_crop(z).float(),visible_logits=self.visible(z).float())
        # 3D 分支保留辅助 2D 输出，用于人物匹配和将 3D 骨骼放回图像。
        # 不假设两次不同任务 forward 的 slot 索引具有同一身份。
        if task==1:
            xyz=self.xyz(z).view(b,length,self.num_queries,22,3).float()
            root=(xyz[:,:,:,16:17]+xyz[:,:,:,19:20])*.5
            result['xyz']=xyz-root
        return result


@torch.no_grad()
def decode_people(prediction, tile_origins, tile_size=(832,480), threshold=.5):
    """逐 batch/逐帧输出人数可变的骨骼。仅做坐标还原，不伪装已实现跨块跟踪。"""
    b,t,q=prediction['presence_logits'].shape
    if tile_origins.shape!=(b,2): raise ValueError('tile_origins must be [B,2]')
    score=prediction['presence_logits'].sigmoid()
    xy=prediction['xy']*prediction['xy'].new_tensor(tile_size)
    xy=xy+tile_origins.to(xy)[:,None,None,None,:]
    result=[]
    for bi in range(b):
        frames=[]
        for ti in range(t):
            selected=(score[bi,ti]>=threshold).nonzero().flatten()
            row=dict(slot=selected,score=score[bi,ti,selected],xy_pixels=xy[bi,ti,selected],
                in_crop_probability=prediction['in_crop_logits'][bi,ti,selected].sigmoid(),
                visible_probability=prediction['visible_logits'][bi,ti,selected].sigmoid())
            if 'xyz' in prediction:row['xyz_root_relative_m']=prediction['xyz'][bi,ti,selected]
            frames.append(row)
        result.append(frames)
    return result
