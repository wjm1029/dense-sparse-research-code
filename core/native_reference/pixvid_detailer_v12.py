"""PixVid Appendix A.2 detailer, adapted to Dense6's 4x32x32 tokens.

Five shared-grid up blocks; no old MLP, pixel residual, VAE, or spatial tiling.
This stage warms up the head on frozen, timestep-modulated native features.
It is not the paper's framewise input conversion or joint DiT finetuning.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class CausalConv(nn.Conv3d):
    def __init__(self, cin, cout, spatial=1):
        super().__init__(cin, cout, (3, spatial, spatial), padding=(0, spatial//2, spatial//2))

    def forward(self, x, state=None, key=None):
        if state is None:
            return super().forward(F.pad(x, (0, 0, 0, 0, 2, 0)))
        past = state.get(key)
        joined = F.pad(x, (0, 0, 0, 0, 2, 0)) if past is None else torch.cat((past, x), 2)
        state[key] = joined[:, :, -2:].detach()
        return super().forward(joined)


class UpBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.up = nn.ConvTranspose3d(cin, cout, (1, 2, 2), stride=(1, 2, 2))
        self.mix = CausalConv(cout, cout, 3)

    def forward(self, x, state=None, key=None):
        return F.silu(self.mix(self.up(x), state, key))


class Detailer(nn.Module):
    widths = (1024, 512, 256, 128, 64)
    temporal_context = 14  # bottleneck + five up blocks + output, each two past frames
    spatial_halo = 32     # exact output receptive-field radius is <=31 pixels

    def __init__(self, dim=3072, widths=None, checkpoint_blocks=True):
        super().__init__()
        widths = self.widths if widths is None else tuple(widths)
        self.dim = dim
        # Native features already contain the fixed t=0 output-head modulation.
        # Five phase offsets distinguish frame0 and the four raw frames/token.
        self.phase = nn.Embedding(5, dim)
        nn.init.normal_(self.phase.weight, std=.01)
        self.bottleneck = CausalConv(dim, dim)
        self.blocks = nn.ModuleList([UpBlock(a, b) for a, b in zip((dim,)+widths[:-1], widths)])
        self.output = CausalConv(widths[-1], 1)
        self.checkpoint_blocks = checkpoint_blocks

    def forward(self, features, frame_ids, state=None):
        # features [B,T,Htok,Wtok,D], already expanded to corresponding raw frames.
        phase = torch.where(frame_ids == 0, 4, (frame_ids-1).remainder(4))
        x = features.permute(0, 4, 1, 2, 3)
        e = self.phase(phase).permute(1, 0)[None, :, :, None, None]
        x = self.bottleneck(F.silu(x + e), state, 'bottleneck')
        for i, block in enumerate(self.blocks):
            if self.training and self.checkpoint_blocks and state is None:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x, state, f'up{i}')
        # Preserve sub-BF16 depth differences in final scalar prediction.
        with torch.autocast('cuda', enabled=False):
            return self.output(x.float(), state, 'output')[:, 0]


def expand_features(features, frame_ids):
    idx = torch.where(frame_ids == 0, 0, (frame_ids-1)//4+1)
    return features[idx][None]


@torch.no_grad()
def infer_video(model, features, chunk=4):
    model.eval()
    features = features.cuda()
    state = {}
    result = []
    for start in range(0, 81, chunk):
        ids = torch.arange(start, min(start+chunk, 81), device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            result.append(model(expand_features(features, ids), ids, state)[0])
    return torch.cat(result)
