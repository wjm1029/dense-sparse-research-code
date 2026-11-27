import torch
import torch.distributed as dist
from wan22_dense6_dispersion import SameTaskSampler, feature_dispersion

dist.init_process_group('gloo')
rank, world = dist.get_rank(), dist.get_world_size()
torch.manual_seed(9)
all_z = torch.randn(world, 2, 8)
z = all_z[rank:rank+1].clone().requires_grad_()
loss, pairs, distance = feature_dispersion(z, torch.tensor([0]))
loss.backward()
reference = all_z.clone().requires_grad_()
flat = reference.flatten(1)
d = (flat[:, None] - flat[None, :]).square().mean(-1)
expected = torch.logsumexp(-d.flatten(), 0) - torch.tensor(world**2).float().log()
expected.backward()
torch.testing.assert_close(loss, expected)
torch.testing.assert_close(z.grad/world, reference.grad[rank:rank+1], rtol=1e-5, atol=1e-6)
assert pairs == world*(world-1)//2
z = all_z[rank:rank+1].clone().requires_grad_()
zero, pairs, _ = feature_dispersion(z, torch.tensor([rank]))
zero.backward()
assert pairs == 0 and zero.item() == 0 and z.grad.abs().max().item() == 0
streams = [list(SameTaskSampler(10, num_replicas=4, rank=r, seed=1)) for r in range(4)]
for batch in zip(*streams):
    assert len({x % 6 for x in batch}) == 1
    assert len({x // 6 for x in batch}) == 4
assert {x for stream in streams for x in stream} == set(range(60))
if rank == 0:
    print('PASS: global dispersion gradient, diagonal/no-pair zero, same-task distinct-clip sampler', flush=True)
dist.destroy_process_group()
