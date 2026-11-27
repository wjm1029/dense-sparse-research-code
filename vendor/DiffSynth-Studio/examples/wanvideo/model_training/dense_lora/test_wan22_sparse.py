import torch
from train_wan22_sparse_ddp import JointHead, loss_and_metrics

torch.manual_seed(42)
head = JointHead(dim=32,hidden=32)
features = torch.randn(1,21*390,32,requires_grad=True)
xy,xyz = head(features)
assert xy.shape == (1,81,24,2) and xyz.shape == (1,81,24,3)
mask = torch.ones(1,81,24,dtype=torch.bool); mask[:,0,0] = False
target2 = torch.randn_like(xy); target3 = torch.randn_like(xyz)
loss,*_ = loss_and_metrics(xy,xyz,target2,target3,mask)
changed = target2.clone(); changed[:,0,0] += 10000
other,*_ = loss_and_metrics(xy,xyz,changed,target3,mask)
assert torch.equal(loss,other), 'Invisible 2D label leaked into loss'
loss.backward()
assert features.grad is not None and torch.isfinite(features.grad).all() and features.grad.abs().sum() > 0
assert head.queries.grad.abs().sum() > 0
zeros = torch.zeros_like(target2); zeros3 = torch.zeros_like(target3)
total,l2,l3,e2,e3 = loss_and_metrics(zeros,zeros3,zeros,zeros3,mask)
assert total == l2 == l3 == e2 == e3 == 0
print('PASS shapes=[B,81,24,2/3], masked loss, gradient through head to DiT features, zero-error metrics',flush=True)
