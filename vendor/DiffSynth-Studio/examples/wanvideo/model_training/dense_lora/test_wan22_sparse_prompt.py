import torch
from torch import nn
from train_wan22_sparse_prompt_ddp import ConditionalHead,coordinate_loss
from wan22_sparse_task_router import SparseTaskRouter

torch.manual_seed(42)
head = ConditionalHead(dim=32,hidden=32)
features = torch.randn(1,21*390,32,requires_grad=True)
xy = torch.randn(1,81,24,2); xyz = torch.randn(1,81,24,3)
mask = torch.ones(1,81,24,dtype=torch.bool); mask[:,0,0] = False
for task,d in [(0,2),(1,3)]:
    head.zero_grad(set_to_none=True); pred = head(features,task)
    assert pred.shape == (1,81,24,d)
    loss,error = coordinate_loss(pred,xy,xyz,mask,task); loss.backward()
    assert head.readouts[task].weight.grad is not None
    assert head.readouts[1-task].weight.grad is None, 'Unselected head must not run/train'
    assert torch.isfinite(features.grad).all()
    if task == 0:
        changed = xy.clone(); changed[:,0,0] += 1000
        other,_ = coordinate_loss(pred.detach(),changed,xyz,mask,task)
        assert torch.equal(loss.detach(),other)
router = SparseTaskRouter(['site'],rank=4)
table = torch.randn(2,4096); router.set_task_feature_table(table)
gates = router(torch.tensor([0,1])); assert torch.equal(gates,torch.ones_like(gates))
assert tuple(router.task_feature_table.shape) == (2,4096)
restored = SparseTaskRouter(['site'],rank=4); restored.load_state_dict(router.state_dict())
assert torch.equal(restored(torch.tensor([0,1])),gates)
try: router(torch.tensor([2]))
except ValueError: pass
else: raise AssertionError('Invalid sparse task accepted')
with torch.no_grad(): router.layernets.weight.normal_(0,0.01)
different = router(torch.tensor([0,1]))
assert not torch.equal(different[0],different[1]), 'Task features must affect route'
different.sum().backward(); assert router.layernets.weight.grad.abs().sum() > 0
print('PASS prompt-specific 2/3D shapes, unselected-head no gradient, mask, distinct task embedding routes, identity init, router reload',flush=True)

