"""CPU checks: gating gradients, set matching, masks, empty scenes, decode parity."""
import argparse
import copy
import json
from pathlib import Path
import unittest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from task_gating import JTATaskRouter, discover_lora_sites
from sparse_decode import MultiPersonDecode, decode_people
from sparse_loss import sparse_objective, load_target_npz, weighted_mse
from dense_decode import DensePixelDecode, decode_native_features, load_direct_head
from dense_loss import dispersion, dense_bidirectional_loss
from native_reference.pixvid_detailer_v12 import Detailer as OriginalDepth

torch.set_num_threads(2)


class ToyProjection(nn.Module):
    def __init__(self):
        super().__init__();self.base=nn.Linear(6,5).requires_grad_(False)
        self.lora_A=nn.ModuleDict({'default':nn.Linear(6,2,bias=False)})
        self.lora_B=nn.ModuleDict({'default':nn.Linear(2,5,bias=False)})
    def forward(self,x):return self.base(x)+self.lora_B['default'](self.lora_A['default'](x))


def target(t=9,n=2):
    present=torch.ones(t,n,dtype=torch.bool)
    return dict(xy=torch.rand(t,n,22,2),xyz=torch.randn(t,n,22,3)*.1,
        present=present,in_crop=present[...,None].expand(t,n,22).clone(),
        visible=present[...,None].expand(t,n,22).clone(),
        valid_3d=present[...,None].expand(t,n,22).clone(),
        intrinsic=torch.tensor([[1158.,0,416],[0,1158,240],[0,0,1]]))


class CoreTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(123)

    def test_rank_gate_and_checkpoint_backward(self):
        dit=nn.Sequential(ToyProjection());sites=discover_lora_sites(dit)
        router=JTATaskRouter([n for n,_ in sites],rank=2,task_feature_dim=8,hidden_dim=8)
        router.set_task_feature_table(torch.randn(2,8));router.bind(dit)
        x=torch.randn(2,4,6,requires_grad=True);task=torch.tensor([0,1])
        gates=router(task);self.assertTrue(torch.equal(gates,torch.ones_like(gates)))
        proj=dit[0]
        manual=proj.base(x)+(x@proj.lora_A['default'].weight.T)@proj.lora_B['default'].weight.T
        torch.testing.assert_close(dit(x),manual)
        nn.init.normal_(router.layernets.weight,std=.2)
        gates=router(task)
        manual=proj.base(x)+((x@proj.lora_A['default'].weight.T)*gates[:,0,None])@proj.lora_B['default'].weight.T
        torch.testing.assert_close(dit(x),manual)
        self.assertTrue((gates>=.5).all() and (gates<=1.5).all())
        checkpoint(dit,x,use_reentrant=False).square().mean().backward()
        self.assertGreater(float(router.layernets.weight.grad.abs().sum()),0)
        self.assertIsNone(router.task_feature_table.grad)
        self.assertIsNone(proj.base.weight.grad)
        router.clear()
        with self.assertRaises(RuntimeError):dit(x)

    def test_sparse_shapes_gradients_and_permutation_invariance(self):
        model=MultiPersonDecode(dim=16,hidden=16,queries=4,layers=1,heads=4,grid=(2,3),latent_frames=3)
        gt=target();features=torch.randn(1,18,16,requires_grad=True)
        for task in (0,1):
            model.zero_grad(set_to_none=True)
            pred=model(features,gt['intrinsic'][None],task)
            self.assertEqual(pred['xy'].shape,(1,9,4,22,2))
            if task==1:torch.testing.assert_close((pred['xyz'][:,:,:,16]+pred['xyz'][:,:,:,19])*.5,torch.zeros(1,9,4,3),atol=1e-7,rtol=0)
            gates=torch.ones(1,2,2,requires_grad=True)
            loss,_,pairs=sparse_objective(pred,[gt],task,gates)
            swapped={k:(v[:,[1,0]] if k!='intrinsic' else v) for k,v in gt.items()}
            permuted,_,_=sparse_objective(pred,[swapped],task,gates)
            torch.testing.assert_close(loss,permuted)
            self.assertEqual(len(pairs[0][0]),2)
            loss.backward()
            self.assertGreater(float(model.person_queries.grad.abs().sum()),0)
            self.assertGreater(float(model.xy.weight.grad.abs().sum()),0)
            if task==1:self.assertGreater(float(model.xyz.weight.grad.abs().sum()),0)

    def test_empty_scene_and_capacity(self):
        model=MultiPersonDecode(dim=8,hidden=8,queries=2,layers=1,heads=2,grid=(1,1),latent_frames=1)
        empty=target(t=1,n=0)
        pred=model(torch.randn(1,1,8),empty['intrinsic'][None],1)
        loss,metrics,pairs=sparse_objective(pred,[empty],1,torch.ones(1,1,2))
        loss.backward();self.assertTrue(torch.isfinite(loss));self.assertEqual(len(pairs[0][0]),0)
        self.assertEqual(float(metrics['xy']),0)
        self.assertGreater(float(model.presence.bias.grad),0)  # 梯度下降降低空图存在概率
        with self.assertRaises(ValueError):sparse_objective(pred,[target(t=1,n=3)],1,torch.ones(1,1,2))

    def test_masked_invalid_values(self):
        p=torch.ones(1,2,3,requires_grad=True)
        gt=torch.tensor([[[0.,0,0],[float('nan')]*3]])
        loss=weighted_mse(p,gt,torch.tensor([[1.,0.]]));loss.backward()
        self.assertEqual(float(loss),1.);self.assertTrue(torch.isfinite(p.grad).all())
        self.assertEqual(float(p.grad[0,1].sum()),0.)

    def test_dense_streaming_native_parity_and_causality(self):
        widths=(8,6,4,3,2)
        native=OriginalDepth(dim=8,widths=widths,checkpoint_blocks=False).eval()
        head=DensePixelDecode('depth',dim=8,widths=widths,checkpoint_blocks=False).eval()
        head.load_state_dict(native.state_dict(),strict=True)
        features=torch.randn(1,5,1,1,8);ids=torch.arange(5)
        with torch.no_grad():
            got=head(features,ids);original=native(features,ids)[:,:,None]
            torch.testing.assert_close(got,original,atol=0,rtol=0)
            state={};stream=torch.cat([head(features[:,i:i+2],ids[i:i+2],state) for i in range(0,5,2)],1)
            torch.testing.assert_close(got,stream,atol=2e-5,rtol=1e-5)
            altered=features.clone();altered[:,3:]+=20
            torch.testing.assert_close(got[:,:3],head(altered,ids)[:,:3],atol=0,rtol=0)
            compressed=torch.randn(1,3,1,1,8)
            result=decode_native_features(head,compressed,chunk=2)
            self.assertEqual(result.shape,(1,9,1,32,32))

    def test_dense_sign_mask_and_dispersion(self):
        velocity=torch.randn(2,3,2,2,2,requires_grad=True);gt=-velocity.detach()
        mask=torch.ones(2,1,2,2,2);f=torch.randn(2,3,4,requires_grad=True)
        loss,terms=dense_bidirectional_loss(velocity,gt,mask,torch.ones(2,1,2),f,torch.zeros(2,dtype=torch.long))
        self.assertEqual(float(terms['latent']),0);self.assertLessEqual(float(terms['dispersion']),0)
        loss.backward();self.assertGreater(float(f.grad.abs().sum()),0)
        self.assertEqual(float(dispersion(f,torch.tensor([0,1]))),0.)

    def test_crop_decode_coordinate_roundtrip(self):
        pred=dict(xy=torch.tensor([[[[[.5,.5]]*22]]]),presence_logits=torch.tensor([[[10.]]]),
            in_crop_logits=torch.ones(1,1,1,22),visible_logits=torch.ones(1,1,1,22))
        row=decode_people(pred,torch.tensor([[544,300]]))[0][0]
        torch.testing.assert_close(row['xy_pixels'][0,0],torch.tensor([960.,540.]))


def check_real_pilot(path):
    directory=Path(path);manifest=json.loads((directory/'manifest.json').read_text())
    model=MultiPersonDecode(dim=32,hidden=32,queries=128,layers=1,heads=4,grid=(2,3)).train()
    results=[]
    for row in manifest:
        gt=load_target_npz(directory/row['target_file']);features=torch.randn(1,21*6,32)
        pred=model(features,gt['intrinsic'][None],1)
        loss,_,pairs=sparse_objective(pred,[gt],1,torch.ones(1,1,2))
        model.zero_grad(set_to_none=True);loss.backward()
        results.append(dict(id=row['id'],tracks=len(pairs[0][0]),finite_loss=bool(torch.isfinite(loss))))
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pilot');parser.add_argument('--native-weights-root')
    parser.add_argument('--report');args=parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CoreTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(unit_tests=result.testsRun,passed=result.wasSuccessful(),scope='CPU functional checks, not model quality or full Wan training')
    if not result.wasSuccessful():raise SystemExit(1)
    if args.pilot:report['actual_jta_targets']=check_real_pilot(args.pilot)
    if args.native_weights_root:
        root=Path(args.native_weights_root);report['native_dense_head_load']={}
        for task in ('depth','matte','normal'):
            path=root/('pixvid_detailer_v12' if task=='depth' else 'pixvid_modalities_v13')/task/'head_30000.pt'
            head=load_direct_head(path,task,device='cpu')
            report['native_dense_head_load'][task]=dict(strict_load=True,parameters=sum(p.numel() for p in head.parameters()))
            del head
    if args.report:Path(args.report).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
