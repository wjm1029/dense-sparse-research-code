"""CPU integration tests for useful workflow invariants, without official weights."""
import json
from pathlib import Path
import tempfile
import unittest
from package_runtime import setup
setup()
import numpy as np
import torch
from torch import nn
from jta_cache import atomic_json, atomic_torch, sha, finalize
from jta_train import CacheDataset, choose_index, adapter_state, load_adapter, pose_metrics
from jta_infer import merge_tiles


class AdapterToy(nn.Module):
    def __init__(self):
        super().__init__();self.frozen=nn.Linear(2,2).requires_grad_(False)
        self.sparse_head=nn.Linear(2,2);self.task_router=nn.Linear(2,2)
        self.task_router.register_buffer('task_feature_table',torch.randn(2,3))
        self.register_buffer('context',torch.randn(2,4,3))


class WorkflowTests(unittest.TestCase):
    def test_adapter_and_optimizer_paired_resume(self):
        torch.manual_seed(3);model=AdapterToy();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        x=torch.randn(2,2)
        def update(m,o):
            o.zero_grad();loss=m.sparse_head(x).square().mean()+m.task_router(x).square().mean()
            loss.backward();o.step()
        update(model,opt)
        with tempfile.TemporaryDirectory() as folder:
            file=Path(folder)/'paired.pt';atomic_torch(file,dict(weights=adapter_state(model),optimizer=opt.state_dict()))
            resume=AdapterToy();other=torch.optim.AdamW([p for p in resume.parameters() if p.requires_grad],lr=.01)
            state=torch.load(file,weights_only=False);load_adapter(resume,state['weights']);other.load_state_dict(state['optimizer'])
            update(model,opt);update(resume,other)
            for name,value in adapter_state(model).items():torch.testing.assert_close(value,adapter_state(resume)[name],rtol=0,atol=0)
            state['weights'].pop('context')
            with self.assertRaises(ValueError):load_adapter(resume,state['weights'])

    def test_cache_integrity_and_deterministic_sampling(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);atomic_json(root/'cache_config.json',dict(num_shards=1,selected_records=2))
            rows=[]
            for i in range(2):
                cid=f'clip{i}';relative=f'clips/{cid}.pt'
                atomic_torch(root/relative,dict(cache_config_sha256=sha(root/'cache_config.json'),row=dict(id=cid),
                    source_latent=torch.zeros(48,21,30,52,dtype=torch.bfloat16),targets={}))
                rows.append(dict(id=cid,split='train',sequence='seq0',empty=bool(i),track_count=0,cache_file=relative,sha256=sha(root/relative)))
            file=root/'cache_shard_0000.jsonl';file.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            atomic_json(root/'shard_0000_COMPLETE.json',dict(manifest_sha256=sha(file)))
            finalize(root);dataset=CacheDataset(root,'train')
            self.assertEqual(len(dataset),2);self.assertEqual(dataset[0][2]['id'],'clip0')
            first=[choose_index(dataset,s,0,2,13,.1) for s in range(100)]
            self.assertEqual(first,[choose_index(dataset,s,0,2,13,.1) for s in range(100)])
            self.assertEqual(choose_index(dataset,0,0,1,0,0),0);self.assertEqual(choose_index(dataset,0,0,1,0,1),1)
            bad=root/rows[1]['cache_file'];bad.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):dataset[1]

    def test_cross_tile_duplicate_suppression(self):
        xy=np.stack([np.linspace(100,120,22),np.linspace(100,160,22)],-1).tolist()
        def person(tile,score,dx=0):return dict(tile=tile,slot=0,source_tiles=[tile],score=score,
            xy_pixels=(np.asarray(xy)+[dx,0]).tolist(),in_crop_probability=[1.]*22,visible_probability=[1.]*22)
        merged=merge_tiles([person(0,.9),person(1,.8,1),person(1,.7,30)])
        self.assertEqual(len(merged),2);self.assertEqual(merged[0]['source_tiles'],[0,1])
        self.assertEqual(len(merge_tiles([person(0,.9),person(0,.8,1)])),2)

    def test_real_metric_units(self):
        present=torch.tensor([[True]])
        gt=dict(present=present,visible=torch.ones(1,1,22,dtype=torch.bool),valid_3d=torch.ones(1,1,22,dtype=torch.bool),
            xy=torch.zeros(1,1,22,2),xyz=torch.zeros(1,1,22,3))
        pred=dict(xy=torch.zeros(1,1,1,22,2),xyz=torch.zeros(1,1,1,22,3),presence_logits=torch.ones(1,1,1)*10)
        pred['xy'][...,0]=1/832;pred['xyz'][...,2]=.001
        metrics=pose_metrics(pred,gt,(torch.tensor([0]),torch.tensor([0])))
        self.assertAlmostEqual(metrics['matched_visible_2d_distance_sum']/22,1.,places=5)
        self.assertAlmostEqual(metrics['matched_3d_distance_mm_sum']/22,1.,places=5)
        self.assertEqual(metrics['count_mae'],0.)


if __name__=='__main__':unittest.main(verbosity=2)
