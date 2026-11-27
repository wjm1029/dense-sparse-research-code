"""Dense6 long self-forcing: inherited AR history, six task routes and DispLoss."""
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from train_wan22_dense6_dispersion_ddp import Dense6LoRAModule
from wan22_cache_integrity import sha256_file as sha
from wan22_dense6_task_router import TASK_TO_ID, ID_TO_TASK, TASK_INSTRUCTIONS
from wan22_dense6_dispersion import feature_dispersion
from wan22_masked_loss import masked_latent_mse
from wan22_causal_attention import install_causal_attention
from wan22_ar_teacher_forcing import ARTeacherForcingConfig, install_ar_history_adapter, combine_rgb_and_history
from wan22_ar_self_forcing import training_streaming_forward, detach_kv_cache

SHAPE=(48,120,30,52)
MAIN=Path('/pfs/mayuema/dense/wan/cache/wan22_long477_rgb_depth_normal_matte_v1_bf16_bright40')
NORMAL=Path('/pfs/mayuema/dense/wan/cache/wan22_long477_normal_flip_yz_v1_bf16_bright40')

def manifest(root,complete=True):
    m=json.loads((root/'manifest.json').read_text())
    if complete:
        receipt=json.loads((root/'CACHE_COMPLETE').read_text())
        assert receipt['manifest_sha256']==sha(root/'manifest.json')
        assert receipt['cache_contract_hash']==m['cache_contract_hash']
    return m

class LongDenseDataset(Dataset):
    def __init__(self,extra,prompts,validation=False,smoke=False):
        self.extra=Path(extra); self.prompts=prompts
        main=manifest(MAIN); normal=manifest(NORMAL); new=manifest(self.extra,not smoke)
        assert main['source_index_sha256']==normal['source_index_sha256']==new['source_index_sha256']==sha(MAIN/'source_index.json')
        assert new['source_manifest_sha256']==sha(MAIN/'manifest.json')
        assert normal['normal_transform_before_vae']==['x','-y','-z']
        assert new['schema']=='wan22_dense6_long477_extra_v1' and new['flow_scale_px']==32
        self.contracts={MAIN:main['cache_contract_hash'],NORMAL:normal['cache_contract_hash'],self.extra:new['cache_contract_hash']}
        rows=json.loads((MAIN/'source_index.json').read_text())['samples']
        if smoke: self.rows=rows[:4]
        else:
            self.rows=[r for r in rows if (int(hashlib.sha256(('dense6-long-v1:'+r['id']).encode()).hexdigest()[:8],16)%20==0)==validation]
        assert len(self.rows)>=4
        self.identities={str(r):sha(r/'manifest.json') for r in self.contracts}
        for row in self.rows:
            for task in ('source',*TASK_TO_ID):
                root=MAIN if task in ('source','depth','matte') else NORMAL if task=='normal' else self.extra
                assert (root/task/(row['id']+'.pt')).is_file(), (row['id'],task)

    def __len__(self): return len(self.rows)*6
    def _read(self,root,task,cid):
        p=torch.load(root/task/(cid+'.pt'),map_location='cpu',weights_only=False)
        assert p['clip_id']==cid and p['modality']==task and p['cache_contract_hash']==self.contracts[root]
        z=p['latent']; assert z.shape==SHAPE and z.dtype==torch.bfloat16 and torch.isfinite(z).all()
        if task!='source': assert p['task_id']==TASK_TO_ID[task]
        if task=='normal': assert p['normal_transform']==['x','-y','-z']
        return z,p.get('prompt')
    def __getitem__(self,i):
        row=self.rows[i//6]; task=ID_TO_TASK[i%6]; cid=row['id']
        root=MAIN if task in ('depth','matte') else NORMAL if task=='normal' else self.extra
        source,_=self._read(MAIN,'source',cid); target,prompt=self._read(root,task,cid)
        expected=f"output: {row['subject_label']} alpha matte" if task=='matte' else TASK_INSTRUCTIONS[i%6]
        assert prompt==expected
        return source,target,i%6,self.prompts.index(prompt),cid

class LongDenseModule(Dense6LoRAModule):
    def __init__(self,model_dir):
        super().__init__(model_dir,32,'q,k,v,o,ffn.0,ffn.2',512,'learned',.001)
        self.ar_config=ARTeacherForcingConfig(120,3,48)
        self.causal_controller=install_causal_attention(self.pipe.dit,latent_shape=SHAPE,frames_per_chunk=3,local_attn_frames=-1,sink_frames=0)
        self.ar_controller=install_ar_history_adapter(self.pipe.dit,self.causal_controller,self.ar_config)
        self.dispersion_weight=.01; self.dispersion_tau=1.

    def forward(self,rgb,history,task,prompt,offset,caches,with_dispersion=True):
        captured=[]
        hook=self.pipe.dit.blocks[8].register_forward_hook(lambda block,args,out:captured.append(out)) if with_dispersion else None
        try:
            gates=self.task_router(task)
            timestep=torch.zeros(rgb.shape[0],device=rgb.device,dtype=torch.bfloat16)
            packed=combine_rgb_and_history(rgb,history,self.ar_config)
            pred=-training_streaming_forward(self.causal_controller,self.pipe.dit,packed,timestep,self.context_table[prompt],frame_offset=offset,caches=caches)
        finally:
            if hook: hook.remove()
        if with_dispersion:
            assert len(captured)==1 and captured[0].requires_grad
            dispersion,pairs,distance=feature_dispersion(captured[0],prompt,1.)
        else: dispersion=pred.float().sum()*0
        return pred,self.task_router.identity_regularization(gates),dispersion

    @torch.no_grad()
    def rollout(self,source,task,prompt):
        # The reused stream math requires dit.train(); no dropout in this Wan,
        # and no_grad disables autograd. Keep the containing module in eval mode.
        assert not self.training
        self.pipe.dit.train()
        caches=self.causal_controller.new_kv_cache(); history=torch.zeros_like(source[:,:,:3]); result=[]
        try:
            for offset in range(0,120,3):
                pred,_,_=self(source[:,:,offset:offset+3],history,task,prompt,offset,caches,False)
                result.append(pred); history=pred.detach(); self.task_router.clear()
        finally: self.pipe.dit.eval()
        return torch.cat(result,2)
