"""Dense6 OPSD: bounded student trajectory, EMA delayed clean-prefix teacher."""
import torch
from wan22_dense6_long477 import LongDenseModule
from wan22_opsd_bounded_causal_attention import install_opsd_bounded_causal_attention
from wan22_opsd_prefix_cache import clone_prefix,replace_chunk_from_replay,swap_trainable_parameters_

class Dense6OPSDModule(LongDenseModule):
    def __init__(self,model_dir):
        super().__init__(model_dir)
        self.causal_controller=install_opsd_bounded_causal_attention(
            self.pipe.dit,latent_shape=(48,120,30,52),frames_per_chunk=3,
            local_attn_frames=12,sink_frames=3)
        self.ar_controller.causal_controller=self.causal_controller

def teacher_forward(module,rgb,target,history,task,prompt,offset,caches,ema,named):
    # This runs before the student forward/backward, so parameter version
    # changes cannot invalidate a retained student graph.
    backup=swap_trainable_parameters_(named,ema)
    try:
        with torch.no_grad():
            pred,_,_=module(rgb[:,:,offset:offset+3],history,task,prompt,offset,caches,False)
            module.task_router.clear()
            if offset>=3:
                previous=offset-3
                start=previous*module.causal_controller.config.tokens_per_frame
                replay=clone_prefix(caches,start)
                clean=torch.zeros_like(history) if previous==0 else target[:,:,previous-3:previous]
                module(rgb[:,:,previous:previous+3],clean,task,prompt,previous,replay,False)
                module.task_router.clear()
                replace_chunk_from_replay(caches,replay,start_token=start,
                    chunk_tokens=module.causal_controller.config.chunk_tokens)
        return pred.detach()
    finally:
        with torch.no_grad():
            for name,q in named: q.copy_(backup[name])
        module.task_router.clear()
