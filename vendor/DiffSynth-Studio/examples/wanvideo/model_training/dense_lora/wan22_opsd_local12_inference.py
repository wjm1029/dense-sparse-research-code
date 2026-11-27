"""Inference wrapper for local-12 OPSD checkpoints using the fixed KV pool."""

from __future__ import annotations

import train_wan22_ar_tf_ddp as ar_train
from wan22_ar_teacher_forcing import install_ar_history_adapter
from wan22_prerope_kv_pool import install_prerope_fixed_pool_attention


class OPSDLocal12InferenceModule(ar_train.ARTeacherForcingDense4LoRAModule):
    """Construct the dense model with local-12 pre-RoPE streaming attention."""

    def __init__(self, *args, **kwargs):
        # Bypass the parent initializer because it installs full-prefix causal
        # attention.  The remaining setup is shared with the training wrapper.
        ar_train.dense4.Dense4LoRAModule.__init__(self, *args, **kwargs)
        self.causal_controller = install_prerope_fixed_pool_attention(
            self.pipe.dit,
            latent_shape=ar_train.dense4.LATENT_SHAPE,
            frames_per_chunk=ar_train.AR_CONFIG.frames_per_chunk,
            local_attn_frames=12,
            sink_frames=3,
        )
        self.ar_controller = install_ar_history_adapter(
            self.pipe.dit, self.causal_controller, ar_train.AR_CONFIG
        )
