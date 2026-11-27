"""Locate complete historical training entry points without running legacy jobs."""
import argparse
import json
from package_runtime import setup,NATIVE,PACKAGE
setup()

STAGES={
 'dense-bidirectional':dict(trainer='train_wan22_dense6_dispersion_ddp.py',model='Dense6LoRAModule',steps=120000),
 'dense-causal':dict(trainer='train_wan22_dense6_causal_ddp.py',driver='run_wan22_dense6_causal_ar_stage.py',model='CausalDense6LoRAModule',steps=45000),
 'dense-ar':dict(trainer='train_wan22_dense6_ar_tf_ddp.py',driver='run_wan22_dense6_causal_ar_stage.py',model='ARTeacherForcingDense6LoRAModule',steps=43500),
 'dense-self-forcing':dict(trainer='train_wan22_dense6_long477_ddp.py',model='wan22_dense6_long477.py:LongDenseModule',steps=4800),
 'dense-opsd':dict(trainer='train_wan22_dense6_long477_opsd_ddp.py',model='wan22_dense6_long477_opsd.py:Dense6OPSDModule',steps=5250),
 'historical-sparse-bidirectional':dict(trainer='train_wan22_sparse_prompt_full_ddp.py',model='PromptSparseModule',steps=25000),
 'historical-sparse-causal':dict(trainer='train_wan22_sparse_causal_ar_ddp.py',model='wan22_sparse_causal_ar.py:CausalARSparseModule',steps=22500),
 'historical-sparse-ar':dict(trainer='train_wan22_sparse_causal_ar_ddp.py',model='wan22_sparse_causal_ar.py:CausalARSparseModule',steps=21750),
 'historical-sparse-self-forcing':dict(trainer='train_wan22_sparse_long477_ddp.py',phase='self_forcing',model='wan22_sparse_long477.py:LongSparseModule',steps=2400),
 'historical-sparse-opsd':dict(trainer='train_wan22_sparse_long477_ddp.py',phase='opsd_local',model='wan22_sparse_long477.py:LongSparseModule',steps=2625),
 'jta-bidirectional':dict(trainer='tools/jta_train.py',model='core/sparse_model.py:build_fresh_jta_model',steps=None),
}


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=STAGES);a=p.parse_args()
 values={a.stage:STAGES[a.stage]} if a.stage else STAGES
 for key,value in values.items():
  base=PACKAGE if key=='jta-bidirectional' else NATIVE
  value=dict(value,absolute_trainer=str(base/value['trainer']))
  if not (base/value['trainer']).is_file():raise FileNotFoundError(value['absolute_trainer'])
  print(json.dumps({key:value},indent=2))
