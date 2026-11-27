import os

import torch.distributed as dist


dist.init_process_group("gloo")
print(f"rank={os.environ['RANK']} gloo=ok", flush=True)
dist.barrier()
dist.destroy_process_group()
