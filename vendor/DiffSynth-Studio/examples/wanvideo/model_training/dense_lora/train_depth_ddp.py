"""Four-GPU cached depth LoRA training for Wan2.1 1.3B."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from diffsynth.diffusion.training_module import DiffusionTrainingModule
from diffsynth.pipelines.wan_video import (
    ModelConfig,
    WanVideoPipeline,
    WanVideoUnit_PromptEmbedder,
)


CACHE_SCHEMA = "wan21_depth_latent_v1"
LATENT_SHAPE = (16, 21, 60, 104)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_train_steps", type=int, default=10_000)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-2)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_target_modules", default="q,k,v,o,ffn.0,ffn.2")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    return parser.parse_args()


class CachedDepthDataset(Dataset):
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        complete = self.cache_dir / "CACHE_COMPLETE"
        manifest_path = self.cache_dir / "manifest.json"
        if not complete.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Depth cache is not complete: {self.cache_dir}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema") != CACHE_SCHEMA:
            raise ValueError(f"Unsupported cache schema: {self.manifest.get('schema')!r}")
        if float(self.manifest.get("min_rgb_mean", -1)) != 40.0:
            raise ValueError("Formal depth training requires min_rgb_mean=40.0")
        self.paths = sorted(self.cache_dir.glob("*/*.pt"))
        expected = int(self.manifest["num_samples"])
        if len(self.paths) != expected:
            raise ValueError(f"Cache count mismatch: manifest={expected}, files={len(self.paths)}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        payload = torch.load(self.paths[index], map_location="cpu", weights_only=False)
        if payload.get("schema") != CACHE_SCHEMA:
            raise ValueError(f"Invalid cache schema: {self.paths[index]}")
        source = payload["source_latent"]
        target = payload["target_latent"]
        if tuple(source.shape) != LATENT_SHAPE or tuple(target.shape) != LATENT_SHAPE:
            raise ValueError(f"Invalid cached shape: {self.paths[index]}")
        return source, target


class DepthLoRAModule(DiffusionTrainingModule):
    def __init__(self, model_dir: str, lora_rank: int, target_modules: str, device: torch.device) -> None:
        super().__init__()
        model_dir = Path(model_dir).resolve()
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=[
                ModelConfig(path=str(model_dir / "diffusion_pytorch_model.safetensors")),
                ModelConfig(path=str(model_dir / "models_t5_umt5-xxl-enc-bf16.pth")),
            ],
            tokenizer_config=ModelConfig(path=str(model_dir / "google/umt5-xxl")),
        )
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models=None,
            lora_base_model="dit",
            lora_target_modules=target_modules,
            lora_rank=lora_rank,
            task="sft",
        )
        self.pipe.device = device

    @torch.no_grad()
    def prepare_context(self, prompt: str, device: torch.device) -> torch.Tensor:
        self.pipe.text_encoder.to(device)
        context = WanVideoUnit_PromptEmbedder().encode_prompt(self.pipe, prompt)
        context = context.detach().to(device=device, dtype=torch.bfloat16)
        self.pipe.text_encoder.to("cpu")
        self.pipe.text_encoder = None
        self.pipe.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        return context

    def forward(
        self,
        source_latent: torch.Tensor,
        target_latent: torch.Tensor,
        context: torch.Tensor,
        gradient_checkpointing: bool,
    ) -> torch.Tensor:
        timestep = torch.zeros(
            source_latent.shape[0], device=source_latent.device, dtype=torch.bfloat16
        )
        velocity = self.pipe.model_fn(
            dit=self.pipe.dit,
            latents=source_latent,
            timestep=timestep,
            context=context,
            use_gradient_checkpointing=gradient_checkpointing,
            use_gradient_checkpointing_offload=False,
            cfg_merge=False,
        )
        return F.mse_loss((-velocity).float(), target_latent.float())


def save_lora(module: DepthLoRAModule, output_dir: Path, step: int, args: argparse.Namespace) -> None:
    trainable = module.trainable_param_names()
    state = {
        name.removeprefix("pipe.dit."): tensor.detach().cpu().contiguous()
        for name, tensor in module.state_dict().items()
        if name in trainable
    }
    path = output_dir / f"step-{step}.safetensors"
    save_file(
        state,
        str(path),
        metadata={
            "task": "depth",
            "method": "genception_clean_source_t0_negated_velocity_latent_mse",
            "base_model": "Wan2.1-T2V-1.3B",
            "step": str(step),
            "world_size": str(dist.get_world_size()),
            "global_batch_size": str(dist.get_world_size()),
            "lora_rank": str(args.lora_rank),
            "lora_targets": args.lora_target_modules,
            "min_rgb_mean": "40.0",
        },
    )
    print(f"Saved {path}", flush=True)


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise ValueError(f"Formal depth training requires exactly 4 GPUs, got {world_size}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_path).resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "training_args.json").write_text(
            json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
        )
    dist.barrier()

    dataset = CachedDepthDataset(args.cache_dir)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
        drop_last=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    module = DepthLoRAModule(args.model_dir, args.lora_rank, args.lora_target_modules, device)
    trainable_names = sorted(module.trainable_param_names())
    bad_names = [
        name for name in trainable_names
        if not (name.startswith("pipe.dit.") and (".lora_A." in name or ".lora_B." in name))
    ]
    if bad_names:
        raise RuntimeError(f"Unexpected trainable parameters: {bad_names[:20]}")
    context = module.prepare_context("output: depth", device)
    module.pipe.dit.to(device).train()
    trainable_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
    if rank == 0:
        print(
            f"dataset={len(dataset)} world_size={world_size} global_batch={world_size} "
            f"trainable={trainable_count:,} tensors={len(trainable_names)}",
            flush=True,
        )

    ddp = DistributedDataParallel(
        module,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    optimizer = torch.optim.AdamW(
        module.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    loss_path = output_dir / "loss.csv"
    if rank == 0:
        loss_path.write_text("step,loss,seconds,max_memory_gib\n", encoding="utf-8")
    start = time.time()
    global_step = 0
    epoch = 0
    while global_step < args.max_train_steps:
        sampler.set_epoch(epoch)
        for source_latent, target_latent in dataloader:
            source_latent = source_latent.to(device=device, dtype=torch.bfloat16, non_blocking=True)
            target_latent = target_latent.to(device=device, dtype=torch.bfloat16, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = ddp(
                source_latent,
                target_latent,
                context,
                args.gradient_checkpointing,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"rank={rank} non-finite loss at step {global_step + 1}")
            loss.backward()
            optimizer.step()
            global_step += 1

            reduced_loss = loss.detach().clone()
            dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
            reduced_loss /= world_size
            if rank == 0:
                elapsed = time.time() - start
                memory = torch.cuda.max_memory_allocated(device) / 1024**3
                value = float(reduced_loss)
                with loss_path.open("a", encoding="utf-8") as stream:
                    stream.write(f"{global_step},{value:.9g},{elapsed:.3f},{memory:.3f}\n")
                if global_step == 1 or global_step % 10 == 0:
                    print(
                        f"step={global_step}/{args.max_train_steps} loss={value:.6f} "
                        f"sec_per_step={elapsed/global_step:.3f} max_mem_gib={memory:.3f}",
                        flush=True,
                    )

            if global_step % args.save_steps == 0 or global_step == args.max_train_steps:
                dist.barrier()
                if rank == 0:
                    save_lora(module, output_dir, global_step, args)
                    state_tmp = output_dir / "trainer_state_latest.tmp"
                    torch.save(
                        {
                            "global_step": global_step,
                            "epoch": epoch,
                            "optimizer": optimizer.state_dict(),
                        },
                        state_tmp,
                    )
                    os.replace(state_tmp, output_dir / "trainer_state_latest.pt")
                dist.barrier()

            if global_step >= args.max_train_steps:
                break
        epoch += 1

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
