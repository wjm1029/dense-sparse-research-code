"""One-clip GenCeption-style depth overfit for Wan2.1 1.3B LoRA."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from tqdm import tqdm


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dataset import DepthVideoDataset  # noqa: E402
from diffsynth.diffusion.training_module import DiffusionTrainingModule  # noqa: E402
from diffsynth.pipelines.wan_video import (  # noqa: E402
    ModelConfig,
    WanVideoPipeline,
    WanVideoUnit_PromptEmbedder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--overfit_clip", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_train_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-2)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument(
        "--lora_target_modules", default="q,k,v,o,ffn.0,ffn.2"
    )
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vae_tiled", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


class DepthOverfitModule(DiffusionTrainingModule):
    def __init__(self, model_dir: str, lora_rank: int, target_modules: str) -> None:
        super().__init__()
        model_dir = str(Path(model_dir).resolve())
        model_configs = [
            ModelConfig(path=f"{model_dir}/diffusion_pytorch_model.safetensors"),
            ModelConfig(path=f"{model_dir}/models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(path=f"{model_dir}/Wan2.1_VAE.pth"),
        ]
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cpu",
            model_configs=model_configs,
            tokenizer_config=ModelConfig(path=f"{model_dir}/google/umt5-xxl"),
        )
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models=None,
            lora_base_model="dit",
            lora_target_modules=target_modules,
            lora_rank=lora_rank,
            task="sft",
        )

    @torch.no_grad()
    def encode_prompt(self, prompt: str, device: torch.device) -> torch.Tensor:
        self.pipe.device = device
        self.pipe.text_encoder.to(device)
        context = WanVideoUnit_PromptEmbedder().encode_prompt(self.pipe, prompt)
        context = context.to(device="cpu", dtype=self.pipe.torch_dtype)
        self.pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()
        return context

    @torch.no_grad()
    def encode_video(
        self, video_tchw: torch.Tensor, device: torch.device, tiled: bool
    ) -> torch.Tensor:
        self.pipe.device = device
        self.pipe.vae.to(device)
        video_cthw = (
            video_tchw.permute(1, 0, 2, 3)
            .contiguous()
            .to(dtype=self.pipe.torch_dtype)
        )
        latent = self.pipe.vae.encode(
            video_cthw.unsqueeze(0),
            device=device,
            tiled=tiled,
            tile_size=(30, 52),
            tile_stride=(15, 26),
        )
        return latent.to(device="cpu", dtype=self.pipe.torch_dtype)

    @torch.no_grad()
    def encode_pair(
        self, sample: dict, device: torch.device, tiled: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.encode_prompt(sample["prompt"], device)
        source_latent = self.encode_video(sample["source_video"], device, tiled)
        target_latent = self.encode_video(sample["target_video"], device, tiled)
        self.pipe.vae.to("cpu")
        torch.cuda.empty_cache()
        return source_latent, target_latent, context

    def forward_cached(
        self,
        source_latent: torch.Tensor,
        target_latent: torch.Tensor,
        context: torch.Tensor,
        *,
        gradient_checkpointing: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = source_latent.shape[0]
        timestep = torch.zeros(
            batch,
            device=source_latent.device,
            dtype=self.pipe.torch_dtype,
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
        prediction = -velocity
        loss = F.mse_loss(prediction.float(), target_latent.float())
        return loss, prediction


def save_lora(
    model: DepthOverfitModule,
    output_dir: Path,
    filename: str,
    metadata: dict[str, str],
) -> Path:
    trainable = model.trainable_param_names()
    state = {
        name.removeprefix("pipe.dit."): tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
        if name in trainable
    }
    if not state:
        raise RuntimeError("No trainable LoRA tensors found while saving")
    path = output_dir / filename
    save_file(state, str(path), metadata=metadata)
    return path


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    dataset = DepthVideoDataset(
        args.dataset_root,
        overfit_clip=args.overfit_clip,
        min_rgb_mean=40.0,
    )
    sample = dataset[0]
    print(
        json.dumps(
            {
                "clip_id": sample["clip_id"],
                "rgb_mean": sample["rgb_mean"],
                "p2": sample["depth_p2"],
                "p98": sample["depth_p98"],
                "source_shape": list(sample["source_video"].shape),
                "target_shape": list(sample["target_video"].shape),
                "source_range": [
                    float(sample["source_video"].min()),
                    float(sample["source_video"].max()),
                ],
                "target_range": [
                    float(sample["target_video"].min()),
                    float(sample["target_video"].max()),
                ],
            },
            indent=2,
        )
    )

    model = DepthOverfitModule(
        args.model_dir,
        lora_rank=args.lora_rank,
        target_modules=args.lora_target_modules,
    )
    trainable_names = sorted(model.trainable_param_names())
    bad_names = [
        name
        for name in trainable_names
        if not (
            name.startswith("pipe.dit.")
            and (".lora_A." in name or ".lora_B." in name)
        )
    ]
    if bad_names:
        raise RuntimeError(f"Unexpected trainable parameters: {bad_names[:20]}")
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"Trainable LoRA parameters: {trainable_count:,} in {len(trainable_names)} tensors")

    source_latent, target_latent, context = model.encode_pair(
        sample, device=device, tiled=args.vae_tiled
    )
    del sample
    print(
        f"Cached latents source={tuple(source_latent.shape)} "
        f"target={tuple(target_latent.shape)} context={tuple(context.shape)}"
    )
    if source_latent.shape != target_latent.shape:
        raise RuntimeError("Source and target VAE latents do not have identical shapes")

    model.pipe.dit.to(device)
    model.pipe.dit.train()
    source_latent = source_latent.to(device)
    target_latent = target_latent.to(device)
    context = context.to(device)
    optimizer = torch.optim.AdamW(
        model.trainable_modules(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    max_steps = 1 if args.dry_run else args.max_train_steps
    loss_path = output_dir / "loss.csv"
    loss_path.write_text("step,loss,seconds,max_memory_gib\n", encoding="utf-8")
    metadata = {
        "task": "depth",
        "method": "genception_clean_source_t0_negated_velocity_latent_mse",
        "base_model": "Wan2.1-T2V-1.3B",
        "clip_id": dataset.samples[0]["id"],
        "lora_rank": str(args.lora_rank),
        "lora_targets": args.lora_target_modules,
    }

    start = time.time()
    progress = tqdm(range(1, max_steps + 1), desc="depth overfit")
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.forward_cached(
            source_latent,
            target_latent,
            context,
            gradient_checkpointing=args.gradient_checkpointing,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.log_steps == 0 or step == max_steps:
            elapsed = time.time() - start
            max_memory = (
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else 0.0
            )
            value = float(loss.detach())
            progress.set_postfix(loss=f"{value:.6f}", mem=f"{max_memory:.1f}GiB")
            with loss_path.open("a", encoding="utf-8") as stream:
                stream.write(f"{step},{value:.9g},{elapsed:.3f},{max_memory:.3f}\n")

        if step % args.save_steps == 0 or step == max_steps:
            path = save_lora(
                model,
                output_dir,
                f"step-{step}.safetensors",
                metadata={**metadata, "step": str(step)},
            )
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
