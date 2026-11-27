"""Wan2.2 long-477 AR self-forcing continuation with truncated BPTT."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import train_wan22_dense4_ddp as dense4
from wan22_ar_self_forcing import detach_kv_cache, training_streaming_forward
from wan22_ar_teacher_forcing import ARTeacherForcingConfig, combine_rgb_and_history, install_ar_history_adapter
from wan22_cache_integrity import sha256_file
from wan22_causal_attention import install_causal_attention
from wan22_dense4_checkpoint import (
    OPTIONAL_PROVENANCE_KEYS,
    REQUIRED_PROVENANCE_KEYS,
    checkpoint_metadata,
    load_dense4_checkpoint,
    save_dense4_checkpoint,
)
from wan22_masked_loss import masked_latent_mse
from wan22_task_router import ID_TO_TASK, TASK_TO_ID


LATENT_SHAPE = (48, 120, 30, 52)
MASK_SHAPE = (1, 120, 30, 52)
CHUNK = 3
NUM_CHUNKS = LATENT_SHAPE[1] // CHUNK
AR_CONFIG = ARTeacherForcingConfig(120, CHUNK, 48)
MAIN_SCHEMA = "wan22_long477_dense_latent_v1"
MAIN_NORMAL_SCHEMA = "wan22_long477_normal_flip_yz_latent_v1"
LINGBOT_SCHEMA = "wan22_depth477_latent_v1"
LINGBOT_NORMAL_SCHEMA = "wan22_lingbot_normal477_latent_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_cache_dir", required=True)
    parser.add_argument("--main_normal_cache_dir", required=True)
    parser.add_argument("--lingbot_depth_cache_dir", required=True)
    parser.add_argument("--lingbot_normal_cache_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--prompt_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--code_patch_path", required=True)
    parser.add_argument("--init_checkpoint", required=True)
    parser.add_argument("--init_code_patch_path", required=True)
    parser.add_argument("--max_train_steps", type=int, default=2000)
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--router_learning_rate", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--router_identity_weight", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--single_gpu_smoke", action="store_true")
    return parser.parse_args()


def verified_manifest(root: Path, schema: str) -> tuple[dict, dict]:
    manifest_path, complete_path = root / "manifest.json", root / "CACHE_COMPLETE"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != schema or complete.get("schema") != schema:
        raise ValueError(f"Unexpected cache schema: {root}")
    if complete.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError(f"Cache manifest hash mismatch: {root}")
    if complete.get("cache_contract_hash") != manifest.get("cache_contract_hash"):
        raise ValueError(f"Cache contract mismatch: {root}")
    return manifest, complete


class Long477Dataset(Dataset):
    def __init__(self, main_root: Path, main_normal_root: Path, lingbot_root: Path, lingbot_normal_root: Path, prompt_texts: list[str]) -> None:
        self.main_root, self.lingbot_root = main_root, lingbot_root
        self.main_normal_root = main_normal_root
        self.lingbot_normal_root = lingbot_normal_root
        self.main_manifest, _ = verified_manifest(main_root, MAIN_SCHEMA)
        self.main_normal_manifest, _ = verified_manifest(main_normal_root, MAIN_NORMAL_SCHEMA)
        self.lingbot_manifest, _ = verified_manifest(lingbot_root, LINGBOT_SCHEMA)
        self.lingbot_normal_manifest, _ = verified_manifest(lingbot_normal_root, LINGBOT_NORMAL_SCHEMA)
        self.main_contract = self.main_manifest["cache_contract_hash"]
        self.main_normal_contract = self.main_normal_manifest["cache_contract_hash"]
        self.lingbot_contract = self.lingbot_manifest["cache_contract_hash"]
        self.lingbot_normal_contract = self.lingbot_normal_manifest["cache_contract_hash"]
        self.prompt_to_id = {text: index for index, text in enumerate(prompt_texts)}
        self.main_ids = sorted(
            path.relative_to(main_root / "source").with_suffix("").as_posix()
            for path in (main_root / "source").glob("*/*.pt")
        )
        self.lingbot_paths = sorted(lingbot_root.glob("*/*.pt"))
        if len(self.main_ids) != int(self.main_manifest["num_clips"]):
            raise ValueError("Main cache count mismatch")
        main_normal_ids = sorted(
            path.relative_to(main_normal_root / "normal").with_suffix("").as_posix()
            for path in (main_normal_root / "normal").glob("*/*.pt")
        )
        if main_normal_ids != self.main_ids:
            raise ValueError("Main source/flip-YZ normal exact path sets differ")
        for key in ("source_index_sha256", "ordered_clip_ids_hash", "num_clips"):
            if self.main_normal_manifest.get(key) != self.main_manifest.get(key):
                raise ValueError(f"Main flip-YZ normal is not bound to source cache: {key}")
        if self.main_normal_manifest.get("normal_transform_before_vae") != ["x", "-y", "-z"]:
            raise ValueError("Main normal cache does not declare the required flip-YZ transform")
        if len(self.lingbot_paths) != int(self.lingbot_manifest["num_samples"]):
            raise ValueError("LingBot cache count mismatch")
        normal_relative = [path.relative_to(lingbot_normal_root).as_posix() for path in sorted(lingbot_normal_root.glob("*/*.pt"))]
        depth_relative = [path.relative_to(lingbot_root).as_posix() for path in self.lingbot_paths]
        if normal_relative != depth_relative:
            raise ValueError("LingBot depth/normal exact path sets differ")
        if (
            self.lingbot_normal_manifest.get("source_cache_contract_hash") != self.lingbot_contract
            or self.lingbot_normal_manifest.get("ordered_clip_ids_hash") != self.lingbot_manifest.get("ordered_clip_ids_hash")
        ):
            raise ValueError("LingBot normal is not bound to this depth cache")
        self.main_samples = [(clip_id, task) for clip_id in self.main_ids for task in ("depth", "normal", "matte")]

    def __len__(self) -> int:
        return len(self.main_samples) + 2 * len(self.lingbot_paths)

    @staticmethod
    def _latent(payload: dict, key: str) -> torch.Tensor:
        value = payload[key]
        if tuple(value.shape) != LATENT_SHAPE or value.dtype != torch.bfloat16 or not torch.isfinite(value).all():
            raise ValueError(f"Invalid cached latent {key}")
        return value

    def __getitem__(self, index: int):
        if index < len(self.main_samples):
            clip_id, task = self.main_samples[index]
            source_payload = torch.load(self.main_root / "source" / f"{clip_id}.pt", map_location="cpu", weights_only=False)
            target_root = self.main_normal_root if task == "normal" else self.main_root
            target_payload = torch.load(target_root / task / f"{clip_id}.pt", map_location="cpu", weights_only=False)
            expected_target_contract = self.main_normal_contract if task == "normal" else self.main_contract
            if source_payload.get("cache_contract_hash") != self.main_contract or target_payload.get("cache_contract_hash") != expected_target_contract:
                raise ValueError(f"Main cache payload contract mismatch: {clip_id}")
            if task == "normal" and target_payload.get("normal_transform") != ["x", "-y", "-z"]:
                raise ValueError(f"Main normal payload lacks flip-YZ contract: {clip_id}")
            source = self._latent(source_payload, "latent")
            target = self._latent(target_payload, "latent")
            task_id = int(target_payload["task_id"])
            prompt = target_payload["prompt"]
            mask = torch.ones(MASK_SHAPE, dtype=torch.float32)
            sample_id = f"main:{clip_id}:{task}"
        else:
            auxiliary_index = index - len(self.main_samples)
            normal_task = auxiliary_index >= len(self.lingbot_paths)
            path = self.lingbot_paths[auxiliary_index % len(self.lingbot_paths)]
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if payload.get("cache_contract_hash") != self.lingbot_contract:
                raise ValueError(f"LingBot payload contract mismatch: {path}")
            source = self._latent(payload, "source_latent")
            if normal_task:
                normal_payload = torch.load(self.lingbot_normal_root / path.relative_to(self.lingbot_root), map_location="cpu", weights_only=False)
                if normal_payload.get("cache_contract_hash") != self.lingbot_normal_contract or normal_payload.get("clip_id") != payload.get("clip_id"):
                    raise ValueError(f"LingBot normal payload mismatch: {path}")
                target = self._latent(normal_payload, "target_latent")
                task_id, prompt = TASK_TO_ID["normal"], normal_payload["prompt"]
                mask = normal_payload["latent_loss_mask"].float()
            else:
                target = self._latent(payload, "target_latent")
                task_id, prompt = TASK_TO_ID["depth"], payload["prompt"]
                mask = payload["latent_loss_mask"].float()
            if tuple(mask.shape) != MASK_SHAPE or not bool(mask.any()):
                raise ValueError(f"Invalid LingBot loss mask: {path}")
            sample_id = f"lingbot:{payload['clip_id']}:{'normal' if normal_task else 'depth'}"
        if prompt not in self.prompt_to_id:
            raise ValueError(f"Prompt absent from checkpoint: {prompt}")
        return source, target, task_id, self.prompt_to_id[prompt], mask, sample_id


class SelfForcingModule(dense4.Dense4LoRAModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.causal_controller = install_causal_attention(
            self.pipe.dit, latent_shape=LATENT_SHAPE, frames_per_chunk=CHUNK,
            local_attn_frames=-1, sink_frames=0,
        )
        self.ar_controller = install_ar_history_adapter(self.pipe.dit, self.causal_controller, AR_CONFIG)

    def forward(self, rgb, history, target, mask, task_ids, prompt_ids, frame_offset):
        if self.context_table is None:
            raise RuntimeError("Prompt table is not ready")
        gates = self.task_router(task_ids)
        timestep = torch.zeros(task_ids.shape[0], device=task_ids.device, dtype=torch.bfloat16)
        ar_input = combine_rgb_and_history(rgb, history, AR_CONFIG)
        prediction = -training_streaming_forward(
            self.causal_controller, self.pipe.dit, ar_input, timestep,
            self.context_table[prompt_ids], frame_offset=int(frame_offset),
            caches=self._active_caches,
        )
        # Sparse LingBot masks can contain a completely invalid temporal chunk.
        # Keep rolling the prediction/KV state forward, but contribute no data
        # gradient for a chunk with zero supervised support.
        data_loss = (
            masked_latent_mse(prediction, target, mask)
            if bool(mask.any())
            else prediction.float().sum() * 0.0
        )
        identity_loss = self.task_router.identity_regularization(gates)
        return prediction, data_loss, identity_loss, gates


def provenance_from_caches(args, init_meta: dict[str, str], main_manifest: dict, main_normal_manifest: dict, lingbot_manifest: dict, lingbot_normal_manifest: dict) -> dict[str, str]:
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=THIS_DIR.parents[3], check=True, capture_output=True, text=True).stdout.strip()
    result = {key: init_meta[key] for key in REQUIRED_PROVENANCE_KEYS}
    result.update(
        {
            "depth_cache_manifest_sha256": sha256_file(Path(args.lingbot_depth_cache_dir) / "manifest.json"),
            "depth_cache_contract_hash": lingbot_manifest["cache_contract_hash"],
            "target_cache_manifest_sha256": sha256_file(Path(args.main_cache_dir) / "manifest.json"),
            "target_cache_contract_hash": main_manifest["cache_contract_hash"],
            "diffsynth_git_head": git_head,
            "runtime_source_patch_sha256": sha256_file(args.code_patch_path),
            "auxiliary_cache_identities": json.dumps(
                {
                    "long477_main": {"path": str(Path(args.main_cache_dir).resolve()), "schema": MAIN_SCHEMA, "cache_contract_hash": main_manifest["cache_contract_hash"]},
                    "long477_main_normal_flip_yz": {"path": str(Path(args.main_normal_cache_dir).resolve()), "schema": MAIN_NORMAL_SCHEMA, "cache_contract_hash": main_normal_manifest["cache_contract_hash"], "normal_transform": ["x", "-y", "-z"]},
                    "long477_lingbot_depth": {"path": str(Path(args.lingbot_depth_cache_dir).resolve()), "schema": LINGBOT_SCHEMA, "cache_contract_hash": lingbot_manifest["cache_contract_hash"]},
                    "long477_lingbot_normal": {"path": str(Path(args.lingbot_normal_cache_dir).resolve()), "schema": LINGBOT_NORMAL_SCHEMA, "cache_contract_hash": lingbot_normal_manifest["cache_contract_hash"]},
                }, sort_keys=True, separators=(",", ":"),
            ),
        }
    )
    return result


def save_checkpoint(module, output: Path, step: int, args, provenance: dict[str, str]) -> None:
    init = Path(args.init_checkpoint).resolve()
    save_dense4_checkpoint(
        module, output / f"step-{step}.safetensors", step=step, world_size=dist.get_world_size(),
        lora_rank=32, lora_targets="q,k,v,o,ffn.0,ffn.2", router_hidden_dim=512,
        router_mode="learned", provenance_metadata=provenance,
        extra_metadata={
            "initialization_checkpoint": str(init), "initialization_checkpoint_sha256": sha256_file(init),
            "causal_attention": "block_prefix_v1", "causal_num_frames": "120",
            "causal_tokens_per_frame": "390", "causal_frames_per_chunk": "3",
            "causal_local_attn_frames": "-1", "causal_sink_frames": "0",
            "causal_streaming_kv": "per_layer_v1", "ar_teacher_forcing": "none",
            "ar_inference_history": "previous_predicted_chunk_v1",
            "ar_history_adapter": "continued_from_teacher_forcing_checkpoint",
            "ar_num_frames": "120", "ar_frames_per_chunk": "3", "ar_num_chunks": "40",
            "ar_latent_channels": "48", "self_forcing": "previous_predicted_chunk_stopgrad_v1",
            "bptt": "chunkwise_truncated_stopgrad_kv_and_history_v1",
            "normal_coordinate_contract": "camera_x_right_y_up_z_toward_camera_v1",
        },
    )


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4 and not (world == 1 and args.single_gpu_smoke):
        raise ValueError("Formal self-forcing training requires four GPUs")
    random.seed(args.seed + rank); np.random.seed(args.seed + rank); torch.manual_seed(args.seed + rank)

    init_meta = checkpoint_metadata(args.init_checkpoint)
    if init_meta.get("ar_teacher_forcing") != "previous_gt_chunk_v1" or init_meta.get("ar_frames_per_chunk") != "3":
        raise ValueError("Initialization must be the audited chunk-3 AR teacher-forcing checkpoint")
    if sha256_file(args.init_code_patch_path) != init_meta["runtime_source_patch_sha256"]:
        raise ValueError("Initialization source patch hash mismatch")
    prompt_texts = json.loads(init_meta["output_prompt_texts"])
    dataset = Long477Dataset(
        Path(args.main_cache_dir).resolve(), Path(args.main_normal_cache_dir).resolve(),
        Path(args.lingbot_depth_cache_dir).resolve(),
        Path(args.lingbot_normal_cache_dir).resolve(), prompt_texts,
    )
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0)

    module = SelfForcingModule(args.model_dir, 32, "q,k,v,o,ffn.0,ffn.2", 512, "learned", args.router_identity_weight)
    module.task_router.to(device=device, dtype=torch.float32)
    module.prepare_condition_tables(prompt_texts, device)
    init_provenance = {key: init_meta[key] for key in REQUIRED_PROVENANCE_KEYS}
    for key in OPTIONAL_PROVENANCE_KEYS:
        if key in init_meta: init_provenance[key] = init_meta[key]
    load_dense4_checkpoint(
        module, args.init_checkpoint, expected_provenance=init_provenance, lora_rank=32,
        lora_targets="q,k,v,o,ffn.0,ffn.2", router_hidden_dim=512, expected_router_mode="learned",
    )
    module.pipe.dit.to(device).train()
    ddp = DistributedDataParallel(module, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
    groups = dense4.optimizer_param_groups(module, argparse.Namespace(
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        router_learning_rate=args.router_learning_rate, router_weight_decay=args.weight_decay,
    ))
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / max(1, args.warmup_steps)))
    output = Path(args.output_path).resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()) and not args.resume_from:
            raise FileExistsError(f"Refusing non-empty output: {output}")
    dist.barrier()
    main_manifest, _ = verified_manifest(Path(args.main_cache_dir), MAIN_SCHEMA)
    main_normal_manifest, _ = verified_manifest(Path(args.main_normal_cache_dir), MAIN_NORMAL_SCHEMA)
    lingbot_manifest, _ = verified_manifest(Path(args.lingbot_depth_cache_dir), LINGBOT_SCHEMA)
    lingbot_normal_manifest, _ = verified_manifest(Path(args.lingbot_normal_cache_dir), LINGBOT_NORMAL_SCHEMA)
    provenance = provenance_from_caches(args, init_meta, main_manifest, main_normal_manifest, lingbot_manifest, lingbot_normal_manifest)
    step = epoch = 0
    if args.resume_from:
        state = torch.load(Path(args.resume_from) / "trainer_state_latest.pt", map_location="cpu", weights_only=False)
        step, epoch = int(state["step"]), int(state["epoch"])
        load_dense4_checkpoint(module, Path(args.resume_from) / f"step-{step}.safetensors", expected_provenance=provenance, lora_rank=32, lora_targets="q,k,v,o,ffn.0,ffn.2", router_hidden_dim=512, expected_router_mode="learned")
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
    if rank == 0 and not args.resume_from:
        (output / "training_args.json").write_text(json.dumps(vars(args) | {"dataset_samples": len(dataset), "latent_shape": LATENT_SHAPE, "chunks": NUM_CHUNKS, "method": "self_forcing_stopgrad_tbptt"}, indent=2), encoding="utf-8")
        with (output / "loss.csv").open("w", newline="") as f: csv.writer(f).writerow(["step","loss","data_loss","identity_loss","task_id","task","grad_norm","seconds","max_memory_gib"])

    started = time.time()
    while step < args.max_train_steps:
        sampler.set_epoch(epoch)
        for source, target, task_ids, prompt_ids, mask, sample_ids in loader:
            source = source.to(device, dtype=torch.bfloat16, non_blocking=True)
            target = target.to(device, dtype=torch.bfloat16, non_blocking=True)
            mask = mask.to(device, dtype=torch.float32, non_blocking=True)
            task_ids = task_ids.to(device, dtype=torch.long); prompt_ids = prompt_ids.to(device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            module._active_caches = module.causal_controller.new_kv_cache()
            history = torch.zeros_like(source[:, :, :CHUNK])
            data_sum = identity_sum = 0.0
            valid_chunk_count = sum(
                int(bool(mask[:, :, offset:offset+CHUNK].any()))
                for offset in range(0, LATENT_SHAPE[1], CHUNK)
            )
            if valid_chunk_count == 0:
                raise ValueError(f"Sample has no valid chunks: {sample_ids}")
            for chunk_index, offset in enumerate(range(0, LATENT_SHAPE[1], CHUNK)):
                sync = nullcontext() if chunk_index == NUM_CHUNKS - 1 else ddp.no_sync()
                with sync:
                    prediction, data_loss, identity_loss, _ = module.task_router.call_with_error_cleanup(
                        lambda o=offset: ddp(source[:, :, o:o+CHUNK], history, target[:, :, o:o+CHUNK], mask[:, :, o:o+CHUNK], task_ids, prompt_ids, o)
                    )
                    loss = (
                        data_loss / valid_chunk_count
                        + args.router_identity_weight * identity_loss / NUM_CHUNKS
                    )
                    if not torch.isfinite(loss): raise FloatingPointError(f"non-finite loss chunk={chunk_index}")
                    loss.backward()
                data_sum += float(data_loss.detach()); identity_sum += float(identity_loss.detach())
                history = prediction.detach()
                detach_kv_cache(module._active_caches)
                module.task_router.clear()
            finite = torch.tensor(int(dense4.gradients_are_finite(module)), device=device)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not int(finite): raise FloatingPointError("non-finite gradient")
            grad_norm = torch.nn.utils.clip_grad_norm_([p for p in module.parameters() if p.requires_grad], args.max_grad_norm, error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); step += 1
            record = torch.tensor([data_sum/NUM_CHUNKS, identity_sum/NUM_CHUNKS, float(task_ids.item()), float(grad_norm)], device=device)
            gathered = [torch.empty_like(record) for _ in range(world)]; dist.all_gather(gathered, record)
            if rank == 0:
                elapsed = time.time() - started
                with (output / "loss.csv").open("a", newline="") as f:
                    writer=csv.writer(f)
                    for row in gathered:
                        tid=int(row[2]); writer.writerow([step,float(row[0])+args.router_identity_weight*float(row[1]),float(row[0]),float(row[1]),tid,ID_TO_TASK[tid],float(row[3]),elapsed,torch.cuda.max_memory_allocated(device)/1024**3])
                print(f"step={step}/{args.max_train_steps} loss={sum(float(x[0]) for x in gathered)/world:.6f} sec_per_step={elapsed/step:.2f} max_mem_gib={torch.cuda.max_memory_allocated(device)/1024**3:.2f}", flush=True)
            if step % args.save_steps == 0 or step == args.max_train_steps:
                dist.barrier()
                if rank == 0:
                    save_checkpoint(module, output, step, args, provenance)
                    torch.save({"step":step,"epoch":epoch,"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict()}, output / "trainer_state_latest.pt")
                dist.barrier()
            if step >= args.max_train_steps: break
        epoch += 1
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
