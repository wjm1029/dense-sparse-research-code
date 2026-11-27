# Wan2.2-TI2V-5B + LoRA 实现 GenCeption 单步统一稠密预测：调研与改造方案

更新时间：2026-08-30  
调研机器：`acdawn`（实际 hostname：`aibox-r915b2f930e6-696888777b-9l2fc`）  
状态：已完成评审修复、v2 cache/checkpoint 契约、CPU 契约测试、单卡与四卡真实 forward/backward、四卡 resume 和 checkpoint roundtrip；全量 v2 cache 尚未重建。

> **模型边界（避免误解）：本方案训练的是 Wan2.2-TI2V-5B，不使用任何 Wan2.1 模型权重、Wan2.1 VAE 或 Wan2.1 latent cache。** 文中出现 Wan2.1 仅有两个原因：论文原实验基座是 Wan2.1，以及服务器上已有的旧 trainer 可用于指出哪些硬编码必须丢弃；它不是本方案的训练基座。

## 1. 结论先行

建议基于服务器上的：

```text
/pfs/mayuema/dense/wan/Wan2.2-TI2V-5B
```

做 GenCeption-style 单步训练，而不是使用 Wan2.2 A14B MoE。现有两套 Wan2.2 latent cache 已经明确绑定 TI2V-5B：

```text
/pfs/mayuema/dense/wan/cache/wan22_depth_latents_bf16_bright40
/pfs/mayuema/dense/wan/cache/wan22_dense4_targets_bf16_bright40
```

TI2V-5B 是单个 5B dense DiT，输入/输出 latent 都是 48 通道；其 Wan2.2 VAE 对 81×480×832 视频输出 `[48,21,30,52]`，与缓存完全一致。A14B 则是高噪声/低噪声双专家，使用 Wan2.1 VAE 的 16 通道 latent；它既不匹配当前缓存，也会引入“`t=0` 到底只训练低噪声专家还是同时训练两个专家”的额外设计问题。

训练核心应保持为：

```python
z_source = cached_clean_rgb_latent
z_target = cached_target_modality_latent
timestep = torch.zeros(batch_size, device=device, dtype=torch.bfloat16)

velocity = pipe.model_fn(
    dit=pipe.dit,
    latents=z_source,
    timestep=timestep,
    context=prompt_context,
    use_gradient_checkpointing=True,
    use_gradient_checkpointing_offload=False,
    cfg_merge=False,
)

prediction = -velocity
loss = torch.nn.functional.mse_loss(prediction.float(), z_target.float())
```

必须避免通用 `FlowMatchSFTLoss`：它会随机采样 timestep、加高斯噪声，并监督 `noise - sample`，与论文的 clean source、固定 `t=0`、单次 forward 不同。

四任务主方案采用 **Wan2.2 共享 LoRA + UniControl-style task embedding HyperRouter**：四个模态共用 LoRA A/B 矩阵，冻结 UMT5 将独立的 task instruction 编码成 task feature，共享 HyperNet 再为每个 LoRA site 生成 rank-channel gate，实现 `B·diag(g_task,site)·A` 路由。纯共享 LoRA（所有 gate 恒为 1）保留为必须运行的消融基线，而不是最终主方案。

## 2. 调研对象与版本

### 2.1 论文

论文：*Video Generation Models are General-Purpose Vision Learners*（GenCeption），本地文件：

```text
/Users/wjm/Documents/dense predictor/videobanana.pdf
```

使用 `pdftotext -layout` 抽取全文，共 19 页、1,125 行。关键依据：

- Sec. 3.2：将 input video 的 clean latent 直接输入 DiT；固定 `t=0`；只做一次 forward；Wan Rectified Flow velocity 输出在进入 loss/decoder 前取负。
- Sec. 3.3：所有 dense task 统一编码到 `[0,1]` RGB 空间；depth/mask 的单通道复制成三通道；任务由文本指令控制。
- Sec. 3.4：离线缓存 input/target video latents 和 prompt text embeddings。
- Sec. 3.5：dense task 统一使用 latent-space L2 loss，不增加 task-specific loss。
- Sec. 4.1：论文使用 Wan2.1、81 帧、480×832、全量微调、Adam、`5e-5`、15k steps、250-step warmup；论文只说明 gradient clipping/dropping 很重要，没有公开可唯一复现的阈值。

### 2.2 Wan2.2 官方仓库

```text
repo:   /pfs/mayuema/dense/wan/Wan2.2
commit: 42bf4cfaa384bc21833865abc2f9e6c0e67233dc
remote: https://github.com/Wan-Video/Wan2.2.git
```

该仓库本身主要是推理实现。README 指向 DiffSynth-Studio 作为 Wan2.2 LoRA/full training 支持。TI2V-5B 官方配置为：

```text
vae_stride: (4,16,16)
patch_size: (1,2,2)
dim: 3072
ffn_dim: 14336
num_heads: 24
num_layers: 30
in_dim/out_dim: 48/48
```

### 2.3 DiffSynth-Studio 官方 LoRA 训练实现

```text
repo:   /pfs/mayuema/dense/wan/DiffSynth-Studio
commit: 102fe9980b9375ecb6436d360297a00327472535
remote: https://github.com/modelscope/DiffSynth-Studio.git
status: examples/wanvideo/model_training/dense_lora/ 为现有未跟踪项目代码
```

官方 Wan2.2 TI2V-5B LoRA launcher：

```text
examples/wanvideo/model_training/lora/Wan2.2-TI2V-5B.sh
```

其推荐 LoRA 配置：

```text
lora_base_model: dit
lora_target_modules: q,k,v,o,ffn.0,ffn.2
lora_rank: 32
learning_rate: 1e-4
```

这会命中 30 个 block 中 self-attention 4 个投影、cross-attention 4 个投影和 FFN 2 个投影，共 300 个线性层。rank 32 时 LoRA 参数量为 80,609,280；LoRA FP32 权重约 307.5 MiB，若粗略计入 FP32 参数、梯度和 AdamW 两个 moment，约 1.20 GiB，不含激活和冻结的 BF16 DiT。

### 2.4 UniControl task embedding / Task-aware HyperNet

新增依据：[*UniControl: A Unified Diffusion Model for Controllable Visual Generation in the Wild*](https://arxiv.org/abs/2305.11147)，以及作者官方实现 [salesforce/UniControl](https://github.com/salesforce/UniControl)：

```text
paper: arXiv:2305.11147，32 页
official code commit inspected: e84d8d2085f58c88c913631fe0320f82a701a174
key file: cldm/cldm_unicontrol_v11.py
```

论文和代码给出的结构事实：

1. 普通内容 prompt `c_text` 与 task instruction `c_task` 是两个独立条件，不能把 task routing 等同于 cross-attention prompt。
2. task instruction 使用冻结 CLIPText 编码；官方 `get_input` 将 task instruction 追加为额外文本，编码后取 `c_all[BS:, :1, :]`，即 task instruction 的第一个 contextual token，维度为 768。
3. 共享 `task_id_hypernet` 为 `Linear(768,time_embed_dim) -> SiLU -> Linear -> SiLU`。
4. 每个 zero-conv 层有独立 layernet，把共享 task embedding 投影到该层输入通道数。
5. 调制采用 StyleGAN-style input-channel weight modulation：每个输入通道的卷积核乘对应 task scalar。
6. Appendix A.2 说明训练后期会冻结 Task-aware HyperNet 以稳定 dynamics，但没有给出通用冻结 step。
7. Table 4 中完整的 MoE Adapter + TaskHyperNet 平均 FID 为 24.0，优于无二者的 28.7 和只有 MoE Adapter 的 25.7，支持 task-aware routing 的有效性。

这里不照搬 UniControl 的 task-specific shallow MoE Conv adapter：UniControl 面对的是 Canny/depth/pose 等不同类型的**输入控制图**，而本项目四个任务的输入始终是同一种 clean RGB video latent，差异在输出模态。需要迁移的是 Task-aware HyperNet 的结构原则，而不是输入端的多 ConvNet。

## 3. 为什么选择 TI2V-5B，而不是 A14B MoE

| 项目 | TI2V-5B | T2V/I2V A14B |
|---|---|---|
| DiT | 单个 5B dense model | 高噪声/低噪声两个约 14B expert |
| VAE | Wan2.2 VAE | Wan2.1 VAE |
| latent channel | 48 | 16 |
| 81×480×832 latent | `[48,21,30,52]` | `[16,21,60,104]` |
| 当前缓存兼容性 | 完全匹配 | 不匹配 |
| `t=0` 单步语义 | 单模型直接 forward | 只自然落到 low-noise expert；双 LoRA 训练语义需另行定义 |

因此当前任务不应把 Wan2.2 等同于 A14B MoE。缓存 manifest 中的 `model_dir`、VAE hash、latent shape 都已经将实验锁定到 TI2V-5B。

## 4. 缓存现状与已完成的只读验证

### 4.1 Depth/source cache

```text
schema: wan22_depth_latent_v1
samples: 7,350
prompt: output: depth
latent shape: [48,21,30,52]
dtype: bfloat16
```

每个 payload 包含：

```text
schema, clip_id, prompt, rgb_mean, depth_p2, depth_p98,
source_latent, target_latent
```

其中 `source_latent` 是 RGB clean latent，`target_latent` 是 depth RGB 表示的 latent。

### 4.2 Dense4 target cache

```text
schema: wan22_dense4_target_latent_v1
reader_schema: wan21_dense4_readers_v2
clips: 7,350
tasks: normal, matte, raymap
target payloads: 22,050
```

与 depth target 合并后，四任务训练集是：

```text
7,350 clips × 4 tasks = 29,400 task samples
```

`reader_schema` 名字仍带 `wan21`，只表示上游 normal/matte/raymap 的读入与表示规范沿用既有 v2 reader；latent 本身由官方 Wan2.2 VAE 重新编码，不是 Wan2.1 latent。

### 4.3 2026-08-30 只读测试结果

已执行：

1. manifest、ordered clip ID hash、每任务 relative-path hash 和实际文件数核对；
2. 从首部、中部、尾部及固定偏移抽 7 个 clip，反序列化四任务共 28 个 payload；
3. 检查所有抽样 tensor 为 BF16、shape `[48,21,30,52]`、全部 finite，clip/task/schema 一致；
4. 检查 DiT 3 个 safetensors shard 与 index 完整；
5. 根据权重 index 检查 300 个 LoRA target weight 全部存在，并核对 patch/head/首尾 block 代表性 shape；
6. 对 `dense_lora/*.py` 执行 `py_compile`。

结果全部通过：

```json
{
  "status": "PASS",
  "clips": 7350,
  "task_payloads": 29400,
  "sample_payloads_deserialized": 28,
  "latent_shape": [48, 21, 30, 52],
  "dit_shards": 3,
  "lora_linear_sites_expected": 300
}
```

新增 task routing 结构 CPU 单元检查也已通过：用 300 个 Wan2.2 LoRA site、rank 32 实例化现有 `TaskHyperRouter`，得到 gate shape `[4,300,32]`，zero-init gate 与全 1 tensor bitwise 相等，可训练 router 参数量为 7,293,312，冻结 task feature table 不带梯度。

本轮没有做真实 5B GPU forward/backward：调研时四张卡均显示 100% utilization，且每卡只剩约 52 GiB free。为避免干扰现有作业，只做了 CPU/文件级只读测试。

## 5. 为什么必须新建纯 Wan2.2 trainer，而不能沿用旧 Wan2.1 trainer

现有实现位于：

```text
examples/wanvideo/model_training/dense_lora/train_depth_ddp.py
examples/wanvideo/model_training/dense_lora/train_dense4_ddp.py
```

需要修改的硬编码至少包括：

1. `LATENT_SHAPE` 从 `(16,21,60,104)` 改为 `(48,21,30,52)`；
2. depth schema 从 `wan21_depth_latent_v1` 改为 `wan22_depth_latent_v1`；
3. dense target schema 从 `wan21_dense4_target_latent_v2` 改为 `wan22_dense4_target_latent_v1`；
4. cache contract 中 VAE 文件从 `Wan2.1_VAE.pth` 改为 `Wan2.2_VAE.pth`；
5. checkpoint metadata 的 base model 改为 `Wan2.2-TI2V-5B`；
6. Wan2.1 使用单个 `diffusion_pytorch_model.safetensors`，而 TI2V-5B 使用 3 个 shard；当前单文件加载路径不存在；
7. 对 trainable LoRA site 数量、checkpoint site hash 和重载逻辑重新建基线；
8. 训练/验证入口不得传 `input_image`，因为当前任务输入是完整 clean video latent，不是 TI2V 的首帧条件模式。

建议以 DiffSynth 官方 `Wan2.2-TI2V-5B.sh` 的模型加载和 LoRA 注入为起点，新建纯 Wan2.2 入口。旧 Wan2.1 文件只用于核对项目已有的 dataset/sampler/checkpoint 需求，不作为运行时依赖，也不应原地改成同时兼容两个模型：

```text
dense_lora/train_wan22_depth_ddp.py
dense_lora/train_wan22_dense4_ddp.py
dense_lora/validate_wan22_single_forward.py
dense_lora/Wan2.2-TI2V-5B-depth-smoke.sh
dense_lora/Wan2.2-TI2V-5B-dense4-shared-lora-smoke.sh
```

可以再抽取共享的 cache/model contract helper，减少两个版本重复，但不要把不同 VAE/schema/shape 隐藏在模糊的自动猜测中。

## 6. 模型加载的正确改法

TI2V-5B 的 DiT 必须按 shard 列表加载：

```python
from pathlib import Path
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

model_dir = Path(args.model_dir).resolve()
dit_paths = sorted(model_dir.glob("diffusion_pytorch_model-*-of-*.safetensors"))
if len(dit_paths) != 3:
    raise ValueError(f"Expected 3 TI2V-5B DiT shards, got {len(dit_paths)}")

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cpu",
    model_configs=[
        ModelConfig(path=[str(path) for path in dit_paths]),
        ModelConfig(path=str(model_dir / "models_t5_umt5-xxl-enc-bf16.pth")),
    ],
    tokenizer_config=ModelConfig(path=str(model_dir / "google/umt5-xxl")),
)
```

训练全程读取 cached latent，因此不需要加载 VAE；只在验证解码进程中加载 `Wan2.2_VAE.pth`。这样可以降低训练常驻显存/内存。启动后要断言：

```text
pipe.dit.in_dim == 48
pipe.dit.out_dim == 48
pipe.dit.dim == 3072
pipe.dit.num_layers == 30
pipe.dit.patch_size == (1,2,2)
pipe.dit.seperated_timestep is True
```

TI2V-5B 支持 token-wise separated timestep，主要用于 I2V 的 clean 首帧与 noisy 其余帧混合。当前 full-video clean latent 任务应传一个 batch scalar `timestep=zeros[B]`，让模型内部对整个序列广播为 0；不要构造首帧 mask，也不要设置 `fuse_vae_embedding_in_latents=True`。

## 7. LoRA、冻结和统一任务设计

### 7.1 LoRA 注入

沿用官方：

```python
self.switch_pipe_to_training_mode(
    pipe,
    trainable_models=None,
    lora_base_model="dit",
    lora_target_modules="q,k,v,o,ffn.0,ffn.2",
    lora_rank=32,
    task="sft",  # 仅复用训练模式/LoRA 注入；forward 不调用 SFT loss
)
```

建议 LoRA A/B 和 optimizer state 保持 FP32；冻结 DiT 保持 BF16。启动时枚举所有 `requires_grad=True` 参数，只允许：

```text
pipe.dit.*.lora_A.*
pipe.dit.*.lora_B.*
```

主方案额外允许 `task_router.*`。VAE、UMT5、冻结 DiT 权重一旦出现在 trainable 列表中应立即失败。

### 7.2 两套文本条件必须分离

Wan cross-attention 的任务输出 prompt：

| task | prompt |
|---|---|
| depth | `output: depth` |
| normal | `output: surface normal` |
| matte | `output: <subject> alpha matte` |
| raymap | `output: camera raymap` |

路由器的固定 task instruction：

| task ID | task | task instruction |
|---:|---|---|
| 0 | depth | `video to depth map` |
| 1 | normal | `video to surface normal map` |
| 2 | matte | `video to alpha matte` |
| 3 | raymap | `video to camera raymap` |

Matte 的 subject 只进入 cross-attention prompt，不进入 task instruction；否则 person/bear 等 subject 会被错误拆成不同的路由任务。task instruction 必须作为 checkpoint contract 固定，不能在训练和推理间自由改写。

两种条件都使用冻结的 Wan UMT5，但分别编码、分别消费：

```text
output prompt -> 完整 [512,4096] context -> Wan cross-attention
task instruction -> 第一个 contextual token [4096] -> TaskHyperRouter
```

这对应 UniControl 官方实现中 `c_text` 与 `c_task` 的分离；不是把同一个 prompt embedding 同时复用两遍。

### 7.3 主方案：Task-aware HyperLoRA

对 Wan2.2 的每个 LoRA residual site `s`：

```text
frozen task instruction
  -> frozen UMT5 first contextual token h_task ∈ R^4096
  -> shared HyperNet: LayerNorm(4096)
                      Linear(4096,512) -> SiLU
                      Linear(512,512)  -> SiLU
  -> site-specific layernet
  -> logits_s ∈ R^rank
  -> g_s = 1 + 0.5 * tanh(logits_s) ∈ [0.5,1.5]

LoRA_s(x, task) = B_s(g_s(task) ⊙ A_s(x)) * scale
```

这等价于：

```text
B_s · diag(g_s(task)) · A_s
```

UniControl 原始 modulated convolution 是对 trainable zero-conv 的 input-channel kernels 做 task scaling。将其迁移到 LoRA 时，LoRA-B 的输入通道正是 rank bottleneck，因此 rank gate 是结构上最直接且参数高效的对应。它是“依据 UniControl 原理进行的 LoRA 适配”，不是声称逐层复刻 ControlNet。

Wan2.2 TI2V-5B 有 300 个 LoRA site、rank 32，因此 layernet 一次输出 `300×32=9600` 个 gate logits。上述 router 约 7,293,312 个可训练参数，远小于约 80.6M LoRA 参数。

初始化必须满足：

- layernet weight/bias 全零，使初始 gate 严格为 1；
- PEFT LoRA-B 保持官方零初始化，使 step 0 与冻结 Wan2.2 完全一致；
- LoRA FP32，router FP32，冻结 DiT BF16；
- router LR 建议先用 `1e-5`，LoRA LR `1e-4`；
- 使用 `mean((g-1)^2)` identity regularization，初始候选权重 `1e-3`。

四卡每个 global step 严格分配四任务各一个样本，使每次同步更新都同时包含四种 route。现有 hook 在 LoRA-A 输出后乘 gate，数学上正好实现上述 `B·diag(g)·A`。

UniControl 提到训练后期冻结 HyperNet，但未给出可迁移的固定比例。Wan2.2 trainer 应提供显式 `--router_freeze_step`，默认 smoke/短实验不冻结；全量 10k 可把 step 8000 作为待验证候选，必须与“不冻结”做小规模消融后再确定，不能悄悄硬编码。

### 7.4 必须保留的 identity ablation

现有 `train_dense4_ddp.py --router_mode identity` 虽然 gate 恒为 1，仍保留 router/hook 代码路径。为了得到最干净的论文 baseline，建议新增一个完全不绑定 TaskHyperRouter 的 shared-LoRA 模式；至少也要用单元测试证明 identity 模式与无 router 的 velocity/gradient 完全一致。

主方案仍需与 identity shared-LoRA 做同条件对照。Task HyperRouter：

- 不是 GenCeption 论文结构；
- 是本项目根据 UniControl task-aware HyperNet 明确要求加入的路由结构；
- 不能用标准 fused LoRA loader；
- 需要专用 checkpoint 和 unfused reload；
- 训练集 latent loss 改善不等于 held-out decoded-space 改善。

因此正式实验应报告 `identity shared LoRA` 与 `task-aware HyperLoRA` 两组；最终交付模型使用后者，前者用于证明路由的净贡献。

### 7.5 对服务器现有路由代码的具体修改

现有：

```text
examples/wanvideo/model_training/dense_lora/task_router.py
```

已经实现了正确的核心形式：冻结 task feature、共享 HyperNet、一次性 parallel layernet、LoRA-A output hook、`1+0.5*tanh` identity gate。Wan2.2 迁移应：

1. 将 `TASK_INSTRUCTIONS` 固定为本节的四条 video-to-modality instruction，并与普通 output prompt 分开保存；
2. LoRA 注入后对 Wan2.2 DiT 重新 `discover_lora_sites`，断言 site 数为 300，禁止沿用旧 checkpoint 的 site list/hash；
3. 使用冻结 Wan UMT5 分别编码四条 instruction，取 `[0,0,:]`，生成 `[4,4096]` feature table；
4. Wan2.2 trainer 每次 forward 前调用 `task_router(task_ids)` 激活 gate；开启 gradient checkpointing 时 gate 必须保留到 backward 完成，之后清除；forward 异常路径单独清除，防止污染下一 batch；
5. DDP batch 当前每 rank 为单一 task，与 UniControl 官方“一 batch 一 task”假设一致；实现仍应保留 `[B]` vectorized task ID，不能依赖 `batch[0]`；
6. 加入 `--router_mode {learned,identity}`、`--router_learning_rate`、`--router_identity_weight` 和 `--router_freeze_step`；
7. `dense4_checkpoint.py` 更新 Wan2.2 base identity、300-site schema，并保存 `task_router.*`、task instructions/features/hash；
8. `validate_wan22_single_forward.py` 必须使用 unfused LoRA + router loader，根据请求模态激活对应 route 后再做 `t=0` forward。

不要增加四套 LoRA 或四个 task-specific decoder。路由发生在一套共享 LoRA 的 rank 通道内，backbone、输出 head、VAE decoder 和 latent MSE 均保持统一。

## 8. Dataset、sampler 与 prompt cache

`CachedDense4Dataset` 的总体逻辑可复用，但常量和 contract 必须更新。每次取样：

```text
source: depth cache 中该 clip 的 source_latent
target:
  depth  -> depth cache 的 target_latent
  others -> dense4 target cache 的 target_latent
task_id: depth=0, normal=1, matte=2, raymap=3
prompt_id: 指向预先编码的 UMT5 context table
```

推荐保留四任务 step-stratified sampler：四卡每步 task ID 恰好为 `{0,1,2,3}`，并随 step/epoch 轮换 task-rank 对应，避免某个 task 永久绑定某张卡。

UMT5 prompt embedding 可以在训练开始时一次编码后缓存：

```text
depth: 1 条
normal: 1 条
raymap: 1 条
matte: 每个 subject 1 条
```

随后释放 text encoder/tokenizer，再把 context table 保留为 BF16。必须在 checkpoint/training args 中保存 prompt 文本列表或 hash，防止 resume 时 prompt ID 顺序变化。

同时单独编码四条固定 task instruction，取每条 UMT5 输出的第一个 contextual token，形成 `[4,4096]` FP32 `task_feature_table`。该表由冻结 UMT5 产生，训练时作为 buffer 保存；HyperNet/LayerNet 可训练。checkpoint reload 必须同时校验 task ID、instruction 文本、feature table、UMT5 checkpoint hash 和 feature hash。

## 9. Loss、数值和 timestep 的关键约束

Wan/DiffSynth 的 flow-matching 约定是：

```text
x_t = (1 - sigma) * x0 + sigma * epsilon
training target = epsilon - x0
timestep = sigma * 1000
```

所以 `t=0` 表示无噪声端。论文并不是在 `t=0` 使用 scheduler 的解析 `x0` 重建公式，而是把 DiT velocity 的负值本身当作新的 dense prediction head，并直接监督目标模态 latent：

```text
prediction = -DiT(z_source, t=0, prompt)
loss = MSE(prediction, z_target)
```

以下行为都应通过代码审查或测试禁止：

- 随机 timestep；
- `scheduler.add_noise`；
- `scheduler.training_target`；
- `scheduler.step`；
- 额外 Gaussian noise；
- CFG 的 conditional/unconditional 双 forward；
- 把 `z_target` 当作 DiT 输入；
- 使用 TI2V `input_image`/首帧 mask 路径；
- 在 BF16 中直接累计 MSE（应转 FP32）。

## 10. 建议训练参数与执行顺序

论文 full fine-tune 参数不能原样当作 LoRA 已验证配置。建议：

### Gate A：结构/缓存检查（CPU）

- 全量 exact path/hash 检查；
- 每个 task 分层抽样反序列化；
- 检查 model config、3 shard、300 LoRA sites；
- `py_compile`。

本轮已通过。

### Gate B：单卡、单 clip overfit

```text
task: 四个 task 分别做单 clip gate；随后做四 task/四 clip 小过拟合
batch: 1
steps: 100-200
rank: 32
LoRA LR: 1e-4
router LR: 1e-5
gradient checkpointing: on
```

验收不只看 latent MSE：还要把 `-velocity` 用 Wan2.2 VAE decode，并保存 source/GT/pred 的第 1、41、81 帧。单 clip loss 明显下降但 decoded 输出错误，仍视为失败。四任务小过拟合还要确认四个 task 的 gate 不再完全相同，且无 gate 边界饱和。

### Gate C：四卡 4-20 step smoke

- global batch = 4（每卡 1）；
- 每个 global step 四任务各 1；
- 只有 LoRA 与 `task_router.*` 有梯度；
- loss、gradient、参数均 finite；
- 每步四个 task 的 active gate 与 task ID 一致，forward 后立即 clear；
- 初始 gate 全 1；更新后 task-pair gate difference 非零；
- 记录未裁剪 grad norm 和峰值显存；
- checkpoint 保存后新建进程 reload，四任务输出逐项一致。

### Gate D：64-clip baseline

用完全相同的 seed、sampler、clip、LoRA 初始化和训练步数分别运行：

```text
A: identity shared LoRA（g=1，不训练 router）
B: task-aware HyperLoRA（主方案）
```

每组 500 steps，必须加入 held-out clips，并比较：

- per-task latent MSE；
- VAE decoded RGB-space MSE/PSNR；
- depth/normal/matte/raymap 的任务指标；
- 可视化与时间一致性。

### Gate E：全量训练

初始建议：

```text
model: Wan2.2-TI2V-5B
resolution/frames: 832×480 / 81
dtype: frozen DiT BF16, LoRA FP32, loss FP32
LoRA: rank 32, q/k/v/o/ffn.0/ffn.2
task router: UMT5 [4,4096] features -> shared 512-d HyperNet -> 300×32 rank gates
optimizer: AdamW
LoRA LR: 1e-4（先沿用官方 LoRA 示例）
router LR: 1e-5
router identity regularization: 1e-3（候选值，先由 64-clip 对照确认）
router freeze step: 显式配置；8000 是待验证候选，不是默认事实
global batch: 4
warmup: 先用 smoke/64-clip 曲线决定；可将 250 作为待验证候选，不当成论文强制值
max grad norm: 先记录分布再定；若沿用现有 1.0，必须在实验 metadata 标明是项目选择
steps: 10,000 可作为首轮预算，不等同论文 15,000 full-finetune 配置
save: 每 500 steps
```

四任务 10k steps、global batch 4 相当于处理 40k task samples，约 1.36 个 29,400-sample epoch。depth-only 10k/global batch 4 则约 5.44 个 7,350-sample epoch。报告实验时应同时写 steps、global batch 和等效 epochs。

论文未公开 gradient dropping 的完整定义/阈值，首版不要自行猜测。可以先实现“只记录异常 grad norm，不丢 batch”，得到分布后再单独设计 ablation。

## 11. Checkpoint 与 resume 契约

LoRA checkpoint metadata 至少应包含：

```text
method: genception_clean_latent_t0
base_model: Wan2.2-TI2V-5B
base_model_config_sha256
ordered DiT shard names + index SHA256
LoRA rank/targets/site list/site hash
depth/target cache schema and manifest SHA256
latent shape/dtype
task ID mapping、output prompt list/hash
task instruction list/hash、UMT5 checkpoint hash
task feature table/hash、HyperNet/layernet config
router mode、gate parameterization/range、freeze step
git commits: Wan2.2 + DiffSynth-Studio
global step/world size/seed
```

resume 时要核对所有影响语义的字段，并恢复 optimizer、scheduler、epoch/batch offset 和每 rank RNG。输出目录非空时默认拒绝覆盖。

主 checkpoint 必须同时保存 LoRA 与 `task_router.*`，并使用专用 unfused loader；官方 fused `pipe.load_lora` 会消除运行时 rank routing，不能用于 task-aware HyperLoRA。identity shared-LoRA 消融 checkpoint 应单独验证官方 loader 兼容性。

## 12. 单次 forward 推理/验证入口

验证脚本应同时支持：

1. 从 cache 直接读取 `source_latent`，用于精确回归；
2. 从新 RGB 视频通过官方 Wan2.2 VAE encode，验证真实部署链路。

核心流程：

```python
with torch.inference_mode():
    context = cached_or_encoded_prompt
    task_ids = torch.tensor([requested_task_id], device=device)
    gates = task_router(task_ids)
    velocity = pipe.model_fn(
        dit=pipe.dit,
        latents=z_source,
        timestep=torch.zeros(z_source.shape[0], device=device, dtype=torch.bfloat16),
        context=context,
        cfg_merge=False,
    )
    task_router.clear()
    z_prediction = -velocity
    video_prediction = wan22_vae.decode([z_prediction[0]])[0]
```

必须确认训练与验证使用相同的官方 Wan2.2 VAE checkpoint和 48-channel mean/std 归一化。输出 modality 的逆表示也应固定：depth 的 inverse-camera-Z 显示/指标、normal 的坐标系、matte 连续 alpha、raymap 的 RGB layout 不得各自临时解释。

## 13. 建议新增的自动化测试

1. `test_wan22_cache_contract.py`
   - schema/count/hash/path/shape/dtype/finite；
   - 随机与固定索引抽样；可选全量反序列化模式。
2. `test_wan22_model_contract.py`
   - 3 shard、config、48 channel、30 blocks、300 LoRA sites；
   - LoRA trainable 参数白名单。
3. `test_genception_forward_contract.py`
   - monkeypatch scheduler，使任何 `add_noise/training_target/step` 调用立即失败；
   - timestep 必须全 0；
   - 每 batch 只调用一次 DiT；
   - loss 左侧必须是 `-velocity`；去掉负号时测试失败。
4. `test_checkpoint_roundtrip.py`
   - 保存、新进程重载、同一输入输出一致；
   - shared LoRA 官方 loader；HyperRouter 专用 loader分别测试。
5. `test_task_embedding_router.py`
   - output prompt 与 task instruction 分开编码；
   - task feature table 固定为 `[4,4096]` 且 finite；
   - Wan2.2 恰好发现 300 sites，gate 为 `[B,300,32]`；
   - zero-init 时 gate bitwise 等于 1，identity hook 与无 hook 输出/梯度一致；
   - 不同 task 更新后 gate 有差异，gate 始终位于 `[0.5,1.5]`；
   - checkpoint roundtrip 后 task features、gates 和 velocity 一致。
6. `test_dense4_sampler.py`
   - 任意 global step 的 task 集合为 `{0,1,2,3}`；
   - epoch 内每个 task 恰好遍历 7,350 clips；
   - resume 后顺序可复现。
7. `test_wan22_decode_triplet.py`
   - 固定输出第 1、41、81 帧；
   - 检查 decode finite、shape `[81,3,480,832]`、reload 一致。

## 14. 主要风险与决策点

1. **LoRA 容量风险。** GenCeption 是全量微调；即使加入 task-aware routing，共享 rank-32 LoRA 仍可能容量不足。用 identity baseline 定量路由收益后，再决定增加 rank 或扩大 target modules。
2. **路由迁移边界。** UniControl 调制 ControlNet zero-conv，本项目调制 LoRA rank bottleneck；这是结构原则迁移而非原样复现，必须与 identity shared LoRA 分开报告。
3. **Task embedding 混淆风险。** output prompt 和 task instruction 若共用同一张表，matte subject 或 prompt 改写可能错误改变 route；两条条件链必须独立并哈希锁定。
4. **只看 latent loss 的风险。** VAE latent MSE 下降不保证 task metric 或 decoded 视觉正确，所有 gate 都要包含 decode 和 held-out 评估。
5. **Depth 表示偏差。** 当前数据使用 clip-global relative inverse camera-Z，而论文描述 median normalization + log mapping。这是项目明确的数据表示差异，报告中必须说明。
6. **TI2V timestep 模式误用。** full clean video 输入要用广播的 scalar `t=0`；不要套用 I2V 首帧 clean mask/token-wise timestep。
7. **服务器工作树风险。** `dense_lora/` 当前整体为 untracked 用户代码，实施时应先归档/commit，再修改；不要清理或覆盖。
8. **资源冲突。** 调研时四卡满载。任何真实 smoke 前先确认作业归属和空闲显存，避免并发 OOM 或影响他人任务。

## 15. 最小实施清单

- [x] 以 DiffSynth 官方 Wan2.2-TI2V-5B LoRA 入口为基座，新建纯 Wan2.2 depth/dense4 trainer。
- [x] 启动时通过 Wan2.2 cache schema、48-channel shape、模型根目录和 VAE hash 契约排除 Wan2.1 runtime。
- [x] 更新 48-channel shape、Wan2.2 schemas、VAE hash 和 checkpoint metadata。
- [x] 用 3-shard 列表加载 TI2V-5B DiT。
- [x] 训练进程不加载 VAE。
- [x] 保持 clean source、数值 `t=0`、单 forward、`-velocity`、FP32 latent MSE。
- [x] 分离 output prompt 与四条固定 task instruction，缓存 `[4,4096]` UMT5 task features。
- [x] 实现 300-site、rank-32 的 `B·diag(g_task,site)·A` TaskHyperRouter 主方案。
- [x] 保存/恢复 LoRA、router、task feature table、site schema 和 task instruction contract。
- [x] 保留 identity shared LoRA 消融模式。
- [x] 通过单 GPU 四任务 smoke 和四任务 checkpoint bitwise roundtrip。
- [ ] 增加 Wan2.2 VAE decoded triplet 验证和单 clip overfit。
- [ ] 在四张训练卡均可用时通过正式四卡 smoke。
- [ ] 完成 identity 与 task-aware HyperLoRA 的同条件 64-clip/500-step 对照。
- [ ] 64-clip held-out decoded 评估通过后才启动全量 10k。
- [ ] 所有 run 固化 cache/model/git/prompt/LoRA metadata 和 resume 契约。

## 16. 最终建议

当前数据不需要重建 cache，Wan2.2 专用 cached-latent trainer 已按本方案实现并通过单卡四任务 smoke。运行链路只使用 Wan2.2 5B 三分片 DiT、冻结 UMT5 和现有 48-channel Wan2.2 cache；主模型是共享 LoRA + UMT5 task embedding HyperRouter，四模态按 `B·diag(g_task,site)·A` 路由，identity shared LoRA 只作为消融。下一关是四卡短 smoke、decoded triplet 验证和 64-clip 对照；旧 Wan2.1 trainer 不进入运行链路。

## 17. 2026-08-30 实施记录

### 17.1 新增服务器文件

代码位于：

```text
/pfs/mayuema/dense/wan/DiffSynth-Studio/examples/wanvideo/model_training/dense_lora/
```

本次只新增 Wan2.2 专用文件，没有覆盖现有 Wan2.1 文件：

```text
train_wan22_dense4_ddp.py             # 正式 trainer；默认只允许 world_size=4
wan22_task_router.py                   # UMT5 task feature -> 300×rank gates
wan22_task_sampler.py                  # 四卡逐 step 均衡；显式单卡 smoke 顺序覆盖四任务
wan22_cache_integrity.py               # cache 路径、manifest 与 hash 契约
wan22_dense4_checkpoint.py             # unfused LoRA + router 严格保存/恢复
validate_wan22_dense4_checkpoint.py    # 四任务输出/gate bitwise roundtrip
test_wan22_task_components.py          # CPU 路由器和 sampler 契约测试
smoke_wan22_dense4_1gpu.sh             # 只占 GPU 0 的四步真实 smoke
train_wan22_dense4_4gpu.sh             # 全量缓存正式四卡 launcher
```

### 17.2 关键实现

- 模型只加载 `Wan2.2-TI2V-5B` 的 3 个 DiT shard 和冻结 UMT5；训练不加载 VAE。
- cache 契约固定为 `wan22_depth_latent_v1`、`wan22_dense4_target_latent_v1`、BF16 `[48,21,30,52]`，并核对 `Wan2.2_VAE.pth` hash。
- 训练目标为一次 `pipe.model_fn`：clean source latent、广播数值 `t=0`、`MSE((-velocity).float(), target.float())`；不调用 scheduler、加噪或 CFG。
- output prompt 继续描述具体输出；独立 task instruction 固定为 `video to depth map`、`video to surface normal map`、`video to alpha matte`、`video to camera raymap`。
- 冻结 UMT5 的首个 contextual token 形成持久化 `[4,4096]` feature table；router 输出 `[B,300,32]` gate，参数化为 `1+0.5*tanh(logits)`。
- LoRA 参数保持 FP32，冻结 Wan2.2 主干；实测 LoRA 80,609,280 参数，router 7,293,312 参数，总可训练 87,902,592。
- 增加 `--router_freeze_step`；默认 `-1` 不冻结。由于 gradient checkpoint backward 会重算 LoRA-A，active gate 保留到 backward/optimizer step 后才清除。
- 正式入口默认拒绝非 4 卡；只有显式 `--single_gpu_smoke` 才允许 world size 1。该服务器单 rank NCCL 会卡在 CXI，smoke 协调改用 Gloo，GPU 运算仍全部在 CUDA 0；正式四卡保持 NCCL。
- checkpoint schema 为 `wan22_ti2v5b_dense4_unfused_hyperlora_v1`，保存 600 个 LoRA tensor、10 个 router tensor、task features、site hash、task instructions、model config/index/shard identity、两个 cache manifest hash。

### 17.3 已执行测试与结果

CPU 测试：

```text
/pfs/mayuema/dense/wan/envs/wan22/bin/python \
  examples/wanvideo/model_training/dense_lora/test_wan22_task_components.py

Wan2.2 Dense4 CPU component contracts: PASS
```

真实 GPU smoke 使用 Wan2.2 smoke1 cache，四步依次覆盖 depth/normal/matte/raymap：

| step | task | loss | grad norm | max GPU memory |
|---:|---|---:|---:|---:|
| 1 | depth | 2.693588 | 0.535560 | 13.989 GiB |
| 2 | normal | 4.208085 | 0.805702 | 14.646 GiB |
| 3 | matte | 5.111416 | 1.357121 | 14.646 GiB |
| 4 | raymap | 4.056458 | 1.699195 | 14.646 GiB |

所有 loss、gradient、latent 和 gate 均 finite。step 4 gate 范围为 `[0.9995767, 1.0004586]`，四任务平均 gate 已出现非零差异，没有边界饱和。

smoke 输出目录：

```text
/pfs/mayuema/dense/wan/outputs/wan22_dense4_hyperlora_smoke1_20260830
```

`step-4.safetensors` 保存后重新构建完整 Wan2.2 module、严格加载，再对四个任务重复 forward；四个任务的 `output_max_abs_diff` 和 `gate_max_abs_diff` 均为 `0.0`，即 bitwise roundtrip 通过。结果见：

```text
/pfs/mayuema/dense/wan/outputs/wan22_dense4_hyperlora_smoke1_20260830/roundtrip_validation.json
```

### 17.4 GPU 进程处理记录

原 GPU burn 由一个带联动 cleanup trap 的父脚本管理。终止 GPU 0 worker 时，该 trap 同时结束了 GPU 1--3 worker；发现后立即以完全相同参数把 GPU 1--3 分别独立恢复。最终状态为 GPU 0 空闲约 3 MiB，GPU 1--3 各由独立 `gpu_burn.py --reserve-gib 42` 占用约 45,227 MiB，不再存在父脚本联动。

### 17.5 尚未完成与下一步

本轮没有释放 GPU 1--3，因此没有冒充完成正式四卡 smoke。单卡 smoke 已覆盖真实 5B forward/backward、四任务路由、保存和重载，但正式训练前仍建议：

1. 在四卡都获准释放后运行 `train_wan22_dense4_4gpu.sh` 的 4--20 step 版本，验证 NCCL、逐 step 四任务集合与跨 rank resume。
2. 新增加载 `Wan2.2_VAE.pth` 的 decoded 第 1/41/81 帧验证；当前 roundtrip 是 latent/velocity 级别。
3. 做 identity 与 learned router 的 64-clip/500-step 对照，再决定是否采用候选 `router_freeze_step=8000`。

## 18. 评审问题修复与四卡复测（2026-08-30）

### 18.1 P1：depth cache 断点续建

`build_wan22_depth_cache.py` 已升级到 `wan22_depth_latent_v2`：

- manifest contract 包含 VAE、官方 VAE 源码、dataset index、有序 clip IDs、相对路径、builder/dataset 源码、亮度阈值、latent shape/dtype 和 prompt，并计算 `cache_contract_hash`；
- 开始构建时先删除 `CACHE_COMPLETE`，manifest 原子替换；
- payload 持久化相同 contract hash；resume 必须完整 `torch.load` 并检查 schema/hash/clip/prompt、两个 BF16 `[48,21,30,52]` latent、finite 和标量字段；文件大小不再构成有效性；
- 完成时核对精确路径集合，`CACHE_COMPLETE` 锁定 contract hash 和 manifest SHA256。

target cache 同步升级为 `wan22_dense4_target_latent_v2`，要求 depth v2 complete/manifest/contract/path 集合一致，并把 source contract hash 写入自身 contract/payload。Trainer 会重算两个 manifest 的内容 hash并全量审计 payload contract。

实际复测新 1-clip depth cache：首次 `written=1 skipped=0`；同配置续建时 `written=0 skipped=1`，且 complete 被重新生成。旧 full v1 payload 无法可靠证明 provenance，因此没有伪造迁移；新 trainer 会拒绝旧 v1 full cache。正式全量训练前必须重建：

```text
/pfs/mayuema/dense/wan/cache/wan22_depth_latents_v2_bf16_bright40
/pfs/mayuema/dense/wan/cache/wan22_dense4_targets_v2_bf16_bright40
```

### 18.2 P1：严格 checkpoint v2

schema 已升级为 `wan22_ti2v5b_dense4_unfused_hyperlora_v2`。必需并在加载时逐项匹配的 provenance 包括：

- 三个 DiT shard 的文件名、大小和完整 SHA256，以及 config/index SHA256；
- UMT5 权重 SHA256、tokenizer 目录内容 hash；
- output prompt 完整 JSON/hash、prompt JSONL manifest hash；
- checkpoint 内 task feature tensor SHA256，并校验 tensor 与 metadata 内部一致；
- depth/target manifest SHA256 和 cache contract hash；
- DiffSynth Git HEAD；
- 完整运行源码 patch SHA256。

validator 使用当前 UMT5、tokenizer、prompt、cache 和源码 patch重新计算 expected provenance，语义不一致时在复制权重前失败。

### 18.3 P2：模型硬契约

Trainer 在模型加载/LoRA 注入时强制：恰好三个标准 shard 名、`in_dim=48`、`out_dim=48`、30 blocks、patch `(1,2,2)`、`seperated_timestep=True`、`fuse_vae_embedding_in_latents=True`、恰好 300 个 LoRA site，且每个 LoRA-A rank 与参数一致。宽泛 glob 或错误 TI2V 目录不再被接受。

### 18.4 P2：源码追溯

整个 `dense_lora/` 是用户原有 untracked 目录，未把不属于本次的文件擅自 commit。改为保存该目录全部 36 个 `.py/.sh` 的完整新增文件 patch：

```text
/pfs/mayuema/dense/wan/docs/wan22_dense4_runtime_source.patch
SHA256: 39e5adc3f5f8f602c3ada51f6f4a862fa4d1ef315a493a05495d49377d09a53e
DiffSynth HEAD: 102fe9980b9375ecb6436d360297a00327472535
```

训练入口要求显式 `--code_patch_path`，checkpoint 锁定其 hash。该 patch 是本次四卡运行时的不可变源码快照。

### 18.5 P2：测试扩充与四卡结果

CPU 测试新增：identity hook 与无 hook 的输出/输入梯度/权重梯度 bitwise 等价；sampler resume suffix；router state roundtrip；异常路径 gate 清理；AST 契约断言只调用一次 `model_fn`、不出现 scheduler/add_noise/training_target/step，loss 使用 `-velocity`。

四卡测试输出：

```text
/pfs/mayuema/dense/wan/outputs/wan22_dense4_v2_4gpu_uninterrupted
/pfs/mayuema/dense/wan/outputs/wan22_dense4_v2_4gpu_resumed
```

结果：

- uninterrupted 4 steps 通过，step 2/4 checkpoint 均成功；
- split run 先到 step 2，再由四个 rank 从 `step=2 epoch=2 batch=0` 恢复到 step 4；
- final trainer `global_step=4`，scheduler `last_epoch=4`，四份 rank RNG state 均存在；
- loss CSV 共 16 行，每个 global step 的 task IDs 都严格为 `{0,1,2,3}`；
- resumed step-4 严格重载后，四任务 output/gate roundtrip 最大误差全部为 `0.0`；
- checkpoint metadata 包含上述全部 provenance，600 个 LoRA tensor 和 10 个 router tensor。

本节点 NCCL 默认 CXI 和强制 Socket 都卡在 communicator 初始化，因此四卡 smoke 显式使用 `--distributed_backend gloo`；模型计算仍在四张 CUDA 卡，正式入口默认仍为 NCCL且不会静默降级。正式长训练前需要单独修复 NCCL/CXI 环境。

两次独立 fresh run 虽使用相同 seed，最终 checkpoint tensor 最大绝对差为 `5.4577e-4`，说明当前 CUDA/Gloo 路径不是跨进程启动 bitwise deterministic。resume 本身已验证完整状态加载和连续执行，但不能把两个独立 fresh run 的 bitwise equality 当作验收条件；若研究需要精确复现，应另加 deterministic algorithms/CUBLAS 配置并评估性能。

## 19. 切换到 cog 环境（2026-08-30）

按使用要求，训练、单/四卡 smoke、validator 和 v2 cache launcher 已统一切换到：

```text
/pfs/mayuema/dense/wan/envs/cog
Python 3.10.20
PyTorch 2.12.0+cu130
CUDA 13.0
NCCL 2.29.7
```

cog 缺少 `modelscope`，因此训练/validator 通过 `cog_runtime_bootstrap.py` 启动：cog site-packages 保持最高优先级，只从 Wan2.2 Python 3.11 site-packages 末尾补充缺失的纯 Python 依赖。DiffSynth trainer `--help`、CPU 契约测试和实际完整模型加载均已通过。官方 Wan2.2 VAE 可直接在 cog 中导入，因此 cache builder 不需要 bootstrap。

PyTorch 2.12 + NCCL 需要两项适配：

1. `broadcast_object_list` 显式使用当前 CUDA device；默认 CPU staging tensor 不能交给 NCCL。
2. 本容器的 NCCL SHM attach 会报 `/dev/shm/nccl-*` 不存在；launcher 固定：

```bash
export NCCL_SHM_DISABLE=1
export NCCL_NET=Socket
export NCCL_SOCKET_IFNAME=lo
```

这会保留 CUDA/NCCL DDP，但禁用有问题的 SHM transport并用 loopback Socket 做本机进程通信。后续若节点的 `/dev/shm`/IPC 配置修好，可重新 benchmark 并移除这些保守变量。

cog 原生 NCCL 四卡测试输出：

```text
/pfs/mayuema/dense/wan/outputs/wan22_dense4_v2_cog_nccl_smoke3
```

测试从 2 steps 保存后由四个 rank 恢复到 step 4，成功生成 `step-4.safetensors`。第一步约 8.04 秒，峰值显存约 14.04 GiB；loss/gradient finite，300-site router 正常分化。该次测试锁定的历史源码 patch为：

```text
/pfs/mayuema/dense/wan/docs/wan22_dense4_runtime_source_cog_nccl_smoke3.patch
SHA256: 2678be9a67bc649217a0f00d8dd125bfd917e1551f5c1370cd82f3f447c4f981
```

当前 launcher 在测试后进一步固化了上述三个 NCCL 环境变量，因此当前源码 patch hash 与历史测试快照不同：

```text
/pfs/mayuema/dense/wan/docs/wan22_dense4_runtime_source.patch
SHA256: 3f94adfd1482d5714a3219017e1176d51728d446d3ffaedf610203d5d6e675bb
```

两份 patch 均保留，已有 smoke checkpoint 用历史文件继续严格验证，新训练使用当前 patch。

## 20. 正式训练完成与四模态推理（2026-08-30）

正式四卡训练已正常完成 `10000/10000` optimizer steps，最终 checkpoint、optimizer/scheduler state 和四个 rank 的 RNG state 均已落盘：

```text
/pfs/mayuema/dense/wan/outputs/wan22_dense4_hyperlora/step-10000.safetensors
/pfs/mayuema/dense/wan/outputs/wan22_dense4_hyperlora/trainer_state_latest.pt
/pfs/mayuema/dense/wan/outputs/wan22_dense4_hyperlora/trainer_rng_rank{0,1,2,3}_latest.pt
```

最后一步四任务平均 loss 为 `0.036674`。最后 100 steps 的逐任务均值约为：depth `0.0435`、normal `0.0976`、matte `0.0110`、raymap `0.0171`。训练全程未出现 non-finite、OOM 或 NCCL 错误。

### 20.1 推理代码

正式推理入口：

```text
/pfs/mayuema/dense/wan/DiffSynth-Studio/examples/wanvideo/model_training/dense_lora/infer_wan22_dense4.py
/pfs/mayuema/dense/wan/DiffSynth-Studio/examples/wanvideo/model_training/dense_lora/run_wan22_dense4_inference.sh
```

源码 SHA256：

```text
infer_wan22_dense4.py                 cbfbe6b08d5b9ed278f929051649cb2b2601c30e988edeb23dcada3b88dd685a
run_wan22_dense4_inference.sh         ecac75418ad220f3d7ce9fc314a271cf7c72f400a8f1a5f6bebff75cebfdc949
```

推理保持训练语义：使用官方 Wan2.2 `Wan2_2_VAE` 将输入视频编码成 `[48,21,30,52]` clean latent，在 `t=0` 对每个选定任务各执行一次 DiT forward，并以 `-velocity` 作为预测 target latent。一个共享 LoRA checkpoint 不做 fuse；checkpoint 内的 task HyperNet 分别激活 depth、normal、matte、raymap route。

推理会重新构造并严格校验训练时的 base model、UMT5、tokenizer、prompt、task feature、cache manifest、Git HEAD 和不可变训练源码 patch provenance。matte prompt 必须使用 checkpoint 中存在的主体标签，目前支持：`bear, crane, creature, crow, dog, gorilla, horse, lion, monkey, person, robot`。

### 20.2 对任意视频运行四模态

输入契约为训练规格：81 帧、`832x480`、RGB；默认拒绝平均亮度低于 40 的域外输入，可显式传 `--min_rgb_mean 0` 覆盖。

```bash
cd /pfs/mayuema/dense/wan/DiffSynth-Studio

CUDA_VISIBLE_DEVICES=0 \
examples/wanvideo/model_training/dense_lora/run_wan22_dense4_inference.sh \
  --input_video /absolute/path/to/input.mp4 \
  --output_dir /absolute/path/to/output \
  --tasks all \
  --matte_subject person
```

可用 `--tasks depth,normal` 等形式只运行子集。也可绕过 VAE encode，直接输入训练同规格 latent：

```bash
CUDA_VISIBLE_DEVICES=0 \
examples/wanvideo/model_training/dense_lora/run_wan22_dense4_inference.sh \
  --source_latent /absolute/path/to/cache_payload.pt \
  --output_dir /absolute/path/to/output \
  --tasks all \
  --matte_subject person
```

每个任务目录包含：

```text
prediction.mp4          # 81 帧、832x480 的预测可视化
comparison.mp4          # 左侧 RGB input，右侧任务预测
preview.png             # 第 1/21/41/61/81 帧纵向预览
prediction_latent.pt    # 原始 BF16 [48,21,30,52] 预测 latent
```

根目录同时保存 `manifest.json` 和仅在全部请求任务成功后创建的 `INFERENCE_COMPLETE`。默认拒绝非空输出目录；显式 `--overwrite` 才允许覆盖同名产物。

### 20.3 四模态端到端 smoke

已使用真实 81 帧视频完成严格 checkpoint load、官方 VAE encode、四次 routed forward、四次 VAE decode 和视频写出：

```text
输入：/pfs/mayuema/dense/ditto/videos_832x480_16fps_81f/3048166_fauxels_1.mp4
输出：/pfs/mayuema/dense/wan/outputs/wan22_dense4_infer_smoke_20260830
checkpoint：step-10000
任务：depth, normal, matte(person), raymap
```

四个 `prediction.mp4` 经 `ffprobe` 验证均为 81 帧、`832x480`、16 fps；四个预测 latent 均为有限 BF16 `[48,21,30,52]`，并已生成 `INFERENCE_COMPLETE`。预览人工检查显示四路输出均非退化且路由语义正确：depth 为连续逆深度、normal 为方向颜色、person matte 为前景二值/软遮罩、raymap 为相机射线颜色编码。

## 21. LingBot depth/normal 增量数据与空洞 masked loss（2026-08-31）

新增数据：

```text
/pfs/mayuema/dense/wan/cache/wan22_lingbot_depth_latents_v2_bf16_bright40
/pfs/mayuema/dense/wan/cache/wan22_lingbot_normal_latents_v1_bf16_bright40
```

两份 cache 各有 2143 个完全同序 clip。原始 depth/normal 在 VAE encode 前已经对传感器空洞做 nearest fill，因此不能从 latent 数值反推出空洞；原始逐帧二值 mask 位于每个 clip 的 `valid_mask/000001.png ... 000081.png`。为避免重新做昂贵的 VAE cache，新增了独立 mask cache：

```text
/pfs/mayuema/dense/wan/cache/wan22_lingbot_validity_v1_bright40
schema: wan22_lingbot_latent_validity_v1
payload: FP16 [1,21,30,52]
samples: 2143/2143
pixel_valid_fraction_mean: 0.925315
latent_valid_fraction_mean: 0.925284
```

mask 映射与 Wan2.2 VAE 的时空降采样规格一致：空间每个 latent cell 使用对应 `16x16` 像素的有效比例；时间上 `latent[0]=frame[0]`，其余每个 latent timestep 对相邻 4 帧取平均。训练 loss 为：

```text
sum((prediction-target)^2 * validity) / (sum(validity) * 48 channels)
```

因此完全空洞 cell 权重为 0，部分有效 cell 按有效像素比例计权，并按有效权重重新归一化，避免不同空洞比例改变 loss 尺度。原 Dense4 样本使用全 1 mask，数学上与原始 MSE 完全等价。注意这是一种 latent-grid 的面积权重近似；VAE 卷积感受野跨越 cell 边界，若要求严格的逐像素排除，只能增加 VAE decode 后的像素域 loss，训练代价会明显上升。

混合数据集包含 33686 个可索引样本：原始四任务 `7350*4`，加 LingBot depth/normal `2143*2`。四任务 pool 大小为 depth=9493、normal=9493、matte=7350、raymap=7350。按实际 pool 数量做确定性全局 shuffle，每个 epoch 每条样本恰好使用一次；四卡对 33686 做最小 padding 后为 8422 optimizer steps，仅重复 2 条样本。任务长期占比为 depth 28.18%、normal 28.18%、matte 21.82%、raymap 21.82%，即允许新增的 depth/normal 获得更高权重。单个全局 step 不再强制四任务各一条，但四卡 DDP 和 task router 均支持任意任务组合，resume 顺序仍由 seed/epoch/step 唯一决定。

训练支持两种严格语义：`--resume_from` 恢复同一混合实验的模型、optimizer、scheduler、RNG 和数据顺序；`--init_checkpoint` 只从已完成的原 Dense4 checkpoint 初始化 LoRA/router，step、optimizer、scheduler 和 RNG 全部重新开始。stage2 launcher 使用后一种方式，并分别校验旧 checkpoint 的旧 source patch 与新 checkpoint 的 stage2 source patch。新 checkpoint provenance 额外锁定 LingBot depth、normal、validity 三份 cache 的绝对路径、schema、manifest SHA256、contract hash 和样本数。

相关代码：

```text
examples/wanvideo/model_training/dense_lora/build_wan22_lingbot_mask_cache.py
examples/wanvideo/model_training/dense_lora/wan22_masked_loss.py
examples/wanvideo/model_training/dense_lora/wan22_task_sampler.py
examples/wanvideo/model_training/dense_lora/train_wan22_dense4_ddp.py
examples/wanvideo/model_training/dense_lora/train_wan22_dense4_lingbot_4gpu.sh
examples/wanvideo/model_training/dense_lora/test_wan22_lingbot_dataset.py
```

CPU component contract 和真实 cache dataset smoke 均已通过；真实 smoke 读到总样本 33686、LingBot 样本对 2143，测试 clip 的 latent mask 均值为 0.908203。当前四张 GPU 被另一用户的 14B 推理进程占满（约 74 GiB/卡），未擅自终止该任务，因此本次尚未运行 GPU forward/DDP smoke，也未启动正式 stage2 训练。
