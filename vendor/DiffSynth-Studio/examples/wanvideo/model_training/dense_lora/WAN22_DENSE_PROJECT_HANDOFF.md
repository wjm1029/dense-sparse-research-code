# Wan2.2 Dense / Causal / AR 项目交接文档

更新日期：2026-09-29

## 1. 项目概况

目标是把 Wan2.2-TI2V-5B 改造成单次 `t=0` forward 的多任务视频稠密预测模型，并逐步扩展为 causal、chunkwise AR、长视频 self-forcing 和 OPSD prefix post-training。

支持的任务：

- depth（相对逆深度）
- normal
- matte
- raymap

所有阶段都是从前一阶段 checkpoint 继续训练。已有 LoRA、任务路由器等参数没有重新初始化；只有新增模块按各阶段设计初始化。官方 Wan2.2 源码不直接修改，自定义实现集中在 `examples/wanvideo/model_training/dense_lora/`。

## 2. 服务器与环境

登录：

```bash
ssh sz6671
```

当前机器信息：

- 用户：`root`
- GPU：4 卡，每卡约 96 GiB
- 仓库：`/pfs/mayuema/dense/wan/DiffSynth-Studio`
- Git 分支：`causal-training-wan22`
- Python：`/pfs/mayuema/dense/wan/envs/cog/bin/python`
- Torchrun：`/pfs/mayuema/dense/wan/envs/cog/bin/torchrun`
- Python 3.10.20，PyTorch 2.12.0 + CUDA 13.0
- Wan2.2 模型目录：`/pfs/mayuema/dense/wan/Wan2.2-TI2V-5B`
- 官方 Wan2.2 仓库：`/pfs/mayuema/dense/wan/Wan2.2`
- Prompt：`/pfs/mayuema/dense/prompts`

不要直接执行训练 Python 文件。统一通过 Cog bootstrap：

```text
examples/wanvideo/model_training/dense_lora/cog_runtime_bootstrap.py
```

训练前必须检查：

```bash
cd /pfs/mayuema/dense/wan/DiffSynth-Studio
git status --short
git branch --show-current
nvidia-smi
df -h /pfs
```

截至交接时 `/pfs` 已接近满盘，只剩约 18 TB。不要随意复制完整 cache、模型目录或长视频 latent。

## 3. Git 与代码管理要求

1. 所有新实现必须放在 `dense_lora/` 独立文件中，不能直接改官方 Wan2.2。
2. 每次训练前提交代码，用小而清晰的 commit 记录修改。
3. launcher 会生成 immutable runtime patch，并把 Git revision、patch hash、数据 cache contract、初始 checkpoint hash写入 checkpoint。
4. checkpoint 加载默认做严格 provenance 校验；不能为了方便删除校验。
5. reviewer 可以这样查看：

```bash
ssh sz6671
cd /pfs/mayuema/dense/wan/DiffSynth-Studio
git log --oneline --decorate -30
git show <commit>
git diff <base_commit>..<target_commit>
```

当前 worktree 有以下历史遗留的未跟踪 cache 构建文件，不属于本交接文档提交，不能误删或未经 review 直接提交：

```text
build_lingbot_depth477_index.py
build_wan22_depth477_cache.py
build_wan22_lingbot_normal477_cache.py
build_wan22_long477_cache.py
dataset_477.py
run_wan22_long477_cache_4gpu.sh
```

其中部分 SHA-256 已写入现有 cache manifest；现有 cache 可复现性依赖这些精确文件内容。

## 4. 模型阶段与 checkpoint 继承链

| 阶段 | 说明 | 推荐 checkpoint |
|---|---|---|
| Bidirectional Stage-2 | 原始离线四任务模型 | `/pfs/mayuema/dense/wan/outputs/wan22_dense4_hyperlora_lingbot_stage2_20k/step-20000.safetensors` |
| Causal Stage-1 | 只替换 temporal attention，无目标历史 AR | `/pfs/mayuema/dense/wan/outputs/wan22_dense4_hyperlora_causal_stage1/step-30000.safetensors` |
| AR teacher forcing | chunk=3 latent frames，用 GT 上一块 | `/pfs/mayuema/dense/wan/outputs/wan22_dense4_hyperlora_ar_tf_chunk3/step-29000.safetensors` |
| 长视频 AR self-forcing | 477 帧，回灌模型预测，normal 已 flip YZ | `/pfs/mayuema/dense/wan/outputs/wan22_long477_ar_self_forcing_flip_yz_v3/step-3200.safetensors` |
| OPSD full-prefix | full history prefix distillation | `/pfs/mayuema/dense/wan/outputs/wan22_long477_opsd_prefix_t0_full_1200/step-1200.safetensors` |
| OPSD local-12 | sink=3、window=12 latent frames | `/pfs/mayuema/dense/wan/outputs/wan22_long477_opsd_prefix_t0_local12_3500/step-3500.safetensors` |

继承关系：

```text
Bidirectional Stage-2
  -> Causal Stage-1
  -> AR teacher forcing
  -> long-477 AR self-forcing
     -> OPSD full-prefix
     -> OPSD local-12（从干净的 self-forcing checkpoint 单独开始，不从 full 继续）
```

## 5. 核心代码入口

训练：

```text
train_wan22_dense4_ddp.py                     # bidirectional 多任务训练
train_wan22_causal_ddp.py                     # causal attention Stage-1
train_wan22_ar_tf_ddp.py                       # chunkwise AR teacher forcing
train_wan22_long477_ar_self_forcing_ddp.py     # 477 帧 free-running/self-forcing
train_wan22_long477_opsd_prefix_ddp.py         # OPSD prefix post-training
```

关键模块：

```text
wan22_task_router.py                 # task HyperLoRA/router
wan22_dense4_checkpoint.py           # checkpoint 与 provenance 严格校验
wan22_causal_attention.py            # causal mask、per-layer KV cache、RoPE offset
wan22_ar_teacher_forcing.py          # AR teacher forcing
wan22_ar_self_forcing.py             # self-forcing rollout
wan22_opsd_prefix_cache.py           # OPSD teacher/student prefix cache
wan22_opsd_bounded_causal_attention.py
wan22_prerope_kv_pool.py             # local-12 pre-RoPE KV pool
wan22_cache_intervention.py           # 推理时静态场景 cache intervention
wan22_masked_loss.py
```

推理与评测：

```text
infer_wan22_dense4.py                  # 单 case bidirectional
infer_wan22_dense4_resident_batch.py   # 常驻模型批量 bidirectional
infer_wan22_causal_resident_batch.py
infer_wan22_ar_tf_long_video.py        # 477 帧 free-running AR
infer_wan22_bidir_benchmarks.py
eval_wan22_long477_gt.py
make_wan22_bidir_five_panel.py
```

更完整的阶段说明见：

```text
WAN22_CAUSAL_TRAINING.md
WAN22_OPSD_PREFIX_POST_TRAINING.md
WAN22_STREAMING_DENSE_PROGRESS_REPORT.md
```

## 6. 数据位置

### 6.1 早期 81 帧数据

- 原始训练集：`/pfs/mayuema/dense/dataset/prod_final_9110_20260816/train`
- RGB/source latent cache：`/pfs/mayuema/dense/wan/cache/wan22_depth_latents_bf16_bright40`
- 四任务 target cache：`/pfs/mayuema/dense/wan/cache/wan22_dense4_targets_bf16_bright40`
- LingBot depth：`/pfs/mayuema/dense/wan/cache/wan22_lingbot_depth_latents_v2_bf16_bright40`
- LingBot normal：`/pfs/mayuema/dense/wan/cache/wan22_lingbot_normal_latents_v1_bf16_bright40`
- LingBot valid mask：`/pfs/mayuema/dense/wan/cache/wan22_lingbot_validity_v1_bright40`

### 6.2 477 帧长视频数据

主合成数据：

- 原始数据：`/pfs/mayuema/dense/long481_prod3000_20260906/filtered_motion_ok_brightness_gt40/outputs/train`
- RGB/depth/normal/matte cache：`/pfs/mayuema/dense/wan/cache/wan22_long477_rgb_depth_normal_matte_v1_bf16_bright40`
- 修正坐标系的 normal cache：`/pfs/mayuema/dense/wan/cache/wan22_long477_normal_flip_yz_v1_bf16_bright40`

该版本共 1079 clips，选择原 481 帧中的第 1..477 帧；输入为 `[477,3,480,832]`，VAE latent 为 `[48,120,30,52]`、BF16。

LingBot：

- 原始 clips：`/pfs/mayuema/dense/lingbot/lingbot-depth_wan832_depth477/clips`
- Depth cache：`/pfs/mayuema/dense/wan/cache/wan22_lingbot_depth477_latents_v1_bf16_bright40`
- Normal cache：`/pfs/mayuema/dense/wan/cache/wan22_lingbot_normal477_latents_v1_bf16_bright40`

现有 cache 约占：主 cache 73 GB、修正 normal 19 GB、LingBot depth 12 GB、LingBot normal 5.8 GB。

每个 cache 目录都有 `manifest.json`、完成标记和哈希。训练前先检查 manifest，禁止混用不同 contract 的 cache。

### 6.3 Benchmark 与真实视频

- Benchmark：`/pfs/mayuema/dense/benchmarks`
- 通用真实视频：`/pfs/mayuema/dense/videos`
- 2026-09-28 精选实拍输入：`/pfs/mayuema/dense/wan/inputs/selected_real_videos_20260928`
- 对应 bidirectional 输出：`/pfs/mayuema/dense/wan/outputs/wan22_bidir_stage2_20k_selected_real_20260928`
- 对应五联画：`/pfs/mayuema/dense/wan/outputs/wan22_bidir_stage2_20k_selected_real_20260928_five_panel`

## 7. 数据处理规范

### 7.1 空间与时间

1. 目标分辨率固定为 `832x480`，比例为 `26:15`。
2. 必须先做最大居中裁剪到 `26:15`，再 resize；禁止直接 resize 导致形变。
3. 81 帧模型必须恰好输入 81 帧；长视频模型必须恰好输入 477 帧。
4. 标准展示视频使用 16 FPS。
5. 长视频 AR 推理代码会严格检查 `477 帧 + 832x480 + 16 FPS`。

注意：2026-09-28 精选实拍 batch 的推理输入是“原视频解码后的前 81 帧”，没有先按时间重采样到 16 Hz；最终五联画按 16 FPS 编码。如果要做严格的 16 Hz 科学评测，应先把源视频重采样到 16 FPS，再取 81 帧并重新推理。

### 7.2 Depth

- 模型输出是相对逆深度，不是米制深度。
- 有 GT 时，在逆深度空间拟合 scale + shift，消除相对尺度/偏移；不需要把模型输出强行还原为米制绝对深度。
- 长视频 cache 规则：ray distance 转 camera-Z，取 inverse，在整段 clip 上用 p2/p98 归一化，再映射到 `[-1,1]`。
- 展示规则：整段视频统一 p2/p98，使用 matplotlib `Spectral_r`；不要逐帧 min-max，否则会掩盖尺度漂移并产生颜色闪烁。

### 7.3 Normal

- 原始 metadata 是 OpenCV camera coordinates。
- 进入 VAE 前统一变换为 `[x,-y,-z]`。
- 早期未 flip 的 normal cache 会造成明显色调/方向偏差，不要再用于正式长视频训练。

### 7.4 Matte 与 prompt

- 推理 prompt 必须来自 checkpoint metadata，不能手写另一套 prompt。
- 当前 matte 默认 `matte_subject=person`，与既有训练/评测脚本一致。
- Dark sample 过滤阈值通常为 RGB mean >= 40。

## 8. 训练设计

### 8.1 Causal 与 KV cache

- 81 RGB 帧经 VAE 得到 21 latent frames。
- causal mask 只允许当前 frame/chunk 访问当前与过去，chunk 内保持双向 attention。
- streaming 使用每层 KV cache，并在全局时间位置应用 RoPE。
- 必须先通过 full causal forward 与 streaming cached forward 的一致性测试。

### 8.2 AR teacher forcing / self-forcing

- chunk size 固定为 3 latent frames，约对应 12 个 RGB 帧。
- teacher forcing：当前 chunk 使用 GT 目标历史。
- self-forcing：训练 rollout 时把模型生成的上一 chunk 回灌，减少训练/推理不一致。
- 当前长视频模型有 120 latent frames，共 40 个 AR chunks。

### 8.3 OPSD prefix post-training

- Student 使用生成历史和自己的 KV cache。
- Teacher 是 trainable post-training 参数的 EMA。
- Teacher 用 clean GT prefix replay 替换更早的生成 cache span。
- 保留任务损失，并加 `t=0` masked x0 prefix distillation loss。
- Full-prefix 是质量优先版本；local-12 使用 3-frame attention sink + 12-frame sliding window，显存更低但改变 context 分布。
- OPSD-V 参考仓库：`/pfs/mayuema/dense/wan/third_party/OPSD-V`
- 固定参考 commit：`8ecb193fea884efdbea6e731498ce0853cae180b`

## 9. 训练启动方式

所有命令在仓库根目录执行。

### 9.1 Causal Stage-1

```bash
cd /pfs/mayuema/dense/wan/DiffSynth-Studio
MAX_TRAIN_STEPS=30000 \
OUTPUT_PATH=/pfs/mayuema/dense/wan/outputs/<new_output> \
bash examples/wanvideo/model_training/dense_lora/train_wan22_causal_4gpu.sh
```

### 9.2 AR teacher forcing

```bash
MAX_TRAIN_STEPS=30000 \
OUTPUT_PATH=/pfs/mayuema/dense/wan/outputs/<new_output> \
bash examples/wanvideo/model_training/dense_lora/train_wan22_ar_tf_4gpu.sh
```

### 9.3 长视频 AR self-forcing

```bash
MAX_TRAIN_STEPS=3200 SAVE_STEPS=25 \
OUTPUT_PATH=/pfs/mayuema/dense/wan/outputs/<new_output> \
bash examples/wanvideo/model_training/dense_lora/train_wan22_long477_ar_self_forcing_4gpu.sh
```

### 9.4 OPSD full-prefix

```bash
ATTENTION_CONTEXT=full MAX_TRAIN_STEPS=<steps> SAVE_STEPS=25 \
OUTPUT_PATH=/pfs/mayuema/dense/wan/outputs/<new_output> \
bash examples/wanvideo/model_training/dense_lora/train_wan22_long477_opsd_prefix_4gpu.sh
```

### 9.5 OPSD local-12

```bash
MAX_TRAIN_STEPS=3500 SAVE_STEPS=100 \
OUTPUT_PATH=/pfs/mayuema/dense/wan/outputs/<new_output> \
bash examples/wanvideo/model_training/dense_lora/train_wan22_long477_opsd_prefix_local12_3500.sh
```

不要覆盖已有 output。继续中断训练时应使用 trainer 的 `--resume_from <原输出目录>`，恢复 optimizer、scheduler 和 step；不能简单把最后一个 safetensors 当作新初始化来冒充严格 resume。现有 shell launcher 没有统一暴露 `RESUME_FROM`，需要在 review 后给 launcher 增加显式参数，或直接按其 torchrun 命令补 `--resume_from`。

## 10. 测试与推理

### 10.1 关键单测

运行模式：

```bash
cd /pfs/mayuema/dense/wan/DiffSynth-Studio
PY=/pfs/mayuema/dense/wan/envs/cog/bin/python
BOOT=examples/wanvideo/model_training/dense_lora/cog_runtime_bootstrap.py
$PY $BOOT examples/wanvideo/model_training/dense_lora/test_wan22_causal_attention.py
$PY $BOOT examples/wanvideo/model_training/dense_lora/test_wan22_ar_teacher_forcing.py
$PY $BOOT examples/wanvideo/model_training/dense_lora/test_wan22_ar_self_forcing.py
$PY $BOOT examples/wanvideo/model_training/dense_lora/test_wan22_opsd_prefix_cache.py
$PY $BOOT examples/wanvideo/model_training/dense_lora/test_wan22_opsd_bounded_causal_attention.py
$PY $BOOT examples/wanvideo/model_training/dense_lora/test_wan22_prerope_kv_pool.py
$PY $BOOT examples/wanvideo/model_training/dense_lora/test_wan22_cache_intervention.py
```

### 10.2 长视频 AR 推理

入口：

```text
infer_wan22_ar_tf_long_video.py
```

输入 manifest 中每个视频必须严格为 477 帧、832x480、16 FPS。`--tasks` 可指定 `depth,normal,matte,raymap`。

local-12 checkpoint 可选推理时 cache intervention：

```text
--cache_intervention
--cache_static_enter_threshold 12
--cache_static_exit_threshold 18
--cache_static_confirm_chunks 2
--cache_static_max_hold_chunks 12
--cache_scene_cut_threshold 60
```

该功能是 inference-time、training-free 的开关，只适用于 bounded/local-12 cache 语义。一次真实长视频实验中只触发 5/40 chunks，normal 稳定性略好、depth 略差，整体收益中性，不应默认开启或宣称明确提升。

### 10.3 Bidirectional 五联画

布局：

```text
RGB | Depth (Spectral_r) | Normal | Matte | Raymap
```

生成器：`make_wan22_bidir_five_panel.py`。输出固定 H.264、yuv420p、16 FPS、81 帧；Depth 使用整段 p2/p98 + `Spectral_r`。

## 11. 已知结论与风险

1. Causal 不等于 AR：只有 causal mask 时，depth frame 之间没有显式目标历史依赖。
2. Teacher forcing 会看到 GT 目标历史；self-forcing 才会回灌模型自己的输出。
3. Full-prefix 与 local-12 是不同 context distribution，不能只在推理时任意切换后假设效果等价。
4. 早期长视频后段出现网格化/误差累积，需要继续重点观察 depth。
5. Normal 必须使用 flip-YZ 后的数据分布。
6. 当前长视频定量结果主要来自训练数据池，只适合方法间配对比较，不能当作独立泛化结论。
7. 单独 GPU 推理前检查其他进程；不要未经确认杀掉别人的训练。
8. VS Code 无法播放时检查编码是否为 H.264 + yuv420p，而不是直接认为模型输出损坏。

## 12. 推荐接手顺序

1. `git log` 阅读 causal、AR、self-forcing、OPSD 相关 commit。
2. 阅读三份阶段文档和上述核心模块。
3. 在不加载大模型的情况下先跑所有 attention/cache 单测。
4. 用一个 81 帧 bidirectional case 和一个 477 帧 AR case 做 smoke。
5. 再启动 4 卡训练；先用独立 output 跑 1 step，确认 checkpoint、日志、显存和 provenance 都正常。
6. 正式实验必须保留 input manifest、完整命令、Git commit、runtime patch、checkpoint hash 和结果目录。
