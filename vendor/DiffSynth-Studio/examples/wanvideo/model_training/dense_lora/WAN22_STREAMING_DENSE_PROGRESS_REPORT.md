# Wan2.2 长视频流式稠密预测阶段汇报

## 1. 研究目标

将已有的 Wan2.2 多任务稠密预测模型从离线双向推理改造成可流式运行的模型，使其能够在长视频上逐块输出 depth、normal 等结果，同时减少长序列中的闪烁和误差累积。

初始模型为：

`/pfs/mayuema/dense/wan/outputs/wan22_dense4_hyperlora_lingbot_stage2_20k/step-20000.safetensors`

所有后续训练均从已有 checkpoint 继续，没有重新初始化原有 LoRA 和任务路由参数。官方 Wan2.2 代码保持不变，新增实现均放在独立文件中并由 Git 记录。

## 2. 已完成工作

### 2.1 Causal attention 与流式推理

- 将 Wan2.2 DiT 的时序 attention 改为 causal attention，使当前帧只能访问当前及历史信息。
- 实现逐层 KV cache，用于逐块 streaming inference。
- 对齐了 full causal forward 与 KV-cache streaming forward 的位置编码和缓存行为。

这一步得到的是 causal 模型，但不同输出帧之间还没有显式的自回归依赖。

### 2.2 Chunkwise AR

- 在 causal 模型上加入目标模态历史，当前 chunk 会显式依赖前一个 depth/normal chunk。
- 每个 chunk 为 3 个 latent frame，约对应 12 个原始视频帧。
- 先完成 teacher forcing 训练，再改为 self-forcing：训练时将模型自己预测的上一块回灌，减小训练与推理之间的 exposure bias。

因此现在的模型同时具备：

1. RGB 的因果历史；
2. 目标模态的自回归历史；
3. 可复用的逐层 KV cache。

### 2.3 长视频训练与数据处理

- 将训练长度扩展到 477 个原始视频帧，对应 120 个 latent frame、40 个 AR chunk。
- 接入长合成数据的 RGB、depth、normal、matte GT，以及 LingBot depth/normal 数据。
- 统一 normal 坐标系为 `[x, -y, -z]`，解决了早期 normal 色调和方向分布不一致的问题。
- 完成长视频 free-running AR 推理和可视化流程。

当前较干净的长视频 AR/self-forcing checkpoint 为：

`/pfs/mayuema/dense/wan/outputs/wan22_long477_ar_self_forcing_flip_yz_v3/step-3200.safetensors`

### 2.4 OPSD post-training

参考 OPSD-V 的 Wan2.1 实现，将其 prefix distillation 方法迁移到 Wan2.2：

- student 使用自己生成的历史和 KV cache；
- teacher 使用 EMA 权重，并提供延迟一块的 GT prefix 表征；
- 保留原任务损失，同时加入 prefix distillation loss；
- 训练和推理均保持 `t=0`、单次 forward 的稠密预测设定。

目前完成了 full-prefix OPSD 训练，并正在训练更接近 OPSD 原配置的 bounded-prefix 版本。

## 3. 当前结果

在 20 条、每条 477 帧的合成长视频上，full-prefix OPSD 相对未加 OPSD 的 AR 模型：

- 总体 latent MSE 下降 **2.94%**；
- 视频后 1/4 的 latent MSE 下降 **3.36%**；
- temporal-delta MSE 下降 **1.13%**，97.5% 的配对样本更好；
- normal 像素 MSE 下降 **3.38%**；
- normal 平均角误差从 **6.519°** 降至 **6.357°**。

定性上，OPSD 对 normal 和长视频时间稳定性的改善较一致，没有观察到后段误差加速累积。Depth 的平均指标有所改善，但逐样本收益还不稳定。

需要说明：目前这 20 条样本来自训练数据池，因此这些数字用于比较方法差异，不能作为独立测试集上的泛化结论。

已有 6 条真实长视频的无 OPSD/有 OPSD 对比结果：

`/pfs/mayuema/dense/wan/outputs/ar_long477_00cd3222/comparison_no_opsd_vs_opsd_step1200`

## 4. 正在进行的实验

正在从干净的 AR/self-forcing step-3200 checkpoint 训练 OPSD bounded-prefix 版本：

- local attention：12 个 latent frame；
- attention sink：3 个 latent frame；
- 目标训练长度：3500 step；
- 预计 2026-09-17 上午完成。

输出目录：

`/pfs/mayuema/dense/wan/outputs/wan22_long477_opsd_prefix_t0_local12_3500`

## 5. 下一步

1. 在相同长视频上比较无 OPSD、full-prefix OPSD 和 bounded-prefix OPSD。
2. 建立独立的长视频 validation split，验证改进是否能泛化到未参与训练的视频。
3. 根据效果、显存和速度，确定最终使用 full-prefix 还是 bounded-prefix。
4. 继续重点观察 depth 的稳定性，以及极长视频后半段的网格化和误差累积问题。
