Dense 现有实现 + JTA 新 Sparse 多人实现
版本：2026-11-27

这份代码交付的范围
==================
Dense：保留现有六任务 LoRA/router、latent loss、OPSD 前向，并整理已训练的
depth/matte/normal 直接像素输出头。其余三个 Dense 任务仍使用官方输出 VAE。
Sparse：新写的 JTA 22 关节多人双向版本，包含 task router、person-query decoder、
整段 Hungarian matching、masked loss 与接入 Wan 的工厂函数；不加载旧 Sparse 权重。
主干仍初始化自官方 Wan2.2 TI2V 5B，LoRA/router/多人头是新参数。

这是核心实现和单步接入示例，不是已经完成训练的新 Sparse 模型。
没有启动新训练；没有将新 Sparse 宣称为已验证效果的模型。
本次验证范围见 CHECKS.json：CPU 数值/梯度、真实 JTA 标签接口和原 Dense 头严格加载。
没有重新执行完整 Wan 5B forward/backward、DDP 训练或跨裁块/跨片段跟踪。

建议阅读顺序
============
1. task_gating.py：任务 embedding 到 LoRA rank 门控。
2. dense_loss.py / sparse_loss.py：旧 Dense 与新 Sparse 各自的目标函数。
3. dense_decode.py / sparse_decode.py：直接像素输出与多人坐标输出。
4. dense_native.py / sparse_model.py：与相应 Wan 前向接线。
5. example_usage.py：真实输入预测或新 Sparse 单次更新示例。
6. jta_tiles.py：已验证的无 resize 裁块、22 关节标注、mask 和相机内参转换。
7. native_reference/：用于溯源的现有源码；部分原训练脚本保留旧 /pfs 路径，
   不要把这些历史脚本当作当前机器的一键启动器。

一、LoRA 的不同任务 embedding 门控
=================================
每个投影层：
  y = W_frozen(x) + scale * B( g_task * A(dropout(x)) )
  e_task = frozen_UMT5(task_instruction)[0,0]        # 4096 维，不是文本均值
  logits = layernets(MLP(LayerNorm(e_task)))       # [B,300,32]
  g_task = 1 + 0.5 * tanh(logits)                  # [0.5,1.5]

现有 rank=32，300 个 LoRA 注入点；target modules=q,k,v,o,ffn.0,ffn.2。
hypernet hidden=512；layernets 的 weight/bias 初始为零，所以 gate 初始恰好为 1。
LoRA A 随机、B 为零，初始 LoRA delta=0。不是每个任务一套 LoRA，也不是专家 softmax。
Dense 和 Sparse 各有自己独立的 LoRA/router；它们不共享适配器。
任务 embedding 是持久化 buffer，文本编码器冻结；hypernet/LoRA/head 可训练。

Dense ID：depth=0, normal=1, matte=2, raymap=3, albedo=4, optical_flow=5。
Dense router 使用六个通用指令特征；DiT cross-attention 使用完整 prompt context，
其中 matte 可带 person/bear/... 主体。task_id 与 prompt_id 不能混用或重排。
Dense checkpoint 的 condition_contexts 和 task_feature_table 都必须加载。

Sparse ID：0='output: 2D keypoints'，1='output: 3D keypoints'。
两个条件分别 forward。3D 条件另外输出辅助 2D 坐标，用于人物匹配和图像定位。
不要假设两次不同条件 forward 的 slot 0 必然对应同一个人。

gradient checkpointing 注意：router 的 active gates 必须保留到 backward 完成。
每次先 forward -> loss.backward()，再切换任务；不能两个任务 forward 后统一 backward，
否则重算可能读到另一任务的门控。training_step 已按此顺序组织。

二、Loss
========
现有 Dense 主干：
  L = masked_MSE(-velocity, target_latent)
      + 0.001 * mean((g-1)^2)
      + 0.01 * dispersion
固定 t=0。dispersion 来自第 9 个 Transformer block 的完整时空特征：
同 prompt 配对、temperature=1、包含对角线，多卡使用可微 all_gather。
dispersion 可以为负；不能仅凭总 loss 为负判断效果提升。

现有 Dense OPSD：另加系数 1 的 EMA teacher latent MSE。
40 个监督块和 33 个蒸馏块分别平均；teacher decay=0.9999，从 latent offset21 开始。
teacher/student KV 分开，teacher 延迟一块重放干净历史。具体前向保留在
native_reference/wan22_dense6_long477_opsd.py，不能用普通双向前向代替。

现有直接像素头训练：
  L_pixel = pixel_MSE + 0.2*temporal_Charbonnier + 0.2*VGG_LPIPS
temporal_Charbonnier = mean(sqrt((Δprediction-Δtarget)^2 + 1e-6)) - 0.001。
通常抽 4 帧算 LPIPS；scalar 复制成 3 通道；normal 用 3 个 signed 通道。
冻结 Dense DiT，仅训练输出头；它不是上述主干 loss 的直接替代或附加项。
原训练还包含时间 context 和空间 halo 裁剪；loss 入口只接收最终监督区域。
normal 数据沿用已有 flipYZ 一次；预测/训练 loss 内不重复翻转，也不做强制单位化。

新 Sparse（初始配置，尚未调优）：
  L = 5*masked_MSE_2D + 1*masked_MSE_3D [仅3D任务]
      + 1*presence_BCE
      + 0.2*(in_crop_BCE + visible_BCE)
      + 0.001*router_identity
2D normalize by [832,480]；3D 为相机坐标、mid-hip root-relative、单位米。
不继承旧数据的 actor-height 归一化，不使用 Dense normal 的 flipYZ。
每条 GT 轨迹与一个预测 slot 在整段 clip 上一次性匹配；Hungarian cost 用
5*2D_L1 + 1*3D_L1[仅3D任务] + 2*presence_BCE。
matching 在 no_grad 内进行，匹配后的各项 loss 对预测可微。
可见关节权重 1，遮挡但在裁块内的关节初始权重 0.25。
出界、padding 和不可用 3D 不进入对应坐标 loss。
每帧至少一个可见关节才定义为可检测的人；完全不可见的人不要求模型凭空预测。
unmatched slot 的 presence 权重 0.1；空场景仍训练“不存在”。
人数超过 slot 上限会报错，不截断、不丢标签。本次原始 train/val 最大轨迹数
分别为 70/111，因此初版配置 128 slots；test 和任意真实视频的上限尚未验证。
新 Sparse 不加 Dense dispersion，不带 OPSD distillation。

三、Decode 与张量格式
====================
Dense 现有两条输出路径：
  RGB -> official input VAE -> Dense6 OPSD -> predicted target latent -> official output VAE
  RGB -> official input VAE -> Dense6 OPSD 内部特征 -> trained direct pixel head
第二条仍保留 target latent readout 用于 AR history；“无输出 VAE”不等于整个网络无 VAE。
直接头取 dit.head.head 的 PRE-hook：已经过原 head 的 norm/modulation。
  [B,21,15,26,3072] -> 1+4n 时间展开+phase embedding
  -> SiLU -> causal3x1x1 conv
  -> 5 次 spatial x2 转置卷积 + causal3x3x3 conv + SiLU
  -> FP32 output conv -> [B,81,C,480,832]
widths=1024,512,256,128,64；depth/matte C=1，normal C=3。
state_dict 名称/形状与现有三份 head_30000.pt 一致，并已 strict load 验证。
每视频/任务必须独立缓存 state。输入不是 48 通道 VAE latent。
已训练头配套 Dense6 OPSD5250 特征；不保证直接挂到 bidirectional120k 仍有同等效果。

Sparse 新路径：
  RGB crop -> official input VAE -> bidirectional DiT(t=0)
  -> pre-head tokens [B,8190,3072] -> multi-person coordinate decoder
  -> xy [B,81,128,22,2] / xyz [B,81,128,22,3] / presence [B,81,128]
Sparse 没有输出 VAE，也不需要 Dense 的上采样像素头。
person queries 在每个 latent frame 内做空间 self/cross attention，沿时间做双向
attention；位置编码同时使用 crop xy 和相机内参计算的视线方向。
输出 3D 强制以左右 hip 中点为零；不是世界坐标或绝对相机深度。
decode_people 用 presence 阈值筛选人数，将 normalized xy 乘尺寸再加 crop origin。
跨裁块去重和跨片段轨迹关联未在此包实现；返回的 slot ID 仅对当前 forward 有效。

四、运行
========
服务器目录：
  /xs-train-nas/LingXiTeam/workspace/msh/my_code/video_dense/experiments/dense_jta_code_20261127
项目环境入口：
  bash /xs-train-nas/LingXiTeam/workspace/msh/my_code/video_dense/tools/wan22_python.sh
依赖现有环境中的 torch/numpy/scipy；整主干示例另依赖已有 DiffSynth、PEFT、safetensors
和官方 Wan 源码/权重。不修改共享环境，不自动下载或安装包。

在代码目录运行核心 CPU 检查：
  CUDA_VISIBLE_DEVICES= bash ../../tools/wan22_python.sh test_core.py

查看真实输入示例参数：
  bash ../../tools/wan22_python.sh example_usage.py --help

RGB 输入是 uint8 [81,480,832,3] 的 .npy，不做 resize。
本次准备好的 RGB/target pairs 位于相邻实验：
  ../jta_sparse_bidir_20261127/native_tile_pilot_v1/
例如 train_seq_85_f0301_t0_rgb.npy 与 train_seq_85_f0301_t0_targets.npz。

Dense 示例（选择当前空卡映射后使用 cuda:0）：
  bash ../../tools/wan22_python.sh example_usage.py dense-predict \
    --root /xs-train-nas/LingXiTeam/workspace/msh/my_code/video_dense \
    --rgb ../jta_sparse_bidir_20261127/native_tile_pilot_v1/train_seq_85_f0301_t0_rgb.npy \
    --task depth --decoder direct --output ./depth_prediction.pt
--decoder vae 可切回官方输出 VAE。输出保留 signed 训练值域，不做可视化拉伸。

Sparse 调试单步示例（该命令会新建模型、执行一次更新；本次没有执行）：
  bash ../../tools/wan22_python.sh example_usage.py sparse-one-step \
    --root /xs-train-nas/LingXiTeam/workspace/msh/my_code/video_dense \
    --rgb ../jta_sparse_bidir_20261127/native_tile_pilot_v1/train_seq_85_f0301_t0_rgb.npy \
    --targets ../jta_sparse_bidir_20261127/native_tile_pilot_v1/train_seq_85_f0301_t0_targets.npz \
    --task 3d --output ./jta_debug_step1.pt
此文件只有一次更新和适配参数，没有 optimizer/RNG，不是可精确 resume 的 checkpoint。
全量训练尚需采样循环、验证指标、断点状态和 DDP 调度；DDP 使用
find_unused_parameters=True（2D task 不执行 3D readout）。

五、溯源
========
交接文档：
  /Users/wjm/Documents/dense_sparse_handoff_20261122/DENSE_SPARSE_PROJECT_HANDOFF_20261122.md
SOURCE_MANIFEST.json：原生源码来源及 SHA256。
PACKAGE_MANIFEST.json：本次代码包文件 SHA256。
Dense loader 校验现有 checkpoint SHA256 与对应 contract 中的原生源码 hash，
并核对完整适配器/router keys，不用 strict=False 掩盖漏加载。
新 Sparse 不兼容旧 24 关节单人 checkpoint，代码中也没有加载它的入口。
