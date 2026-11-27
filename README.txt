Dense / Sparse 研究代码审阅包
2026-11-27

建议从 PAPER_CODE_MAP.txt 开始阅读；运行方式见 RUNNING.txt。
这份包代替上一份 core 代码节选，包含完整底层源码、历史各阶段 trainer、
当前 Dense 像素 decoder 和新 JTA 多人双向训练/验证/推理流程。

目录
====
core/                 可读核心：任务 embedding 门控、Dense/Sparse loss、decode、Wan 接线。
tools/                新 JTA 数据缓存、完整双向训练、推理、测试与原阶段入口索引。
dense_pixel/          当前 depth v12 / matte-normal v13 实验的完整脚本快照。
vendor/DiffSynth-Studio/
                      原项目完整 DiT/pipeline、训练/推理/阶段转换源码。
vendor/Wan2.2/        官方 Wan2.2 源码，含输入/输出 VAE 实现与上游 LICENSE。
configs/              JTA 新实验默认超参和损失权重。
evidence/             交接文档、原 checkpoint config、最终绑定源码、数据审计、测试报告。

实际完成情况
============
1. 现有 Dense：历史已完成 bidirectional -> causal -> AR teacher forcing ->
   long self-forcing -> bounded OPSD。五阶段训练源文件都在本包中。
   历史训练源码保留原路径和来源检查，供审阅和恢复工程使用；旧训练数据/缓存、
   中间 checkpoint、原 paired optimizer/RNG 没有迁移，不能仅凭本代码包立即
   从零逐步精确重现历史结果。具体缺项见 RUNNING.txt。

2. 当前 Dense 输出头：depth/matte/normal 已训练并测试过，替换输出 VAE；
   输入 RGB VAE 和 latent AR history 保留。其余三种 Dense 输出仍使用官方 VAE。

3. 新 JTA Sparse：22 关节、多个人、2D/3D prompt、独立的新 LoRA/router/head。
   本包实现数据 -> RGB latent cache -> 双向训练 -> 官方序列划分验证 ->
   保存/断点恢复 -> 无 GT 推理九块 -> 原图坐标与跨块重复抑制。
   训练更新/数值测试的实际运行范围以 evidence/ 中本次报告为准。
   单次/少量调试更新不是完成训练，更不能作为精度或泛化结果。
   新 JTA 尚未完成 causal/AR/self-forcing/OPSD 多人适配与实验。

4. 历史单人 Sparse：旧 24 关节模型的五阶段原始源码也完整保留，作为历史基线。
   它与新的 JTA 多人头不是同一模型，不加载到新 JTA 中，也不作为新实验结果。

老师审阅时优先看 PAPER_CODE_MAP.txt 中列出的文件，无需逐个看第三方代码。
代码包不携带几十 GB 基座权重、训练视频或隐含数据下载，不自动启动长训练。
上游组件保留各自许可证；这份整理包不改变上游代码归属。
