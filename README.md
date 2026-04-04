# Agentic RL (Python Tool)

基于 `ms-swift` 的 Agent 训练最小工程，当前聚焦工具能力：
`python`：执行代码求解/验证

项目目标是打通：数据清洗 → SFT/GRPO 训练 → 评测。

## 当前仓库能力
- 已实现 PAL 任务的数据清洗与评测链路
- 已提供 ms-swift GYM 环境插件（`python_tool_env`）示例

## 目录结构

```text
.
├── data/
│   ├── raw/                     # 原始数据
│   └── processed/               # 训练/评测 JSONL
├── answer/                      # 训练记录与评测结果
├── scripts/
│   ├── download_model.py
│   └── prepare_pal_data.py
└── src/agentic_rl/
    ├── data/                    # 数据构建
    ├── eval/                    # PAL 评测
    ├── tools/                   # python工具
    └── train/                   # GRPO 插件（环境/奖励）
```

## 安装

```bash
conda create -n train python=3.11 -y
conda activate train
pip install uv
uv pip install 'ms-swift' --torch-backend=auto
pip install -e .
```
## 运行流程
- python scripts/prepare_pal_data.py准备数据
- SFT训练
- python src/agentic_rl/eval/pal_python_eval.py评估
- GRPO训练
- python src/agentic_rl/eval/pal_python_eval.py评估

# 训练命令
```bash
#SFT
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
--model ./models/Qwen2.5-7B-Instruct \
--tuner_type lora \
--dataset ./data/processed/sft_pal_train.jsonl \
--output_dir ./models/sft_pal_qwen2.5 \
--num_train_epochs 5 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 4 \
--learning_rate 1e-4 \
--lora_rank 16 \
--lora_alpha 32 \
--max_length 2048 \
--torch_dtype float16 \
--logging_steps 10 \
--save_steps 100 \
--save_total_limit 5 \
--warmup_steps 50
```
- NPROC_PER_NODE=8
  - 作用：启动 8 个分布式训练进程，每个进程对应一张 GPU

- CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  - 作用：指定参与训练的 GPU 物理编号列表

- swift sft
  - 作用：启动 ms-swift 的 SFT（Supervised Fine-Tuning，监督微调）训练入口

- --model ./models/Qwen2.5-7B-Instruct
  - 作用：指定基座模型（Foundation Model）路径

- --tuner_type lora
  - 作用：选择参数高效微调（PEFT）方法，LoRA = Low-Rank Adaptation
  - 原理：冻结原模型权重，注入可训练的低秩矩阵 A、B，近似权重更新 ΔW = BA
  - 对比选项：
    - full：全参数微调，需 140GB+ 显存，效果通常更好但成本极高
    - qlora：4-bit 量化 + LoRA，显存需求更低，适合 13B+ 模型
    - adalora：自适应 LoRA，动态调整秩，训练更稳定但稍慢

- --dataset ./data/processed/sft_pal_train.jsonl
  - 作用：SFT 训练数据路径，JSON Lines 格式

- --output_dir ./models/sft_pal_qwen2.5
  - 作用：训练输出目录，保存 checkpoint、日志、配置、可视化

- --num_train_epochs 5
  - 作用：完整遍历训练数据集 5 轮
  - 取值影响：
    - 过小（1-2）：欠拟合，模型未充分学习任务模式
    - 适中（3-5）：通常最佳，平衡拟合与泛化
    - 过大（10+）：过拟合风险，模型死记硬背训练数据，泛化能力下降
  - SFT 特点：
    - 通常 3-5 轮足够，因为 LoRA 参数少，收敛快。数据质量高时，1-2 轮也可取得好效果

- --per_device_train_batch_size 2
  - 作用：每张 GPU 每步（step）处理 2 个样本
  - 取值影响：
    - 增大（4-8）：提升 GPU 利用率，减少空闲等待，但显存线性增长
    - 减小（1）：显存省，但 GPU 计算单元可能未饱和，效率低
  - 显存占用估算（7B 模型 + LoRA）：
    - batch=1：约 12-15 GB
    - batch=2：约 18-22 GB  
    - batch=4：约 30-35 GB
  - 当前设置：batch=2，单卡剩余显存约 15-20GB

- --gradient_accumulation_steps 4
  - 作用：每 4 个前向传播步才执行一次参数更新（梯度累积）
  - 原理：模拟大 batch 训练，等效 batch size = per_device_batch × accumulation × GPU数
  - 计算公式：等效 batch = 2 × 4 × 8 = 64，即每 64 个样本更新一次参数
  - 取值影响：
    - 增大：等效 batch 大，梯度估计更准，训练稳定，但内存累积
    - 减小：更新频繁，噪声大，可能震荡，但适应快
  - 典型设置：
    - 小数据（<10K）：accumulation=1-2，快速迭代
    - 大数据（>100K）：accumulation=4-8，稳定训练
  - 当前设置：4，等效 batch=64，适合 7K+ 数据的 PAL 训练

- --learning_rate 1e-4
  - 作用：LoRA 参数的学习率，控制每次更新的步长
  - 取值影响：
    - 过大（>5e-4）：更新激进，loss 震荡，可能发散，遗忘预训练知识
    - 适中（1e-4 ~ 5e-4）：SFT 常用范围，收敛快且稳定
    - 过小（<1e-5）：收敛极慢，可能陷入局部最优
  - LoRA vs 全参数：
    - LoRA 通常用 1e-4 ~ 1e-3（更高，因为参数少）
    - 全参数 SFT 通常用 1e-5 ~ 2e-5
  - 配合 warmup：前 50 步线性增大 lr，避免早期震荡
  - 当前设置：1e-4，LoRA 标准值，配合预热和余弦衰减

- --lora_rank 16
  - 作用：LoRA 低秩矩阵的秩 r，控制适配器表达能力
  - 原理：秩 r 决定 A（d×r）、B（r×d）矩阵的大小，参数量 = 2 × d × r
  - 取值影响：
    - r=1-4：极低，只能学习简单线性变换，适合简单任务
    - r=8-16：常用范围，平衡效果与效率，适合大多数指令微调
    - r=32-64：表达能力更强，适合复杂任务或长文本，显存增加 2-4 倍
    - r>128：接近全参数微调效果，但失去 LoRA 优势
  - 经验法则：
    - 简单分类/提取：r=8
    - 指令遵循/对话：r=16（当前设置）
    - 代码生成/数学推理：r=32-64
  - 当前设置：r=16，约训练 40M 参数（占原模型 0.5%），适合 PAL 工具调用任务

- --lora_alpha 32
  - 作用：LoRA 缩放系数，实际作用于前向传播的缩放因子为 alpha/rank
  - 计算公式：h = Wx + (alpha/rank) × BAx
  - 取值影响：
    - 通常设为 rank 的 1-2 倍
    - alpha/rank = 2.0（当前）：中等强度更新
    - alpha/rank = 1.0：保守更新，适合保持原能力
    - alpha/rank > 4.0：激进更新，可能破坏预训练知识
  - 与 rank 的关系：
    - 固定 rank，增大 alpha → 等效增大学习率
    - 同时增大 rank 和 alpha → 更多参数 + 更大更新
  - 当前设置：alpha=32，rank=16，缩放因子=2.0，标准配置

- --max_length 2048
  - 作用：训练样本的最大 token 长度，超长部分被截断
  - 取值影响：
    - 增大（4096, 8192, 32768）：可处理长文档、长对话历史。但显存占用平方级增长（注意力矩阵 O(n²)），训练速度显著下降
    - 减小（512, 1024）：省显存，适合短回答任务，但长输入被截断可能丢失信息
  - 模型限制：
    - Qwen2.5-7B 支持 32K 上下文，但训练通常用 2K-4K 即可
    - 预训练模型已具备长文本能力，SFT 只需对齐目标长度分布
  - 当前设置：2048，平衡效率与覆盖，适合 PAL 的中等长度代码/工具调用

- --torch_dtype float16
  - 作用：训练计算精度，FP16（半精度浮点，16-bit）
  - 选项对比：
    - float32（FP32）：32-bit，精度最高，显存占用 ×2，通常不必要
    - float16（FP16）：16-bit，显存省 50%，需梯度缩放防溢出
    - bfloat16（BF16）：16-bit，动态范围与 FP32 相同（指数位更多），更稳定
  - 选择建议：
    - A100/A800/H100：优先 bfloat16（当前环境建议升级）
    - V100/RTX3090：float16（不支持 BF16）
    - 消费级显卡：可能需 FP32 或混合精度
  - 当前设置：float16，兼容性好，但建议改为 bfloat16 提升稳定性

- --logging_steps 10
  - 作用：每 10 个 step 打印一次训练日志到终端和文件
  - 取值影响：
    - 过小（1-5）：日志刷屏，IO 开销，观察过细无意义
    - 适中（10-50）：及时监控训练状态，发现异常（loss NaN、发散）
    - 过大（100+）：可能错过早期问题，适合稳定后的长期训练
  - 当前设置：10，约每 1-2 分钟输出（假设 40s/step），监控及时

- --save_steps 100
  - 作用：每 100 个 step 保存一次模型检查点（checkpoint）
  - 取值影响：
    - 过小（10-50）：保存频繁，磁盘 IO 大，训练中断，适合调试阶段
    - 适中（100-500）：平衡安全与效率，可恢复最近进度
    - 过大（1000+）：省磁盘，但中断后回退多，浪费计算

- --save_total_limit 5
  - 作用：最多保留 5 个 checkpoint，旧的自动删除
  - 取值影响：
    - 过小（1-2）：只能恢复最近状态，无法回滚到早期较好模型
    - 适中（3-5）：保留足够历史，通常足够（SFT 后期模型更好）
    - 过大（10+）：占用磁盘，适合需要对比各阶段效果的研究
  - 当前设置：5，配合 save_steps=100，覆盖 500 步历史

- --warmup_steps 50
  - 作用：学习率预热步数，前 50 步从 0 线性增大到设定值 1e-4
  - 原理：早期模型参数随机，大学习率导致震荡；预热稳定早期训练
  - 计算公式：lr = base_lr × (current_step / warmup_steps)
  - 取值影响：
    - 过小（0-10）：几乎无预热，早期可能不稳定
    - 适中（50-500）：标准设置，占训练总步数 5-10%
    - 过大（>1000）：预热过长，实际训练时间缩短，效率低

```bash
#GRPO
# 多轮工具交互 + 非 vLLM（ms-swift 4.0.2）
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift rlhf \
--rlhf_type grpo \
--model models/sft_pal_qwen2.5_merged \
--dataset data/processed/grpo_pal_train.jsonl \
--external_plugins src/agentic_rl/train/tool_env_plugin_pal.py \
--multi_turn_scheduler python_tool_scheduler \
--max_turns 3 \
--reward_funcs tool_reward \
--reward_weights 1.0 \
--num_generations 4 \
--learning_rate 5e-6 \
--max_length 2048 \
--max_completion_length 1024 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--num_train_epochs 2 \
--tuner_type lora \
--lora_rank 16 \
--lora_alpha 32 \
--torch_dtype float16 \
--use_vllm false \
--beta 0.02 \
--loss_scale default \
--dataloader_num_workers 4 \
--logging_steps 5 \
--save_steps 100 \
--eval_steps 100 \
--save_total_limit 1 \
--load_from_cache_file false \
--temperature 1.0 \
--top_p 1.0 \
--top_k 0 \
--warm_ratio 0.1
--output_dir ./models/grpo_pal_qwen2.5
```

- NPROC_PER_NODE=8
  - 作用：启动 8 个训练进程，与 GPU 数量一一对应

- CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  - 作用：指定参与训练的 GPU 编号列表

- wift rlhf
  - 作用：启动 ms-swift 的 RLHF（基于人类反馈的强化学习）训练入口

- --rlhf_type grpo
  - 作用：选择 GRPO（Group Relative Policy Optimization）算法

- --model models/sft_pal_qwen2.5_merged
  - 作用：指定基座模型路径（已合并 LoRA 的 SFT 模型）
  - 注意：必须是合并后的完整模型，不能是 LoRA 检查点

- --dataset data/processed/grpo_pal_train.jsonl
  - 作用：训练数据路径，JSONL 格式，每条包含 prompt 和 ground_truth

- --external_plugins src/agentic_rl/train/tool_env_plugin.py
  - 作用：加载外部 Python 插件，注册自定义奖励函数和调度器
  - 机制：插件中通过装饰器 `@register_reward` 和 `@register_scheduler` 向 swift 注册

- --multi_turn_scheduler python_tool_scheduler
  - 作用：指定多轮交互调度器名称（由 external_plugins 注册）
  - 功能：控制 agent 与环境的交互轮数，处理工具调用/观察的循环

- --max_turns 3
  - 作用：每条样本最多允许 3 轮 rollout（agent-environment 交互）
  - 取值影响：增大可处理更复杂任务，但显存占用和训练时间线性增加；设为 1 则退化为单轮生成

- --reward_funcs tool_reward
  - 作用：指定奖励函数名称（由 external_plugins 注册为 MyToolReward）
  - 功能：根据工具执行结果计算奖励（如正确性、格式合规性）

- --reward_weights 1.0
  - 作用：该奖励函数的权重系数
  - 取值影响：若多个奖励函数，用此加权；增大该奖励的优化优先级

- --num_generations 8
  - 作用：每个 prompt 采样 8 个候选回答（GRPO 核心参数）
  - 取值影响：必须 ≥2；增大可提升策略梯度估计质量，但显存和计算量线性增加；建议 4-16

- --learning_rate 1e-5
  - 作用：LoRA 参数的学习率
  - 取值影响：过大（>5e-5）导致不稳定、遗忘预训练知识；过小（<1e-6）收敛慢；RLHF 通常比 SFT 低 10 倍

- --max_length 2048
  - 作用：输入上下文（prompt）的最大 token 长度，超长截断
  - 取值影响：增大可处理长历史，但显存占用平方级增长（注意力机制）；需 ≤ 模型支持长度

- --max_completion_length 2048
  - 作用：模型生成回复的最大 token 长度
  - 取值影响：增大允许更长输出，但生成耗时增加；若任务需短回答（如工具调用），可设为 512-1024 加速

- --per_device_train_batch_size 2
  - 作用：每张 GPU 每步的前向传播样本数
  - 取值影响：增大提升 GPU 利用率，但显存占用增加；配合 gradient_accumulation_steps 调整等效 batch

- --gradient_accumulation_steps 5
  - 作用：每 5 步才做一次参数更新，模拟大 batch
  - 取值影响：等效 batch = 8卡 × 2 × 5 = 80；增大可稳定训练，但内存占用累积；减小则更新频繁噪声大

- --num_train_epochs 3
  - 作用：完整遍历数据集 3 轮
  - 取值影响：过大易过拟合奖励函数（reward hacking）；GRPO 通常 1-3 轮即可

- --tuner_type lora
  - 作用：使用 LoRA（低秩适配）微调，只训练少量适配器参数
  - 对比：替代全参数微调，显存省 70%+，适合 7B+ 模型

- --lora_rank 16
  - 作用：LoRA 低秩矩阵的秩 r，控制可学习参数量
  - 取值影响：增大（32/64）提升表达能力但显存增加；简单任务 8-16 足够；公式：参数量 ∝ r

- --lora_alpha 32
  - 作用：LoRA 缩放系数，实际缩放为 alpha/rank
  - 取值影响：通常设为 2×rank；过大导致更新幅度大、训练不稳定；过小则 LoRA 效果弱

- --torch_dtype float16
  - 作用：训练计算精度为 FP16（半精度浮点）
  - 取值影响：比 FP32 省 50% 显存，但需配合梯度缩放防溢出；建议用 bfloat16（动态范围更好）

- --use_vllm false
  - 作用：关闭 vLLM 推理加速，使用 transformers 原生生成
  - 取值影响：false 时生成慢 3-5 倍，但兼容性最好；true 时需 vLLM 版本匹配，加速显著

- --beta 0.05
  - 作用：KL 散度惩罚系数，约束新策略不偏离参考策略太远
  - 取值影响：过大（>0.1）策略保守、奖励低；过小（<0.01）易模式崩溃；GRPO 通常 0.01-0.1

- --loss_scale default
  - 作用：多轮场景下的损失缩放策略
  - 选项：default/last_turn/all_turns；default 通常按轮数平均或加权

- --dataloader_num_workers 0
  - 作用：数据加载子进程数为 0（主进程加载）
  - 取值影响：增大（4-8）可预加载数据，但多进程开销大；SSD 存储时 2-4 较好

- --logging_steps 5
  - 作用：每 5 步打印一次训练日志（loss、reward、速度等）
  - 取值影响：过小日志刷屏，过大监控滞后；建议 5-20

- --save_steps 100
  - 作用：每 100 步保存一次 checkpoint
  - 取值影响：过小保存频繁、磁盘 IO 大；过大则中断后回退多；建议 50-500

- --eval_steps 100
  - 作用：每 100 步在验证集上评估
  - 取值影响：需配合 val_dataset 使用；无验证集时不生效

- --save_total_limit 5
  - 作用：最多保留 5 个 checkpoint，旧自动删
  - 取值影响：控制磁盘空间；建议 3-10，根据训练总步数调整

- --load_from_cache_file true
  - 作用：复用 datasets 库生成的预处理缓存
  - 取值影响：首次预处理慢，后续启动快；数据变更时需设为 false 或删缓存

- --output_dir ./models/grpo_pal_qwen2.5
  - 作用：训练输出目录，保存 checkpoint、日志、配置
  - 影响：自动创建子目录如 v0-20260330-172922 避免覆盖


```bash
# 前台最小烟测（1 卡 1 step，确保流程可运行）
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=0 \
swift rlhf \
--rlhf_type grpo \
--model models/Qwen2.5-0.5B-Instruct \
--dataset data/processed/grpo_pal_train.jsonl \
--external_plugins src/agentic_rl/train/tool_env_plugin.py \
--multi_turn_scheduler python_tool_scheduler \
--max_turns 3 \
--reward_funcs tool_reward \
--reward_weights 1.0 \
--num_generations 2 \
--steps_per_generation 2 \
--learning_rate 1e-6 \
--max_length 1024 \
--max_completion_length 96 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--max_steps 1 \
--num_train_epochs 1 \
--tuner_type lora \
--lora_rank 1 \
--lora_alpha 2 \
--torch_dtype float16 \
--use_vllm false \
--beta 0.001 \
--loss_scale default \
--dataloader_num_workers 0 \
--logging_steps 1 \
--save_steps 1000 \
--eval_steps 1000 \
--save_total_limit 2 \
--load_from_cache_file true \
--output_dir ./models/grpo_pal_qwen2.5_smoke
```

GRPO 参数约束（ms-swift 4.0.2）：
- `--use_vllm false` 时不要再传 `--vllm_mode`。
- `--num_generations >= 2`。
- 生成 batch 需要整除 `num_generations`：
  `generation_batch_size = world_size * per_device_train_batch_size * steps_per_generation`。

# 合并检查点与基座模型
```bash
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model ./models/Qwen2.5-7B-Instruct \
    --adapters ./models/sft_pal_qwen2.5/checkpoint-XXX \
    --merge_lora true \
    --output_dir ./models/sft_pal_qwen2.5_merged \
    --safe_serialization true \
    --torch_dtype bfloat16
```

# 推理命令
```bash
#完整模型推理
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift infer \
--model ./models/Qwen2.5-7B-Instruct \
--stream true \
--infer_backend transformers \
--temperature 0 \
--max_new_tokens 2048
```
```bash
#检查点推理
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift infer \
--adapters ./models/sft_pal_qwen2.5/v1-20260322-224857/checkpoint-2340 \
--infer_backend transformers \
--stream true \
--temperature 0 \
--max_new_tokens 2048
```
# 部署命令：
```bash
#如果是完整模型，使用--model替代--adapters指定训练的checkpoint目录。
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift deploy \
    --adapters output/vx-xxx/checkpoint-xxx \
    --infer_backend transformers \
    --temperature 0 \
    --max_new_tokens 2048 \
    --served_model_name model-name
```