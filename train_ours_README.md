# Ours模型训练脚本使用指南

## 概述

`train_ours.sh` 是一个统一的训练脚本，将所有可调参数集中在一个地方，便于超参搜索和实验管理。

## 快速开始

1. **编辑参数**：打开 `train_ours.sh`，修改脚本顶部的参数
2. **运行训练**：`bash train_ours.sh`

## 参数说明

### 1. 训练超参数 (Training Hyperparameters)

| 参数 | 默认值 | 说明 | 建议范围 |
|------|--------|------|----------|
| `LEARNING_RATE` | 0.001 | 学习率 | 0.0001 - 0.01 |
| `WEIGHT_DECAY` | 1e-5 | 权重衰减 | 1e-6 - 1e-3 |
| `EPOCHS` | 20 | 训练轮数 | 10 - 50 |
| `T_MAX` | 500 | 余弦退火T_max | 100 - 1000 |
| `GRAD_CLIP_NORM` | 1.0 | 梯度裁剪阈值 | 0.5 - 2.0 (0=不裁剪) |

### 2. 模型架构参数 (Model Architecture Parameters)

这些参数对应 `ours.py` 中 `Model.__init__` 的参数：

| 参数 | 默认值 | 说明 | 建议范围 |
|------|--------|------|----------|
| `IMG_SIZE_H` | 121 | 图像高度(纬度) | 固定 |
| `IMG_SIZE_W` | 240 | 图像宽度(经度) | 固定 |
| `EMBED_DIM` | 768 | 嵌入维度 | 256, 384, 512, 768 |
| `DEPTH` | 8 | Transformer层数 | 4, 6, 8, 12 |
| `DECODER_DEPTH` | 2 | 解码器层数 | 1, 2, 3 |
| `NUM_HEADS` | 16 | 注意力头数 | 必须能被embed_dim整除 |
| `MLP_RATIO` | 4.0 | MLP扩展比例 | 2.0, 4.0, 8.0 |
| `DROP_PATH` | 0.1 | DropPath率 | 0.0 - 0.3 |
| `DROP_RATE` | 0.1 | Dropout率 | 0.0 - 0.2 |

### 3. 数据参数 (Data Parameters)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BATCH_SIZE` | 32 | 批次大小 |
| `NUM_WORKERS` | 16 | 数据加载器工作进程数 |
| `INPUT_SIZE` | 63 | 输入变量数 |
| `OUTPUT_SIZE` | 63 | 输出变量数 |
| `PRED_LEN` | 2 | 预测时间步数(周数) |

## 使用示例

### 示例1：调整学习率

```bash
# 在 train_ours.sh 中修改
LEARNING_RATE=0.0005  # 降低学习率

# 运行
bash train_ours.sh
```

### 示例2：调整模型容量

```bash
# 在 train_ours.sh 中修改
EMBED_DIM=512         # 减小嵌入维度
DEPTH=6              # 减少层数
NUM_HEADS=8          # 相应调整注意力头数

# 运行
bash train_ours.sh
```

### 示例3：超参搜索

```bash
# 创建多个配置脚本
cp train_ours.sh train_ours_lr001.sh
cp train_ours.sh train_ours_lr0005.sh

# 在不同脚本中设置不同参数
# train_ours_lr001.sh: LEARNING_RATE=0.001
# train_ours_lr0005.sh: LEARNING_RATE=0.0005

# 并行运行
bash train_ours_lr001.sh &
bash train_ours_lr0005.sh &
```

## 超参搜索建议

### 阶段1：学习率搜索
1. 固定其他参数
2. 尝试不同的学习率：`[0.0001, 0.0005, 0.001, 0.005]`
3. 选择验证损失最低的学习率

### 阶段2：正则化搜索
1. 固定学习率（使用阶段1的最佳值）
2. 调整 `WEIGHT_DECAY`: `[1e-6, 1e-5, 1e-4]`
3. 调整 `DROP_PATH` 和 `DROP_RATE`: `[0.0, 0.1, 0.2]`

### 阶段3：模型容量搜索
1. 固定训练超参数
2. 调整 `EMBED_DIM`: `[256, 384, 512, 768]`
3. 调整 `DEPTH`: `[4, 6, 8, 12]`
4. 相应调整 `NUM_HEADS`（必须能被embed_dim整除）

## 输出文件

- **训练日志**: `./logs/ours_YYYYMMDD_HHMMSS.log`
- **Checkpoint**: `./lightning_logs/version_X/checkpoints/`
- **配置副本**: `./lightning_logs/version_X/config.yaml` (自动保存)
- **评估结果**: `./results/ours/ours_metrics_*.csv`

## 注意事项

1. **参数约束**：
   - `NUM_HEADS` 必须能被 `EMBED_DIM` 整除
   - `EMBED_DIM` 建议使用 2 的幂次（256, 512, 768等）

2. **并行训练**：
   - 每个训练任务会自动保存配置到对应的checkpoint目录
   - 评估时会自动从checkpoint目录读取配置，不会混乱

3. **GPU数量**：
   - 可通过环境变量设置：`NP=4 bash train_ours.sh`
   - 或修改脚本中的 `NP` 变量

4. **后台运行**：
   - 默认后台运行，日志保存在 `./logs/` 目录
   - 设置 `BACKGROUND=false` 可前台运行

## 参考模板

完整的参数模板和说明请参考：
- `CIRT/configs/ours_hyperparams_template.yaml` - 参数模板和详细说明

