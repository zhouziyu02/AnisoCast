# Ours模型训练参数快速参考

## 📝 使用方法
1. 编辑 `train_ours.sh` 顶部的参数
2. 运行 `bash train_ours.sh`

## ⚙️ 所有可调参数一览

### 训练超参数
```bash
LEARNING_RATE=0.001      # 学习率 [0.0001-0.01]
WEIGHT_DECAY=1e-5        # 权重衰减 [1e-6-1e-3]
EPOCHS=20                # 训练轮数 [10-50]
T_MAX=500                # 余弦退火T_max [100-1000]
GRAD_CLIP_NORM=1.0       # 梯度裁剪 [0.5-2.0, 0=不裁剪]
```

### 模型架构参数
```bash
EMBED_DIM=768            # 嵌入维度 [256,384,512,768]
DEPTH=8                  # Transformer层数 [4,6,8,12]
DECODER_DEPTH=2          # 解码器层数 [1,2,3]
NUM_HEADS=16             # 注意力头数 [必须被embed_dim整除]
MLP_RATIO=4.0            # MLP扩展比例 [2.0,4.0,8.0]
DROP_PATH=0.1            # DropPath率 [0.0-0.3]
DROP_RATE=0.1            # Dropout率 [0.0-0.2]
```

### 数据参数
```bash
BATCH_SIZE=32            # 批次大小
NUM_WORKERS=16           # 数据加载器进程数
NP=8                     # GPU数量
```

## 🔍 超参搜索顺序建议

1. **学习率** → `LEARNING_RATE` [0.0001, 0.0005, 0.001, 0.005]
2. **正则化** → `WEIGHT_DECAY`, `DROP_PATH`, `DROP_RATE`
3. **模型容量** → `EMBED_DIM`, `DEPTH`, `NUM_HEADS`

## 📊 输出位置

- 日志: `./logs/ours_*.log`
- Checkpoint: `./lightning_logs/version_X/checkpoints/`
- 配置: `./lightning_logs/version_X/config.yaml` (自动保存)
- 结果: `./results/ours/ours_metrics_*.csv`

## 💡 快速示例

```bash
# 示例1: 降低学习率
LEARNING_RATE=0.0005 bash train_ours.sh

# 示例2: 减小模型容量
EMBED_DIM=512 DEPTH=6 NUM_HEADS=8 bash train_ours.sh

# 示例3: 使用4个GPU
NP=4 bash train_ours.sh
```

