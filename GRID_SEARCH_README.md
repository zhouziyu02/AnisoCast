# 超参数网格搜索使用说明

## 概述

`grid_search.sh` 脚本用于对 OST 模型进行超参数网格搜索，自动并发执行多个训练任务并收集结果。

## 搜索参数

脚本会搜索以下4个超参数：

- **learning_rate**: `[5e-4, 1e-3, 2e-3, 5e-3, 1e-2]` (5个候选值)
- **embed_dim**: `[192, 256, 384, 512, 768]` (5个候选值)
- **depth**: `[6, 8, 10]` (3个候选值)
- **decoder_depth**: `[1, 2, 3]` (3个候选值)

**总任务数**: 5 × 5 × 3 × 3 = **225 个任务**

## 使用方法

### 基本用法

```bash
# 使用默认参数（4卡GPU，最大并发5个进程）
bash grid_search.sh

# 指定GPU数量和最大并发数
bash grid_search.sh --np 8 --max-concurrent 5

# 指定训练轮数
bash grid_search.sh --epochs 30 --batch-size 64
```

### 命令行参数

- `--model-name <str>`: 模型名称（默认: ost）
- `--np <int>`: GPU数量（默认: 4）
- `--epochs <int>`: 训练轮数（默认: 20）
- `--batch-size <int>`: 批次大小（默认: 32）
- `--max-concurrent <int>`: 最大并发进程数（默认: 5）

### 示例

```bash
# 使用8卡GPU，最大并发5个任务
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash grid_search.sh --np 8 --max-concurrent 5

# 使用4卡GPU，训练30轮
bash grid_search.sh --np 4 --epochs 30
```

## 输出结构

所有结果保存在 `./grid/` 目录下，每个超参数组合有独立的文件夹：

```
grid/
├── grid_search.log              # 主日志文件
├── progress.txt                 # 进度文件
├── completed.txt                # 已完成任务列表
├── failed.txt                   # 失败任务列表
├── lr5e_4_emb192_dep6_dec1/     # 第一个任务目录
│   ├── training.log             # 训练日志
│   ├── task.pid                 # 任务信息
│   ├── config.yaml              # 配置文件
│   ├── metrics.csv              # 评估结果CSV
│   └── checkpoints/             # checkpoint文件
├── lr5e_4_emb192_dep6_dec2/     # 第二个任务目录
│   └── ...
└── ...
```

## 目录命名规则

每个任务目录的命名格式为：
```
lr{learning_rate}_emb{embed_dim}_dep{depth}_dec{decoder_depth}
```

例如：
- `lr5e_4_emb192_dep6_dec1` 表示 `lr=5e-4, embed_dim=192, depth=6, decoder_depth=1`
- `lr1e_3_emb256_dep8_dec2` 表示 `lr=1e-3, embed_dim=256, depth=8, decoder_depth=2`

## 监控进度

### 查看主日志

```bash
tail -f grid/grid_search.log
```

### 查看进度文件

```bash
cat grid/progress.txt
```

### 查看已完成/失败任务

```bash
# 已完成任务数
wc -l grid/completed.txt

# 失败任务数
wc -l grid/failed.txt
```

### 查看特定任务的日志

```bash
tail -f grid/lr1e_3_emb256_dep8_dec2/training.log
```

## 结果分析

### 查看所有任务的评估结果

```bash
# 查找所有CSV文件
find grid -name "metrics.csv" -type f

# 汇总所有结果到一个文件
find grid -name "metrics.csv" -type f -exec sh -c 'echo "=== {} ==="; cat {}' \; > grid/all_results.txt
```

### 提取最佳结果

```bash
# 查找val_loss最小的配置（需要从CSV中提取）
# 可以编写Python脚本分析所有metrics.csv文件
```

## 注意事项

1. **磁盘空间**: 225个任务会产生大量日志和checkpoint文件，确保有足够的磁盘空间
2. **GPU资源**: 确保有足够的GPU资源，建议使用 `CUDA_VISIBLE_DEVICES` 指定可用GPU
3. **并发控制**: `--max-concurrent` 不要设置过大，避免资源竞争
4. **训练时间**: 每个任务训练20轮，总时间取决于GPU数量和训练速度
5. **中断恢复**: 脚本不支持中断恢复，如果中断需要重新运行（已完成的任务不会重复执行）

## 自定义搜索空间

如需修改搜索的超参数和候选值，编辑 `grid_search.sh` 文件中的以下部分：

```bash
learning_rates=(5e-4 1e-3 2e-3 5e-3 1e-2)
embed_dims=(192 256 384 512 768)
depths=(6 8 10 12 14)
decoder_depths=(1 2 3 4 5)
```

## 故障排查

### 任务失败

1. 查看任务日志：`cat grid/{setting}/training.log`
2. 检查GPU资源是否充足
3. 检查磁盘空间是否足够

### 找不到结果文件

1. 检查训练是否成功完成
2. 查看 `grid/{setting}/training.log` 确认评估是否执行
3. 手动检查 `results/{model_name}/` 目录

### 并发冲突

如果出现端口冲突或其他并发问题，减少 `--max-concurrent` 的值。

