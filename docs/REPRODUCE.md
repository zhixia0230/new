# 复现步骤与资源边界

## 1. 环境和安装

本次整理验证环境为 Linux、Python 3.13.12、PyTorch 2.11.0+cu128、CUDA 12.8、NVIDIA RTX 5090（32 GiB）。依赖为 torch、numpy、Pillow、scipy、PyYAML；测试使用 pytest。其他软硬件组合未逐一验证，不保证逐位一致。

在仓库根目录执行：

```bash
python -m pip install -e '.[dev]'
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q -p no:cacheprovider
```

## 2. 必须另行准备的资源

数据应包含：

```text
RadioMapSeer/
  png/buildings_complete/<scene>.png
  png/antennas/<scene>_<tx>.png
  gain/DPM/<scene>_<tx>.png
```

图像必须为 256×256 的 L 模式 uint8 PNG；每份 TX 图有且仅有一个非零像素。坐标按 tensor[y,x] 读取，存储为 (x,y)。目标为原 PNG/255，不做阈值化或数据增强。

`metadata/splits.json` 提供既有 500/100/100 场景划分，该划分不是官方划分；测试集编号只是用于审计场景隔离，本入口不自动评估 test。`metadata/selection.json` 提供固定 4000 张训练图和 400 张开发验证图的顺序。

必须取得**已有强普通父模型 EMA**：

```text
parent EMA SHA256:
407aefc3944facdc0e3e4a8e92b36c156bbd52abb3ef4b2246c041f1ec619842
selected parent phase checkpoint step: 13750
EMA cap: 0.99
development RMSE: 0.01971660571769083
```

父模型文件中的 13750 是其增强训练阶段记录，不代表从头训练总步数。本核心包不包含父模型从头训练流水线或权重；需要项目所有者另行提供合法可分发的父模型文件，或另行发布完整预训练协议。任意相同架构的模型不能保证复现报告结果；入口默认拒绝不匹配的父模型哈希。

原最佳 RR EMA 的 SHA256 为 `dec45239326bf9d469c6b327fbdd8f8704d02d92cb41f2900d2ea0e72b76a7ae`，大小 129445011 字节，也未包含在 Git 仓库中。完整适配训练还需较大的原始 Adam/EMA 检查点。不要将这些文件误提交为普通 Git 对象。

## 3. 只读预检查

```bash
export RADIOMAP_DATA_ROOT=/your/path/RadioMapSeer
python scripts/run_rr.py train-pair --parent /your/path/parent_ema.pt --dry-run
```

这会验证配置、父模型文件哈希和固定清单，不写结果，也不代表完整数据内容审计或训练已成功。数据读取时还会逐图检查格式。仅加载可信来源的检查点；旧 PyTorch 检查点带 Python 元数据，本入口使用 `weights_only=False`。

## 4. 生成原 RR 缓存

```bash
tmux new-session -s radiomap-rr
python scripts/run_rr.py cache --parent /your/path/parent_ema.pt --cache-dir runs/cache
```

需要 CUDA；长缓存/训练任务强制要求 tmux。缓存包含 1003/1004 两份 FP32 轨迹，每份约 7.34 GB，总计约 14.68 GB；缓存构建和训练还需额外主存、显存及检查点磁盘空间，不要只按最终缓存尺寸预留资源。缓存入口仅读取条件，GT 不进入生成路径。

每份缓存都有 JSON 侧车记录父模型、样本选择、时间索引和文件 SHA256。入口拒绝覆盖已存在缓存。若中途停止，先检查并保留已有文件，再选择新的缓存目录重新执行；不自动删除或覆盖不完整文件。

## 5. 同父模型普通/RR 配对训练

```bash
python scripts/run_rr.py train-pair --parent /your/path/parent_ema.pt \
  --cache-dir runs/cache --output runs/rr_pair
```

默认两组各训练 1250 步并验证，输出 EMA 和包含两组原始模型、Adam、EMA、抽样位置、控制器及 RNG 的 joint 检查点。它会同时训练普通对照，而非只训练 RR；不会自动进入其他实验或 test。

若需复现原 RR 自身的完整曲线，应在**新的输出目录**显式设置：

```bash
python scripts/run_rr.py train-pair --parent /your/path/parent_ema.pt \
  --cache-dir runs/cache --output runs/rr_curve --steps 5000
```

每 1250 步验证。配对控制规则保留原实验：当两组均连续两次未达到 0.1% 的相对有效改善时，同步将 LR 乘 1/3；降 LR 后两组均再次连续两次未改善则共同早停。原报告跑到 5000 步时未触发降 LR，实际为 warmup 后恒定 1e-5；不能把之后的降 LR 分叉当作原 RR 结果。

新入口支持在完整配对更新后处理 SIGINT/SIGTERM，并保存可恢复状态。恢复必须显式指定 `--resume`，不得只加载 EMA 假装恢复原 Adam 状态：

```bash
python scripts/run_rr.py train-pair --parent /your/path/parent_ema.pt \
  --cache-dir runs/cache --output runs/rr_pair --resume
```

发布包采用新的源码哈希和检查点命名空间。原研究树检查点可作为模型验证/指定父模型初始化来源，**不能作为新入口的原始训练断点直接恢复**。改动实现后新入口会拒绝静默恢复。

## 6. 验证当前主模型

```bash
python scripts/run_rr.py evaluate \
  --checkpoint runs/rr_pair/RR/checkpoints/ema_step_00001250.pt \
  --output runs/RR_1250_evaluation
```

验证固定使用 val_select 的 400 张图、1001/1002 两组噪声、Euler25；保存逐图 SSE、整图与自由区 RMSE 等。输出目录必须是新的；不会覆盖已有验证结果。CPU 也可运行验证，但速度和数值一致性需自行核对。

## 7. 本次发布已核验与未核验的内容

- 已运行核心测试：损失、GT 隔离、梯度/Adam/EMA、真实轨迹前状态与历史、固定抽样哈希、配对控制器、原始断点恢复和 CSV 指标。
- 已用发布入口对原 RR1250 检查点完成 800 次预测，整图 RMSE 与原实验记录完全一致。
- 未在本次整理时重新执行 1250/5000 步训练或重建 4000 张图的全部缓存。已有训练结果来自原研究实验，证据明确标注来源；发布包不把整理后的测试当作新训练结果。
- 未发布数据、缓存、权重、其他探索实验或任何密钥。模型权重和合法数据获取需要另行安排。
