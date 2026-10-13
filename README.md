# RR：实际采样状态与自条件历史监督的 RadioMap Flow Matching

本仓库整理项目当前选定的论文核心：在强普通自条件模型上，保留原 Flow Matching（FM）损失，用冻结父模型**实际采样的状态和实际自条件历史**增加针对性监督，降低 RadioMap 预测误差。这里 RR 指 rollout state + rollout history 均来自真实模型采样，而非实测轨迹或真实标签构造的插值轨迹。

当前主模型为原 RR 的 **1250 步 EMA 检查点**，验证 RMSE 为 **0.0191921267**。这是已评估开发节点中的最佳候选，不是独立测试集的最终论文结论。

## 实现目标与核心设计

普通自条件 FM 训练通常在真实图与噪声之间的直线路径上预测；部署时却沿自己的预测逐步采样，并使用自己产生的历史。RR 直接在父模型实际遇到的“状态—历史”组合上训练，使网络学会从这些偏离理想路径的输入继续向正确终点预测。

1. 冻结已充分增强的普通自条件父模型，保持建筑/TX 条件、U-Net、25 步 Euler 采样器不变。
2. 在固定训练池上，用噪声 1003/1004 生成无 GT 采样轨迹，记录第 7、15、21 次预测前的状态和传入的自条件历史。
3. 从同一父模型分出普通对照和 RR；两组使用全新 AdamW/EMA，匹配样本、噪声、时间、历史间隔、空参考掩码及学习率。
4. 普通对照只优化原 FM 损失；RR 额外将缓存状态预测的终点对齐训练标签。缓存和 RR 预测输入不包含 GT，标签仅用于监督。
5. 使用同一个无 GT Euler25 采样器验证/部署，不增加网络参数或推理步数。

RR 不是新增几何平滑，也不是把建筑/TX 再注入一个新分支。当前方法采用原有 identity 自条件参考，主要变化发生在**监督输入的状态与历史分布**。

详细公式及与普通自条件的区别见 [方法说明](docs/METHOD.md)。

## 已有结果

共同父模型 RMSE：`0.0197166057`。

| 新增训练步数/组 | 普通自条件 | 原 RR | RR 相对同节点普通模型的 RMSE 降幅 |
| --- | ---: | ---: | ---: |
| **1250（当前主模型）** | 0.0196868976 | **0.0191921267** | **2.5132%** |
| 2500 | 0.0196289189 | 0.0192450221 | 1.9558% |
| 3750 | 0.0195993966 | 0.0192838197 | 1.6101% |
| 5000 | 0.0195924570 | 0.0193088207 | 1.4477% |

1250 步 RR 相对冻结父模型降低 RMSE **2.6601%**；相对同节点普通模型，50 个验证场景中 **49 个**改善，场景 bootstrap 的相对 RMSE 降幅 95% 区间为 **[2.1742%, 2.8559%]**。5000 步同预算终点对照也有收益，但不是当前最佳 RR 检查点。

协议：500 个训练场景 × 8 个 TX = 4000 张训练图；验证为 50 个场景 × 8 个 TX × 2 组噪声 = 800 次预测；整幅图汇总 SSE/像素后开方，目标为原 PNG/255，不裁剪、不删除建筑像素、不转换为 dB。

这是单个父模型、一个适配种子（43）、复用开发验证集的结果；尚非多随机种子、全 40000 张训练图或独立 test 确认。详情及晚期回退说明见 [结果与限制](docs/RESULTS.md)。其他探索实验未收入本仓库，本地记录不受影响。

## 代码入口

| 文件 | 实现内容 |
| --- | --- |
| `radiomap/training/strong_rr.py` | GT-free RR 预测、原 FM + RR 联合损失与更新 |
| `radiomap/training/losses.py` | 终点误差到速度误差的等价变换；RR 固定 block_size=1 |
| `radiomap/training/trainers.py` | 普通三次前向自条件训练、AdamW 参数分组 |
| `radiomap/training/rr_runtime.py` | 实际轨迹缓存、固定抽样流、配对学习率控制 |
| `radiomap/models/` | 与原实验相同的条件编码器和 U-Net |
| `radiomap/flow/solver.py` | 普通模型与 RR 共用的 GT-free Euler25 采样器 |
| `scripts/run_rr.py` | 缓存、配对训练、验证的可移植入口 |
| `configs/base.yaml` | 原 RR 科学参数，默认运行到当前选定的 1250 步 |
| `metadata/` | 固定场景划分与训练/验证图编号，不含图像数据 |
| `evidence/` | 原实验节点指标、逐图 SSE、来源哈希和整理后复现核验 |
| `tests/` | 损失、梯度/Adam/EMA、GT 隔离、轨迹、抽样与结果核验 |

保留少量基础代码中的旧 schema/通用辅助接口以维持原实现；运行入口仅允许 identity/unit 的普通自条件和 RR，不包含几何/oracle/其他分叉实验队列。

## 使用方法

需要自行准备合法取得的 RadioMapSeer DPM 数据，以及与记录 SHA256 一致的强普通父模型 EMA。**数据集、约 14.7 GB 的缓存和模型检查点均未上传。仅克隆仓库不能凭空复现预训练父模型。**

```bash
python -m pip install -e '.[dev]'
python -m pytest -q -p no:cacheprovider
export RADIOMAP_DATA_ROOT=/your/path/RadioMapSeer

# 只检查协议和父模型，不写文件、不启动训练。
python scripts/run_rr.py train-pair --parent /your/path/parent_ema.pt --dry-run

# 长任务必须在 tmux 中执行；默认配对训练到1250步。
tmux new-session -s radiomap-rr
python scripts/run_rr.py cache --parent /your/path/parent_ema.pt --cache-dir runs/cache
python scripts/run_rr.py train-pair --parent /your/path/parent_ema.pt \
  --cache-dir runs/cache --output runs/rr_pair

# 使用新的输出目录验证保存的RR EMA。
python scripts/run_rr.py evaluate \
  --checkpoint runs/rr_pair/RR/checkpoints/ema_step_00001250.pt \
  --output runs/rr_validation
```

完整环境、依赖资源、5000 步曲线复现与断点恢复见 [复现步骤](docs/REPRODUCE.md)。本次整理没有重新训练；已用新入口对原 1250 步 EMA 完整验证，得到与原记录完全一致的 RMSE `0.019192126704927724`。

## 发布范围与论文表述

本仓库提供论文核心实现和可核验的开发结果，不宣称“首次提出”已被文献查新证明，也不宣称已完成正式论文所需的全部实验。新增训练步数匹配不代表训练 FLOPs 匹配；RR 增加训练计算和离线缓存，推理结构及 NFE 不变。

公开仓库不代表已授予任意开源许可；当前未附加 LICENSE。数据及第三方组件须遵循各自原有许可。项目代码与结果的进一步使用、权重发布及论文署名需要由项目所有者另行决定。
