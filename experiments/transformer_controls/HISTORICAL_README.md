# 普通 Transformer 的两个组合检索对照

本轮只训练普通 Transformer。两个条件，各 3 个 seed（101、102、103），共 6 次训练，最多并行使用 4 张 GPU。
代码基于 2026-09-12 的重建版本；原始论文训练源码不可用，不能把本轮数值当成原始模型检查点的复测。

## 实验设计

| 设置 | 第一组 `heldout_pairs` | 第二组 `random_operator` |
|---|---|---|
| 真实算术 | 01、02、21，每种 90,000 条/轮 | 完全相同 |
| 检索训练 | 135,000 条/轮 | 完全相同 |
| 检索中排除的相邻组合 | 01、02、21、9,10、15,20 | 完全相同 |
| 新增样本 | 无 | `r 9 10 s → 随机值`，180,000 条/轮 |
| 9,10 的含义 | 无指定运算意义 | 占据运算符位置，但结果没有稳定规则 |
| 15,20 | 未见普通组合 | 仍未见，作为未加曝光的参照 |
| 每轮总样本 | 405,000 | 585,000 |
| 每轮优化器步数 | 50 | 72 |

“两倍频率”按运算符槽位中的样本次数定义：每个真实运算 90,000 次，随机运算符 9,10 为 180,000 次。
其他位置偶然产生的相邻数字不赋予运算意义，不作为设计频率的计数口径。
检索排除同时覆盖 memory 中相邻组合，以及原实现排除的 `(query target, answer)` 组合。
9、10、15、20 等单个 token 均会作为普通检索的 memory、查询和答案出现；只排除指定组合。

### 随机结果不是一次生成后固定

随机运算输入为 `[BOS, r, 9, 10, s]`，输出从整数 0–330 均匀抽取。
同一输入在不同训练轮、同轮重复出现时都使用独立抽样的目标。代码每轮为每个随机样本重采样一次，
每个样本每轮呈现一次；不会生成一个固定的随机函数供模型记忆。
目标同时写入 shifted label 的第 4 位和 shifted input 的第 5 位；预测目标发生在第 4 位，因果掩码防止读取未来答案。
随机标签 RNG 与模型初始化、训练 shuffle 独立，由 seed 和 epoch 决定，断点恢复可复现。

随机输出任务本身不可预测，预期猜测准确率是 1/331≈0.30%。它的准确率单列，不能混入真实算术指标。
第二组多了随机标签训练，其总训练步数和任务比例发生变化；需要同时检查真实算术和普通检索是否正常学会。

## 模型与训练

- 4 层纯 softmax causal self-attention，宽度 256，8 heads，GELU MLP 1024，Pre-LN，RoPE，词表 336。
- 3,331,920 参数；没有线性注意力或卷积，没有额外注意力消融。
- 1,200 epochs；优化器 batch 8,190，microbatch 默认 2,048，梯度累计实现相同 batch 损失。
- AdamW，betas=(0.9,0.999)，weight decay=0.1；500 update warmup；学习率 1e-3 余弦下降到 1e-4。
- float32，允许 TF32；无 dropout，无梯度裁剪。
- 沿用加权 next-token CE：检索 memory 0.1、查询 token 2、答案 10；算术操作数 0.1、结果 10；其他非 PAD token 1。
- 同 seed 的两组模型初始化一致，共享同一个数据文件中的基础训练集与测试集。
- 每 25 epochs 和第 1 epoch 评估，每 10 epochs 保存完整检查点，最终保存逐样本预测。

## 测试

五个组合共用同一套上下文、查询 token、memory 长度和插入位置；查询的是组合后面的 token，正确答案是组合的第二项。
例如 `[9,10,Q]` 查询 Q 应输出 10；输出 9 计为首 token 错误。
每组组合测试 2,560 个不同样本：memory 长度 3 为 256 条，长度 4/5/6 各 768 条。
这样避免原重建版本中短序列重复测试的问题；不同组合之间的测试内容严格配对。
普通检索测试 2,000 条、真实算术每种 4,096 条，另保留 x 1 y、x 2 y 控制。
各指标从尚未看到正确答案的位置读取 argmax。准确率、首 token 错误、查询 token 错误、EOS、其他互斥且覆盖全部样本。

## 验证与运行

现有环境：Python 3.10、PyTorch 2.5.1+cu124、NumPy、screen。

```bash
cd /home/user/toy_transformer_controls_20261018
python validate.py --full
bash start_qt1.sh --gpu-indices 0 1 2 3
```

启动脚本在已有 `screen qt1` 中增加 `controls_20261018` 窗口；不结束或覆盖其他窗口。
每张指定 GPU 同时只运行一个 seed，剩余任务自动排队，不调度到 GPU 4–7。
关闭 VS Code/SSH 不影响 screen 中的训练。恢复已有任务时必须使用相同设置。
查看会话：`screen -r qt1`；切换窗口：Ctrl-a，然后按双引号选择；离开并保留运行：Ctrl-a，再按 d。

单次训练命令：

```bash
python train.py --condition heldout_pairs --seed 101
python train.py --condition random_operator --seed 101
```

这些命令需要在能访问宿主机 NVIDIA 设备的终端中执行；沙箱内可能看不到 `/dev/nvidia*`。

## 文件

- `results.md`：每 15 秒刷新状态及最近一次评估；最终三 seed 均值和标准差仅在完整训练后显示。
- `datasets/seed*.json`：训练排除审计、单 token 覆盖次数、数据 SHA256 和测试规模。
- `validation_full.json`：普通 Transformer 与重建参考实现的一致性、因果性、数据和随机标签校验。
- `runs/<condition>_seed<seed>/train.log`、`history.jsonl`：训练日志和每轮指标。
- `latest.pt`：模型及优化器断点；`result.json`：完整训练结果；`predictions.npz`：逐样本预测。
- `scheduler_status.json`、`run_manifest.json`、`supervisor.log`：调度、GPU 限制与运行配置。
- `reference_20260912/`：原重建代码快照，仅用于验证；调度器只运行本目录的普通 Transformer。

所有结果以实际运行数据为准，不预设普通组合一定答对或随机运算符一定不会出现首 token 错误。

## 追加seed104、105（2026-11-08）

只追加random_operator条件，各1200轮；使用未改动的data.py/model.py/train.py。新调度在screen qt1/randomop_104105，共享GPU1、6。五seed结果见random_operator_5seeds_results.md，协议与验证见extension_104105/。
