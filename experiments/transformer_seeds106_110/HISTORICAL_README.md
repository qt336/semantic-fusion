# 原始算术＋检索混合任务：seed106–110

用户要求在GPU4、5训练最初的普通Transformer任务，补跑106、107、108、109、110五个seed。
这是2026-09-12按论文重建的实现的新增训练，不是找回了不可用的原始训练代码。

- 普通4层softmax Transformer，宽度256、8头、MLP1024、RoPE，3331920参数。
- 每轮算术270000条（01、02、21每种90000）与检索135000条；没有其他辅助任务。
- 检索训练只排除01、02、21；不排除普通9,10、15,20、17,29。
- NTP加权交叉熵，序列含BOS/EOS，batch8190，microbatch2048，1200轮（60000更新）。
- AdamW，betas0.9/0.999，weight decay0.1；500步warmup，lr1e-3余弦下降至1e-4；float32允许TF32。
- seed同时控制数据采样、初始化和每轮打乱；每个seed单独保存数据。
- 每25轮评估并保存latest.pt，保留每轮history.jsonl；最终固定1200轮，不按测试指标选checkpoint。
- 保留原训练程序完成时的第一层B←A边遮挡评估，单独保存在result.json及masked_predictions.npz；实时主表显示未遮挡结果。
- 数据、模型、训练三个文件与reference_20260912快照逐字节相同；该快照的模型与9月实现相同，训练文件只多记录numpy版本，数据文件增加原子保存和文件锁。

## 启动与查看

数据生成与核验：`python prepare.py`。全部核验通过后：`bash start.sh`。
初始调度在`screen qt1`的`original_106_110`窗口，使用物理GPU4、5。
2026-11-08追加接续调度：`original_queue_handoff`窗口运行`queue_after_random.py`，接管既有106、107进程，不重启训练。
GPU4、5继续按原队列接续；GPU1、6分别等随机输出对照104、105完成1200轮并释放run锁后，可接续尚未启动的108、110。
同一seed不会重复启动；GPU1、6不会提前加入。训练代码、数据与全部超参数保持原协议。
旧队列管理进程暂时暂停，GPU训练worker继续使用原PID；新调度器在全部五个seed完成后恢复旧管理进程收尾。
当前调度状态见`queue_after_random_status.json`，接管记录与检查见`queue_handoff.json`、`queue_handoff_checks.json`。
退出VSCode不影响训练；其余seed自动接续。

`results.md`每15秒从已保存指标更新。每个run在`runs/transformer_seed<seed>/`，含config、日志、history、status、latest.pt，以及完成后的result.json和逐样本预测。
`protocol_manifest.json`记录数据和源文件哈希、审计结果；全部完成后运行finalize.py重新核对逐样本结果。
中断后重新启动同一脚本可从最近保存的checkpoint续训；如从旧checkpoint重跑，history可能保留重复轮数，按最后一条同轮记录处理。

语义测试每组2560条，长度3–6各640条；遵守原始采样，短序列可能重复，不擅自改为后续版本的唯一性/长度配比。
普通检索2000条，与检索训练签名不重复；算术每种4096对留出操作数，与训练不重叠。
控制组内部数据名control_a1b/control_a2b保留兼容性，表格按x1y/x2y标注。
