# 运行与验证

以下命令均从此目录根部执行。推荐 Python 3.10；用适合本机 GPU 的 PyTorch 2.5.1 构建。历史玩具实验使用 NumPy 1.26.4。训练输出指定到 `outputs/`，方便保留已归档的结果。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 无需训练或联网的验证

```bash
python verify_results.py
python experiments/qwen3_8b/verify_results.py
python experiments/loop_transformer_4x2/validate.py
python experiments/loop_transformer_4x3/validate.py
python experiments/linear_probe/robustness/verify.py
python experiments/linear_probe/robustness/verify_controls.py
```

根目录验证程序从保存的预测重新计算各玩具实验计数；重新评分 9,000 条 LLM 回复并核对论文表 6–7；按从论文提取时记录的 SHA-256 核对 19 张表格；核对最终文件校验和。Qwen 的验证脚本还会检查原始输出哈希、人工评分、成对干预和其他控制实验。原有实验验证脚本可能重写该实验的验证报告；如需保持归档字节不变，请在目录副本中运行它们。

## Transformer 和 Hybrid

```bash
python experiments/transformer_seeds106_110/train.py --architecture transformer --seed 106 --device cuda --root outputs/transformer106
python experiments/toy_hybrid/train.py --architecture hybrid --seed 101 --device cuda --root outputs/hybrid101
python experiments/transformer_controls/train.py --condition heldout_pairs --seed 101 --device cuda --root outputs/heldout101
```

默认完整预算为 1,200 轮。可先使用 `--device cpu --small --epochs 1 --batch-size 256 --microbatch 128` 验证环境。`supplementary/README.md` 有原补充实现的更详细参数和 CPU 检查方法。

## Loop transformer

两组复用 `experiments/toy_hybrid/datasets/` 下已保存并核验的 101–105 数据集。将下列 `--seed` 改为 101–105 即可逐 seed 运行；`CUDA_VISIBLE_DEVICES` 可由使用者自行指定。

```bash
python experiments/loop_transformer_4x2/train.py --seed 101 --device cuda --root outputs/loop4x2_seed101 --data-root experiments/toy_hybrid/datasets
python experiments/loop_transformer_4x3/train.py --seed 101 --device cuda --root outputs/loop4x3_seed101 --data-root experiments/toy_hybrid/datasets
```

四个独立 Transformer block 分别复用 2 次/3 次，有效深度为 8/12，独立参数量均为 3,331,920。保留原训练协议与完整反向传播。

## 线性探针

`experiments/linear_probe/robustness/` 中的 `driver.py`、`afixed_control.py`、`factorial_control.py` 是原拟合与控制实验代码。它们使用同目录数据和 `experiments/transformer_controls/runs/heldout_pairs_seed*/latest.pt` 中的冻结模型；默认使用 CUDA，并会写入各自实验目录。重新拟合时请先复制整个归档。查看结果无需拟合，打开 `robustness_summary.json`、`results.md` 或运行验证脚本即可。

## API 模型

历史结果可离线重新评分：

```bash
cd supplementary
python -m llm.validate_data
python -m unittest discover -s checks -v
python -m llm.score --responses ../results/llm/kimi.jsonl --output ../outputs/kimi_rescored
python -m llm.score --responses ../results/llm/deepseek.jsonl --output ../outputs/deepseek_rescored
python -m llm.score --responses ../results/llm/hy4.jsonl --output ../outputs/hy4_rescored
```

重新请求 API 的环境变量、模型标识和命令见 `supplementary/README.md`。API 服务与模型是否仍可访问取决于各提供商；此归档保存的是实际历史请求设置与结果。

## Qwen3-8B

阅读 `experiments/qwen3_8b/README.md`，运行该目录的 `verify_results.py` 即可复核现存输出。完整输入 token IDs、目标 attention 边、模型参数及运行版本均已归档。原生成程序未找到，因此当前包只提供结果复核，不宣称包含该程序或已经重新生成这些结果。

在 Windows PowerShell 中，请先执行 `$env:PYTHONUTF8 = "1"`，使 Python 按 UTF-8 读取中英文实验记录。
