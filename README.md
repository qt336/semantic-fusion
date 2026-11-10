# Semantic Fusion

《Semantic Fusion and Subspace Separation in Language Models: A Hypothesis and Its Predictions》的训练代码、评估数据与实验结果，按 2026-11-08 修订稿整理。实验考察语义组合对成分检索的影响，以及注意力干预、循环计算和混合架构如何改变检索表现。

## 从哪里开始

- [复现实验](REPRODUCING.md) 包含环境配置、训练命令、评分和离线验证步骤。
- [实验结果](RESULTS.md) 汇总各组结果，按论文表号链接到对应文件。
- [数据来源](PROVENANCE.md) 区分历史实验、重建实现和后续重训，并列出缺失的原始记录。

| 实验 | 代码或结果 |
|---|---|
| Transformer、Hybrid 和注意力干预 | [原补充代码](supplementary/)、[Hybrid 五个种子的历史汇总](experiments/toy_hybrid/results.md)、[Transformer 种子 106–110](experiments/transformer_seeds106_110/)、[Hybrid 种子 104 重训](experiments/hybrid_seed104_rerun/) |
| 循环 Transformer | [4 层循环 2 次](experiments/loop_transformer_4x2/)、[4 层循环 3 次](experiments/loop_transformer_4x3/) |
| 留出普通组合与线性探针 | [训练与对照](experiments/transformer_controls/)、[探针和敏感性分析](experiments/linear_probe/) |
| 三款 API 模型的中英文检索 | [固定输入、调用和评分](supplementary/llm/)、[9,000 条回复与评分](results/llm/) |
| Qwen3-8B 干预和控制实验 | [29,000 条生成记录与核验代码](experiments/qwen3_8b/) |
| 论文表格与逐运行汇总 | [19 张表及索引](results/paper_tables/)、[24 次运行的指标 CSV](results/toy_runs.csv) |

在仓库根目录运行以下命令，可用已保存的预测与评分核对结果，无需训练或调用模型 API。

```bash
python -m pip install numpy
python -X utf8 verify_results.py
python -X utf8 experiments/qwen3_8b/verify_results.py
```
