# 结果索引

## Loop transformer

两组均为四个独立 block 复用，seed101–105，全数固定在第 1,200 轮评估。均值由逐 seed 的 `result.json` 重新计算；括号内为样本标准差。单位为百分比。

| 架构 | 算术正确率 | 普通检索正确率 | 语义组合检索正确率 | 语义组合首成分错误率 |
|---|---:|---:|---:|---:|
| 4×2（有效 8 层） | 99.95 (0.06) | 99.08 (1.75) | 22.97 (11.82) | 46.85 (15.83) |
| 4×3（有效 12 层） | 91.87 (17.56) | 99.54 (0.32) | 14.05 (9.47) | 39.83 (21.03) |

逐 seed 结果、各 operator pair、训练曲线和逐条预测：

- [4×2 结果](experiments/loop_transformer_4x2/results.md) · [运行记录](experiments/loop_transformer_4x2/runs/)
- [4×3 结果](experiments/loop_transformer_4x3/results.md) · [运行记录](experiments/loop_transformer_4x3/runs/)
- [全部 24 个已验证玩具模型运行的 CSV](results/toy_runs.csv)

4×3 的 seed101 算术正确率为约 60.45%，原样保留，五 seed 均值包含该次运行。Hybrid seed104 重训的最终结果对应末尾 spike 状态；此 CSV 中的该行不是论文报告的第 1,175 轮数值。

## 论文内容与数据位置

| 表格/图 | 数据与代码 |
|---|---|
| 表 1–2、8–9、11、17：任务、协议和提示词 | [全部表格源文件](results/paper_tables/)；[补充代码与输入](supplementary/) |
| 表 3–5：原 Transformer 五 seed 与 attention 遮挡 | [表 3](results/paper_tables/table_03_retrieval-results-seeds.tex)、[表 4](results/paper_tables/table_04_x1y-x2y-validation.tex)、[表 5](results/paper_tables/table_05_seed_results.tex)；原始运行文件未找到 |
| 表 6–7、10：三款 API 模型 | [9,000 条回复与汇总](results/llm/)；[代码](supplementary/llm/) |
| 表 12–13：Hybrid | [保留的历史汇总](experiments/toy_hybrid/results.md)；[代码](experiments/toy_hybrid/) |
| 图 4：Hybrid seed104 曲线 | [后续重训](experiments/hybrid_seed104_rerun/)；[来源核验](results/paper_evidence/hybrid_spike_evidence.json) |
| 表 14：留出普通组合 | [代码与逐条结果](experiments/transformer_controls/) |
| 表 15：线性探针 | [代码、拟合权重与结果](experiments/linear_probe/robustness/)；[汇总](experiments/linear_probe/robustness/robustness_summary.json) |
| 表 16：Qwen3-8B 语义输入前 16 层遮挡 | [对应输入与原始输出](experiments/qwen3_8b/early16_attention/)；[全结果包](experiments/qwen3_8b/) |
| 表 18–19：新增 Transformer seed106–110 | [代码、数据和预测](experiments/transformer_seeds106_110/) |

## 本次离线核验

- 三款 API 模型：重新评分全部 9,000 条，结果与历史评分和论文表 6–7 一致。
- 玩具模型：核对 24 个完整运行、270 个预测条件，共 823,680 条逐样本预测。
- Qwen3-8B：29,000 条生成记录、人工评分、原始输出哈希和成对控制核验通过。
- Loop 4×2/4×3：参数共享、展开模型的梯度求和等价性、因果性及五个数据集哈希验证通过。
- 线性探针：三个 seed 的 4,134 项预测统计与数据分割核验通过。
- 原补充代码的 6 项 LLM 请求/评分回归检查通过；3,000 条固定输入与候选池检查通过。

本次没有重新进行 GPU 训练、生成新的模型回复或重新验证远端 tokenizer。来源限制见 [PROVENANCE.md](PROVENANCE.md)。
