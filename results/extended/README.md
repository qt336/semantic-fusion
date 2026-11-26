# 扩展实验结果

[Transformer 深度与宽度实验：逐 seed 成绩、训练曲线和线性探针](toy/transformer_depth_width/)

截至 2026-11-26。API 成绩沿用原先的首答评分，并对纳入样本使用下述统一标准。逐题文件保存提示词、目标、可见回答和已归档评分。公开接口返回的模型名和推理计数按实验记录解释。

[API 实验筛后的分条件计数 CSV](conditions.csv) · [论文三模型按当前口径复核的结果](../llm/) · [Qwen3-8B 遮挡实验](../../experiments/qwen3_8b/) · [论文与 toy model 结果总索引](../../RESULTS.md)

## 中英文结果速查

[Fable 5.1 与 Haiku 5.5 进度与结果](fable51_haiku55_rightapi_20261125.md)


默认规则仅纳入明确报告 0 个思考 token 且无思考块或可见推理的回答；空的或加密的 reasoning 项也排除。

**五项历史口径例外。** GPT‑5 请求 `minimal`，回执均报告 0 个思考 token，同时含加密的 `reasoning` 项。按原实验口径纳入全部回答。DeepSeek V4 Pro、DeepSeek V4.1 Flash 1017、Qwen3.7 Flash 和 Qwen3.8 Max-0902 的原测试均请求关闭思考。四次运行各有 3,000 条回执，均未报告思考 token 数，沿用原评分，每组 500 条。表中已标出例外。回执只能说明接口报告了什么。

**开启思考的对照组。** DeepSeek V4 Pro 与 DeepSeek V4.1 Flash 1017 另有两组 `high` 实验，分别收到 128/180 和 180/180 条回答，每条均报告正数思考 token。下表单列这两组的成绩。

每格依次给出**正确率 / 错答首成分率**（%），以各条件实际纳入数为分母。语义及其加逗号版本独立筛选；两组留下的题目可能不同。没有合格回答时显示“—”。[分条件计数](conditions.csv) · [各轮纳入及排除原因](eligibility.csv)

### 中文

| 模型 | 思考模式（请求） | 等级 | 纳入 / 已有回执（本语言） | 语义 | 独立随机 | 语义加逗号 |
|---|---|---|---|---:|---:|---:|
| [Claude Fable 5（RightAPI）](fable5_rightapi_20261125/README.md) | 自适应（筛选明确零计数） | `low` | 960 / 1,500 | 89.8 / 8.6 | 95.5 / 2.2 | 99.1 / 0.3 |
| [Claude Fable 5.1（RightAPI）](fable51_rightapi_20261125/README.md) | 自适应（筛选明确零计数） | `low` | 941 / 1,500 | 94.0 / 5.4 | 99.3 / 0.4 | 100.0 / 0.0 |
| [Claude Fable 5.2（RightAPI）](fable52_rightapi_20261125/README.md) | 自适应（筛选明确零计数） | `low` | 1,352 / 1,500 | 95.8 / 3.0 | 98.7 / 1.3 | 99.8 / 0.2 |
| [Claude Haiku 5.5（RightAPI）](haiku55_rightapi_20261125/README.md) | 自适应（筛选明确零计数） | `low` | 602 / 1,500 | 52.7 / 34.2 | 85.1 / 10.0 | 91.2 / 4.2 |
| [claude-opus-4-8](opus48_rightapi/README.md) | 请求关闭 | 未设 | 199 / 300 | 58.7 / 36.5 | 89.6 / 10.4 | 100.0 / 0.0 |
| [Claude Opus 5（ZenMux）](opus5_zenmux_20261126/README.md) | 请求关闭（按响应筛选） | `low` | 1,500 / 1,500 | 23.0 / 64.6 | 73.4 / 26.4 | 100.0 / 0.0 |
| [claude-opus-5-5](opus55_20261124/README.md) | 自适应（仅计回执明确为零且无思考块） | `low` | 936 / 1,500 | 95.0 / 3.8 | 98.5 / 1.2 | 99.4 / 0.0 |
| [claude-sonnet-5-5](sonnet55_20261124/README.md) | 请求关闭答题前思考（按响应筛选） | `high` | 1,197 / 1,500 | 83.0 / 11.0 | 98.5 / 1.3 | 99.5 / 0.0 |
| [DeepSeek V4 Pro](../llm/) | 请求关闭（历史口径） | 未设 | 1,500 / 1,500 | 31.8 / 67.4 | 66.0 / 34.0 | 94.2 / 5.6 |
| [DeepSeek V4 Pro（128/180）](deepseek_v4pro_thinking_20261124/README.md) | 开启思考对照 | `high` | 68 / 68 | 100.0 / 0.0 | 100.0 / 0.0 | 100.0 / 0.0 |
| [DeepSeek V4.1 Flash 1017（180 题）](deepseek_v41flash_thinking_20261124/README.md) | 开启思考对照 | `high` | 90 / 90 | 93.3 / 0.0 | 100.0 / 0.0 | 100.0 / 0.0 |
| [deepseek-v4.1-flash-1017](deepseek_v41_flash_20261020/README.md) | 请求关闭（历史口径） | 未设 | 1,500 / 1,500 | 58.4 / 38.2 | 81.2 / 18.6 | 100.0 / 0.0 |
| [gemini-3.6-flash](gemma4_gemini36_zenmux_20261124/full3000/README.md) | 最低推理档 | `minimal` | 1,500 / 1,500 | 87.2 / 10.8 | 97.4 / 0.2 | 100.0 / 0.0 |
| [gemma-4-31b-it](gemma4_gemini36_zenmux_20261124/full3000/README.md) | 请求关闭（按响应筛选） | 未设 | 1,500 / 1,500 | 79.4 / 18.4 | 98.8 / 1.0 | 100.0 / 0.0 |
| [hy4（暂定评分）](hy4_20261104/README.md) | 请求关闭 | 未设 | 1,498 / 1,500 | 54.5 / 44.7 | 90.4 / 9.6 | 99.8 / 0.2 |
| [HY4 preview](../llm/) | 请求关闭 | 未设 | 1,500 / 1,500 | 55.6 / 40.8 | 91.0 / 8.8 | 99.8 / 0.0 |
| [Kimi K3](../llm/) | 请求关闭 | `none` | 1,500 / 1,500 | 76.4 / 23.4 | 97.4 / 2.6 | 100.0 / 0.0 |
| [MiniMax/MiniMax-M3](minimax_m3_20261020/README.md) | 请求关闭 | 未设 | 484 / 1,500 | 18.5 / 26.8 | 39.5 / 35.7 | 59.7 / 27.0 |
| [Grok 4.3](grok43_none_full_20261125/README.md) | 请求关闭 | `none` | 1,500 / 1,500 | 10.4 / 59.4 | 34.8 / 63.0 | 76.2 / 18.4 |
| [gpt-5](gpt5_minimal_full_20261125/README.md) | 最低推理档（历史口径） | `minimal` | 1,500 / 1,500 | 68.4 / 18.2 | 95.6 / 2.6 | 99.0 / 0.0 |
| [gpt-5.1](gpt51_none_full_20261125/README.md) | 请求关闭 | `none` | 1,500 / 1,500 | 76.0 / 16.0 | 97.8 / 2.0 | 100.0 / 0.0 |
| [gpt-5.4](gpt54_none_full_20261125/README.md) | 请求关闭 | `none` | 1,500 / 1,500 | 96.0 / 3.8 | 99.8 / 0.2 | 100.0 / 0.0 |
| [gpt-5.5](gpt55_none_full_20261125/README.md) | 请求关闭 | `none` | 1,500 / 1,500 | 99.4 / 0.4 | 100.0 / 0.0 | 100.0 / 0.0 |
| [gpt-5.6-luna](gpt56luna_none_full_20261125/README.md) | 请求关闭 | `none` | 1,500 / 1,500 | 76.4 / 20.0 | 99.0 / 1.0 | 99.8 / 0.2 |
| [gpt-5.6-sol](gpt56sol_none_full_20261125/README.md) | 请求关闭 | `none` | 1,500 / 1,500 | 98.2 / 1.6 | 99.6 / 0.0 | 100.0 / 0.0 |
| [gpt-5.6-terra](gpt56terra_none_full_20261125/README.md) | 请求关闭 | `none` | 1,500 / 1,500 | 80.0 / 18.6 | 99.4 / 0.6 | 100.0 / 0.0 |
| [gpt-6-luna](gpt6luna_none_full_20261126/README.md) | 请求关闭 | `none` | 1,500 / 1,500 | 93.0 / 5.0 | 99.4 / 0.6 | 99.6 / 0.2 |
| [gpt-6.1-terra](gpt61terra_zenmux_20261124/README.md) | 请求关闭 | `none` | 1,500 / 1,500 | 99.6 / 0.2 | 99.8 / 0.0 | 99.8 / 0.0 |
| [qwen3.7-flash](qwen37_flash/README.md) | 请求关闭（历史口径） | 未设 | 1,500 / 1,500 | 79.4 / 6.6 | 94.6 / 1.6 | 100.0 / 0.0 |
| [qwen3.8-max-0902](qwen38_max_20261020/README.md) | 请求关闭（历史口径） | 未设 | 1,500 / 1,500 | 89.2 / 7.4 | 96.8 / 2.6 | 99.0 / 0.8 |

### 英文

| 模型 | 思考模式（请求） | 等级 | 纳入 / 已有回执（本语言） | 语义 | 独立随机 | 语义加逗号 |
|---|---|---|---|---:|---:|---:|
| [Claude Fable 5（RightAPI）](fable5_rightapi_20261125/README.md) | 自适应（筛选明确零计数） | `low` | 313 / 1,500 | 76.7 / 20.4 | 94.8 / 4.3 | 96.8 / 3.2 |
| [Claude Fable 5.1（RightAPI）](fable51_rightapi_20261125/README.md) | 自适应（筛选明确零计数） | `low` | 908 / 1,500 | 97.4 / 1.9 | 98.3 / 1.4 | 93.8 / 0.3 |
| [Claude Fable 5.2（RightAPI）](fable52_rightapi_20261125/README.md) | 自适应（筛选明确零计数） | `low` | 1,401 / 1,500 | 97.1 / 2.7 | 99.1 / 0.7 | 97.0 / 0.0 |
| [Claude Haiku 5.5（RightAPI）](haiku55_rightapi_20261125/README.md) | 自适应（筛选明确零计数） | `low` | 496 / 1,500 | 17.0 / 42.2 | 54.2 / 33.1 | 40.2 / 19.6 |
| [claude-opus-4-8](opus48_rightapi/README.md) | 请求关闭 | 未设 | 207 / 300 | 37.3 / 47.8 | 66.2 / 28.2 | 75.4 / 13.0 |
| [Claude Opus 5（ZenMux）](opus5_zenmux_20261126/README.md) | 请求关闭（按响应筛选） | `low` | 1,489 / 1,500 | 32.7 / 59.5 | 81.7 / 15.7 | 97.2 / 1.8 |
| [claude-opus-5-5](opus55_20261124/README.md) | 自适应（仅计回执明确为零且无思考块） | `low` | 593 / 1,500 | 91.0 / 7.5 | 91.9 / 6.2 | 100.0 / 0.0 |
| [claude-sonnet-5-5](sonnet55_20261124/README.md) | 请求关闭答题前思考（按响应筛选） | `high` | 1,199 / 1,500 | 95.4 / 2.9 | 96.5 / 2.3 | 93.4 / 0.8 |
| [DeepSeek V4 Pro](../llm/) | 请求关闭（历史口径） | 未设 | 1,500 / 1,500 | 33.6 / 32.8 | 75.0 / 11.4 | 55.2 / 12.0 |
| [DeepSeek V4 Pro（128/180）](deepseek_v4pro_thinking_20261124/README.md) | 开启思考对照 | `high` | 60 / 60 | 100.0 / 0.0 | 100.0 / 0.0 | 87.5 / 0.0 |
| [DeepSeek V4.1 Flash 1017（180 题）](deepseek_v41flash_thinking_20261124/README.md) | 开启思考对照 | `high` | 90 / 90 | 100.0 / 0.0 | 100.0 / 0.0 | 100.0 / 0.0 |
| [deepseek-v4.1-flash-1017](deepseek_v41_flash_20261020/README.md) | 请求关闭（历史口径） | 未设 | 1,500 / 1,500 | 25.6 / 74.2 | 80.8 / 17.8 | 75.4 / 19.0 |
| [gemini-3.6-flash](gemma4_gemini36_zenmux_20261124/full3000/README.md) | 最低推理档 | `minimal` | 1,500 / 1,500 | 54.4 / 6.6 | 82.4 / 3.4 | 64.2 / 13.6 |
| [gemma-4-31b-it](gemma4_gemini36_zenmux_20261124/full3000/README.md) | 请求关闭（按响应筛选） | 未设 | 1,500 / 1,500 | 56.2 / 18.2 | 83.4 / 6.4 | 72.6 / 3.2 |
| [hy4（暂定评分）](hy4_20261104/README.md) | 请求关闭 | 未设 | 1,500 / 1,500 | 36.6 / 63.0 | 94.6 / 4.8 | 100.0 / 0.0 |
| [HY4 preview](../llm/) | 请求关闭 | 未设 | 1,500 / 1,500 | 65.8 / 30.8 | 89.0 / 7.8 | 93.2 / 6.0 |
| [Kimi K3](../llm/) | 请求关闭 | `none` | 1,500 / 1,500 | 83.8 / 12.6 | 94.4 / 5.0 | 100.0 / 0.0 |
| [MiniMax/MiniMax-M3](minimax_m3_20261020/README.md) | 请求关闭 | 未设 | 477 / 1,500 | 45.8 / 49.4 | 66.0 / 24.4 | 69.0 / 20.6 |
| [Grok 4.3](grok43_none_full_20261125/README.md) | 请求关闭 | `none` | 1,500 / 1,500 | 6.2 / 93.4 | 59.2 / 37.0 | 70.2 / 28.0 |
| [gpt-5](gpt5_minimal_full_20261125/README.md) | 最低推理档（历史口径） | `minimal` | 1,500 / 1,500 | 79.4 / 16.0 | 96.0 / 3.2 | 89.6 / 2.0 |
| [gpt-5.1](gpt51_none_full_20261125/README.md) | 请求关闭 | `none` | 1,500 / 1,500 | 79.4 / 13.0 | 96.8 / 1.6 | 77.8 / 15.8 |
| [gpt-5.4](gpt54_none_full_20261125/README.md) | 请求关闭 | `none` | 1,500 / 1,500 | 88.8 / 3.6 | 99.6 / 0.2 | 44.8 / 3.0 |
| [gpt-5.5](gpt55_none_full_20261125/README.md) | 请求关闭 | `none` | 1,500 / 1,500 | 99.0 / 1.0 | 99.4 / 0.0 | 88.2 / 4.4 |
| [gpt-5.6-luna](gpt56luna_none_full_20261125/README.md) | 请求关闭 | `none` | 1,500 / 1,500 | 43.4 / 26.0 | 81.2 / 10.2 | 9.6 / 9.4 |
| [gpt-5.6-sol](gpt56sol_none_full_20261125/README.md) | 请求关闭 | `none` | 1,500 / 1,500 | 99.2 / 0.2 | 99.6 / 0.2 | 99.8 / 0.0 |
| [gpt-5.6-terra](gpt56terra_none_full_20261125/README.md) | 请求关闭 | `none` | 1,500 / 1,500 | 64.2 / 28.0 | 98.0 / 1.0 | 69.6 / 5.2 |
| [gpt-6-luna](gpt6luna_none_full_20261126/README.md) | 请求关闭 | `none` | 1,500 / 1,500 | 69.8 / 18.4 | 88.6 / 7.0 | 37.6 / 8.4 |
| [gpt-6.1-terra](gpt61terra_zenmux_20261124/README.md) | 请求关闭 | `none` | 1,500 / 1,500 | 100.0 / 0.0 | 99.6 / 0.4 | 100.0 / 0.0 |
| [qwen3.7-flash](qwen37_flash/README.md) | 请求关闭（历史口径） | 未设 | 1,500 / 1,500 | 37.6 / 56.6 | 81.4 / 6.2 | 58.2 / 33.4 |
| [qwen3.8-max-0902](qwen38_max_20261020/README.md) | 请求关闭（历史口径） | 未设 | 1,500 / 1,500 | 47.4 / 52.2 | 93.6 / 5.6 | 96.6 / 3.0 |

排除原因见 `eligibility.csv`。每条回执只记一次，依次检查计数缺失（missing）、计数为正（positive）、报零但含思考块或推理内容（block）、块状态无法确认（unverified）和其他协议或评分条件（other）。keep 为总纳入数，其中历史口径例外及开启思考对照组的纳入数另列在 historical_exception 列。missing 和 block 仅统计因这些原因排除的回执。各条件原有回执数见 `conditions.csv` 的 received 列。后续新增 API 结果也须遵守同一规则，并通过 `python results/extended/verify_zero.py` 核对。

## 模型测试与对照

逐题下载只含纳入的回答。不同运行可能复用题目，跨运行合计不能视为独立样本数。

| 实验 | 请求设置 | 纳入 / 已有回执 | 逐题下载 |
|---|---|---:|---|
| [DeepSeek V4.1 Flash 前序检索测试](deepseek_v41_flash_20261020/README.md) <br><sub>deepseek_v41_flash_20261020</sub> | 请求关闭（历史口径），未设等级 | 3,000 / 3,000 | [3,000 条](deepseek_v41_flash_20261020/answers.jsonl.gz) |
| [DeepSeek V4.1 Flash 1017 开启思考的前序检索测试](deepseek_v41flash_thinking_20261124/README.md) <br><sub>deepseek_v41flash_thinking_20261124</sub> | 开启思考对照，`high` | 180 / 180 | [180 条](deepseek_v41flash_thinking_20261124/answers.jsonl.gz) |
| [DeepSeek V4 Pro 开启思考的前序检索测试](deepseek_v4pro_thinking_20261124/README.md) <br><sub>deepseek_v4pro_thinking_20261124</sub> | 开启思考对照，`high` | 128 / 128 | [128 条](deepseek_v4pro_thinking_20261124/answers.jsonl.gz) |
| [Gemma 4 31B IT 与 Gemini 3.6 Flash 的前序检索](gemma4_gemini36_zenmux_20261124/fixed180/README.md) <br><sub>gemma4_gemini36_zenmux_20261124/fixed180</sub> | Gemma 请求关闭，未设等级；Gemini `minimal` | 360 / 360 | [360 条](gemma4_gemini36_zenmux_20261124/fixed180/answers.jsonl.gz) |
| [Gemma 4 31B IT 与 Gemini 3.6 Flash 的前序检索](gemma4_gemini36_zenmux_20261124/full3000/README.md) <br><sub>gemma4_gemini36_zenmux_20261124/full3000</sub> | Gemma 请求关闭，未设等级；Gemini `minimal` | 6,000 / 6,000 | [6,000 条](gemma4_gemini36_zenmux_20261124/full3000/answers.jsonl.gz) |
| [Gemma 4 31B IT 与 Gemini 3.6 Flash 单题对照](gemma4_gemini36_zenmux_20261124/README.md) <br><sub>gemma4_gemini36_zenmux_20261124</sub> | Gemma 请求关闭，未设等级；Gemini `minimal` | 通道或参数诊断 | — |
| [GLM-5.4 前序检索可用性核验](glm54_20261124/README.md) <br><sub>glm54_20261124</sub> | 请求关闭，未获回答 | 通道或参数诊断 | — |
| [GPT-5.1 中英文前序检索测试](gpt51_none_full_20261125/README.md) <br><sub>gpt51_none_full_20261125</sub> | 请求关闭，`none` | 3,000 / 3,000 | [3,000 条](gpt51_none_full_20261125/answers.jsonl.gz) |
| [GPT-5.1 中英文前序检索测试](gpt51_none_zenmux_20261125/README.md) <br><sub>gpt51_none_zenmux_20261125</sub> | 请求关闭，`none` | 180 / 180 | [180 条](gpt51_none_zenmux_20261125/answers.jsonl.gz) |
| [GPT-5.4 中英文前序检索测试](gpt54_none_full_20261125/README.md) <br><sub>gpt54_none_full_20261125</sub> | 请求关闭，`none` | 3,000 / 3,000 | [3,000 条](gpt54_none_full_20261125/answers.jsonl.gz) |
| [GPT-5.5：关闭思考连通性检查](gpt55_none_20261124/README.md) <br><sub>gpt55_none_20261124</sub> | 请求 `none`，未获回答 | 通道或参数诊断 | — |
| [GPT-5.5 中英文前序检索测试](gpt55_none_full_20261125/README.md) <br><sub>gpt55_none_full_20261125</sub> | 请求关闭，`none` | 3,000 / 3,000 | [3,000 条](gpt55_none_full_20261125/answers.jsonl.gz) |
| [GPT-5.6 Luna 中英文前序检索测试](gpt56luna_none_full_20261125/README.md) <br><sub>gpt56luna_none_full_20261125</sub> | 请求关闭，`none` | 3,000 / 3,000 | [3,000 条](gpt56luna_none_full_20261125/answers.jsonl.gz) |
| [GPT-5.6 Luna 中英文前序检索测试](gpt56luna_zenmux_20261124/README.md) <br><sub>gpt56luna_zenmux_20261124</sub> | 请求关闭，`none` | 180 / 180 | [180 条](gpt56luna_zenmux_20261124/answers.jsonl.gz) |
| [GPT-5.6 Sol 在其他模型错答首成分的题目上的表现](gpt56sol_hard_pairs_20261124/README.md) <br><sub>gpt56sol_hard_pairs_20261124</sub> | 请求关闭，`none` | 24 / 24 | [24 条](gpt56sol_hard_pairs_20261124/answers.jsonl.gz) |
| [GPT-5.6 Sol：请求关闭推理的小样本测试](gpt56sol_none_20261124/README.md) <br><sub>gpt56sol_none_20261124</sub> | 主测 `none`；另有档位诊断 | 180 / 180 | [180 条](gpt56sol_none_20261124/answers.jsonl.gz) |
| [GPT-5.6 Sol 中英文前序检索测试](gpt56sol_none_full_20261125/README.md) <br><sub>gpt56sol_none_full_20261125</sub> | 请求关闭，`none` | 3,000 / 3,000 | [3,000 条](gpt56sol_none_full_20261125/answers.jsonl.gz) |
| [GPT-5.6 Sol 位置对照](gpt56sol_position_control_20261124/README.md) <br><sub>gpt56sol_position_control_20261124</sub> | 请求关闭，`none` | 24 / 24 | [24 条](gpt56sol_position_control_20261124/answers.jsonl.gz) |
| [GPT-5.6 Terra 中英文前序检索测试](gpt56terra_none_full_20261125/README.md) <br><sub>gpt56terra_none_full_20261125</sub> | 请求关闭，`none` | 3,000 / 3,000 | [3,000 条](gpt56terra_none_full_20261125/answers.jsonl.gz) |
| [GPT-5 中英文前序检索测试](gpt5_minimal_full_20261125/README.md) <br><sub>gpt5_minimal_full_20261125</sub> | 历史口径，`minimal` | 3,000 / 3,000 | [3,000 条](gpt5_minimal_full_20261125/answers.jsonl.gz) |
| [GPT-5 中英文前序检索测试](gpt5_zenmux_20261125/README.md) <br><sub>gpt5_zenmux_20261125</sub> | 历史口径，`minimal` | 180 / 180 | [180 条](gpt5_zenmux_20261125/answers.jsonl.gz) |
| [GPT-6.1 Terra 中英文前序检索测试](gpt61terra_zenmux_20261124/README.md) <br><sub>gpt61terra_zenmux_20261124</sub> | 请求关闭，`none` | 3,000 / 3,000 | [3,000 条](gpt61terra_zenmux_20261124/answers.jsonl.gz) |
| [GPT-6 Luna（全量）](gpt6luna_none_full_20261126/README.md) <br><sub>gpt6luna_none_full_20261126</sub> | 请求关闭，`none` | 3,000 / 3,000 | [3,000 条](gpt6luna_none_full_20261126/answers.jsonl.gz) |
| [GPT-6 Luna 中英文前序检索测试](gpt6luna_zenmux_20261124/README.md) <br><sub>gpt6luna_zenmux_20261124</sub> | 请求关闭，`none` | 180 / 180 | [180 条](gpt6luna_zenmux_20261124/answers.jsonl.gz) |
| [GPT 转接通道：额外输入与请求隔离核验](gpt_channel_context_audit_20261124/README.md) <br><sub>gpt_channel_context_audit_20261124</sub> | 通道诊断，主请求 `none` | 通道或参数诊断 | — |
| [GPT 关闭推理通道试测](gpt_none_20261124/README.md) <br><sub>gpt_none_20261124</sub> | 主测 `none`；另有 `high` 诊断 | 360 / 360 | [360 条](gpt_none_20261124/answers.jsonl.gz) |
| [Grok 4.3（ZenMux，全量）](grok43_none_full_20261125/README.md) <br><sub>grok43_none_full_20261125</sub> | 请求关闭，`none` | 3,000 / 3,000 | [3,000 条](grok43_none_full_20261125/answers.jsonl.gz) |
| [Grok 4.3 前序检索测试](grok43_none_20261124/README.md) <br><sub>grok43_none_20261124</sub> | 请求 `none`，未获回答 | 通道或参数诊断 | — |
| [HY4 正式版前序检索测试](hy4_20261104/README.md) <br><sub>hy4_20261104</sub> | 请求关闭，未设等级 | 2,998 / 3,000 | [2,998 条](hy4_20261104/answers.jsonl.gz) |
| [MiniMax M3 前序检索测试](minimax_m3_20261020/README.md) <br><sub>minimax_m3_20261020</sub> | 请求关闭，未设等级 | 961 / 3,000 | [961 条](minimax_m3_20261020/answers.jsonl.gz) |
| [可访问的无思考模型核验](no_thinking_model_audit_20261124/README.md) <br><sub>no_thinking_model_audit_20261124</sub> | 多种关闭设置，见各模型记录 | 通道或参数诊断 | — |
| [Opus 5.5 low 前序检索测试](opus55_20261124/README.md) <br><sub>opus55_20261124</sub> | 自适应，`low` | 1,529 / 3,000 | [1,529 条](opus55_20261124/answers.jsonl.gz) |
| [Opus 5 前序检索测试](opus5_20261103/high_pilot/README.md) <br><sub>opus5_20261103/high_pilot</sub> | 请求关闭（按响应筛选），`high` | 91 / 120 | [91 条](opus5_20261103/high_pilot/answers.jsonl.gz) |
| [Claude Opus 5 · ZenMux 重测](opus5_zenmux_20261126/README.md) <br><sub>opus5_zenmux_20261126</sub> | 请求关闭，`low` | 2,989 / 3,000 | [2,989 条](opus5_zenmux_20261126/answers.jsonl.gz) |
| [Opus 5 · RightAPI 历史结果](opus5_20261103/README.md) <br><sub>opus5_20261103</sub> | 请求关闭（按响应筛选），`low` | 2,360 / 3,000 | [2,360 条](opus5_20261103/answers.jsonl.gz) |
| [Qwen3.8 Max-0902 前序检索测试](qwen38_max_20261020/README.md) <br><sub>qwen38_max_20261020</sub> | 请求关闭（历史口径），未设等级 | 3,000 / 3,000 | [3,000 条](qwen38_max_20261020/answers.jsonl.gz) |
| [Qwen4-Max-Preview 前序检索测试](qwen4_max_preview_20261112/README.md) <br><sub>qwen4_max_preview_20261112</sub> | 请求关闭，未获回答 | 通道或参数诊断 | — |
| [Sonnet 5.5 请求关闭思考的前序检索测试](sonnet55_20261124/README.md) <br><sub>sonnet55_20261124</sub> | 请求关闭答题前思考（按响应筛选），`high` | 2,396 / 3,000 | [2,396 条](sonnet55_20261124/answers.jsonl.gz) |
| [Sonnet 5.5 思考开关对照核验](sonnet55_20261124/thinking_control_audit/README.md) <br><sub>sonnet55_20261124/thinking_control_audit</sub> | 请求关闭 `low`、`high`；开启 `high` | 通道或参数诊断 | — |
| [GPT-6.2 Sol / Astra：none 实测](gpt62_diagnostics/README.md) <br><sub>gpt62_diagnostics</sub> | `none` 未获回答；`low` 对照成功 | 通道或参数诊断 | — |
| [Fable 5 初版（RightAPI，全量）](fable5_rightapi_20261125/README.md) <br><sub>fable5_rightapi_20261125</sub> | 自适应，`low` | 1,273 / 3,000 | [1,273 条](fable5_rightapi_20261125/answers.jsonl.gz) |
| [Fable 5.1（RightAPI，全量）](fable51_rightapi_20261125/README.md) <br><sub>fable51_rightapi_20261125</sub> | 自适应，`low` | 1,849 / 3,000 | [1,849 条](fable51_rightapi_20261125/answers.jsonl.gz) |
| [Haiku 5.5（RightAPI，全量）](haiku55_rightapi_20261125/README.md) <br><sub>haiku55_rightapi_20261125</sub> | 自适应，`low` | 1,098 / 3,000 | [1,098 条](haiku55_rightapi_20261125/answers.jsonl.gz) |
| [Fable 5.2（RightAPI，全量）](fable52_rightapi_20261125/README.md) <br><sub>fable52_rightapi_20261125</sub> | 自适应，`low` | 2,753 / 3,000 | [2,753 条](fable52_rightapi_20261125/answers.jsonl.gz) |
| [Fable 5 关闭思考设置核验](fable5_20261103/README.md) <br><sub>fable5_20261103</sub> | 请求关闭，未设等级，返回思考块 | 通道或参数诊断 | — |
| [GPT-5.6 Sol 官方直连可用性检查](openai_direct_gpt56sol/README.md) <br><sub>openai_direct_gpt56sol</sub> | 请求 `none`，未获回答 | 通道或参数诊断 | — |
| [RightAPI 标称 Opus 4.8](opus48_rightapi/README.md) <br><sub>opus48_rightapi</sub> | 请求关闭，未设等级 | 406 / 600 | [406 条](opus48_rightapi/answers.jsonl.gz) |
| [Qwen3.7 Flash](qwen37_flash/README.md) <br><sub>qwen37_flash</sub> | 请求关闭（历史口径），未设等级 | 3,000 / 3,000 | [3,000 条](qwen37_flash/answers.jsonl.gz) |

## 补充 toy model 对照

包含训练至预定轮数的结果、提前停止的运行和历史设置；各页标明评估轮次。

- [卷积 + Softmax 注意力：算术与留出对子检索](toy/toy_conv_softmax_heldout_20261113/all_seeds_results.md)
- [卷积 + Softmax 注意力：算术与留出对子检索](toy/toy_conv_softmax_heldout_20261113/before_seeds104_110_20261113_105850/results.md)
- [卷积 + Softmax 注意力：算术与留出对子检索](toy/toy_conv_softmax_heldout_20261113/results.md)
- [toy_conv_softmax_heldout_20261113 · history.csv](toy/toy_conv_softmax_heldout_20261113/history.csv)
- [toy_conv_softmax_heldout_20261113 · history_all_seeds.csv](toy/toy_conv_softmax_heldout_20261113/history_all_seeds.csv)
- [高频共现模型：低层注意力路径遮挡](toy/toy_cooccurrence_attention_ablation_20261108/results.md)
- [同一记忆改变查询对象：EOS诊断](toy/toy_cooccurrence_len4_20261107/diagnostics/query_specificity/results.md)
- [普通 Transformer：固定长度 4 的高共现对照](toy/toy_cooccurrence_len4_20261107/results.md)
- [普通 Transformer：长度4高共现对照，辅助序列无EOS](toy/toy_cooccurrence_len4_noeos_20261107/results.md)
- [普通 Transformer：每轮9万共现，长度4，辅助序列无EOS](toy/toy_cooccurrence_len4_noeos_90k_20261107/results.md)
- [18 万条共现实验检查点：相同向量的线性探针](toy/toy_cooccurrence_long_20261107/linear_probe_20261107/results.md)
- [普通 Transformer：随机长序列共现 17,29（两倍频率）](toy/toy_cooccurrence_long_20261107/results.md)
- [普通 Transformer：100 倍共现 + 检索，去掉算术任务](toy/toy_cooccurrence_no_arithmetic_20261107/archive_all_180000_sequences/results.md)
- [普通 Transformer：高频共现 + 检索，去掉算术任务](toy/toy_cooccurrence_no_arithmetic_20261107/archive_before_100x/results.md)
- [普通 Transformer：仅含 17,29 的共现序列 + 检索](toy/toy_cooccurrence_no_arithmetic_20261107/results.md)

### 各次训练的保存记录

- [toy_conv_softmax_heldout_20261113](toy/toy_conv_softmax_heldout_20261113/)（30 个指标、训练历史或数值预测文件）
- [toy_cooccurrence_attention_ablation_20261108](toy/toy_cooccurrence_attention_ablation_20261108/)（13 个指标、训练历史或数值预测文件）
- [toy_cooccurrence_len4_20261107](toy/toy_cooccurrence_len4_20261107/)（7 个指标、训练历史或数值预测文件）
- [toy_cooccurrence_len4_noeos_20261107](toy/toy_cooccurrence_len4_noeos_20261107/)（9 个指标、训练历史或数值预测文件）
- [toy_cooccurrence_len4_noeos_90k_20261107](toy/toy_cooccurrence_len4_noeos_90k_20261107/)（9 个指标、训练历史或数值预测文件）
- [toy_cooccurrence_long_20261107](toy/toy_cooccurrence_long_20261107/)（15 个指标、训练历史或数值预测文件）
- [toy_cooccurrence_no_arithmetic_20261107](toy/toy_cooccurrence_no_arithmetic_20261107/)（12 个指标、训练历史或数值预测文件）
- [toy_linear_probe_20261107/attention_mlp_no_embedding](toy/toy_linear_probe_20261107/attention_mlp_no_embedding/)（4 个指标、训练历史或数值预测文件）
- [toy_linear_probe_20261107/attention_only](toy/toy_linear_probe_20261107/attention_only/)（4 个指标、训练历史或数值预测文件）
- [toy_linear_probe_20261107/mlp_only_paired_xgp29af8](toy/toy_linear_probe_20261107/mlp_only_paired_xgp29af8/)（4 个指标、训练历史或数值预测文件）
