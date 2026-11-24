# 扩展实验结果

截至 2026-11-25。每个实验页面保留测试条件、完成状态和原有评分口径。逐题文件保存提示词、目标、可见回答和已归档评分。公开接口返回的模型名和推理计数按实验记录解释。

[所有扩展实验的分条件计数 CSV](conditions.csv) · [论文三模型的原始结果与评分](../llm/) · [Qwen3-8B 遮挡实验](../../experiments/qwen3_8b/) · [论文与 toy model 结果总索引](../../RESULTS.md)

## 中英文结果速查

每格为百分比，依次是 **答对第二成分 / 误答首成分**。各条件分母及完整错误分布见运行页面。每行列出客户端请求的思考模式和等级。「未设」表示没有单独指定等级。`minimal` 表示仍允许推理的最低档，即使接口报告 0 个推理 token，也按该档登记。

Opus 5 和 Sonnet 5.5 主表均按各自的关闭设置筛选，各排除 1 条出现思考的回复。Opus 5.5 开启自适应思考，全部 3,000 条均纳入。各模型的等级沿用各自 API 的定义，中转通道的记录只能核实请求及返回字段。

**哪些模型整轮都报告了 0 个推理 token？** 在主表的 3,000 题测试中，Gemini 3.6 Flash、Gemma 4 31B IT、Kimi K3、HY4 preview、GPT-5、GPT-5.1、GPT-5.4、GPT-5.5、GPT-5.6 Luna / Sol / Terra 和 GPT-6.1 Terra 均为 3,000/3,000。Gemini 3.6 Flash 和 GPT-5 请求的都是 `minimal`。Gemini 未返回可见思考内容。GPT-5 的每条响应都带有加密的 reasoning 项，公开正文和摘要为空。

“推理 token 计数”按纳入评分的回复统计。混合结果依次列出**明确报 0 / 明确报正数 / 未报告**的题数。单一结果在括号内给出题数。缺失或空值单列。计数取自接口或历史逐题记录，服务端未计入这些字段的计算无法核验。

### 中文

| 模型 | 思考模式（请求） | 等级 | 推理 token 计数（本语言） | 语义 | 独立随机 | 语义加逗号 |
|---|---|---|---|---:|---:|---:|
| [claude-opus-4-8](opus48_rightapi/README.md) | 请求关闭 | 未设 | 199 / 0 / 101 | 70.0 / 26.0 | 93.0 / 7.0 | 100.0 / 0.0 |
| [claude-opus-5](opus5_20261103/README.md) | 请求关闭（按响应筛选） | `low` | 1,175 / 0 / 325 | 71.8 / 21.4 | 91.8 / 7.0 | 95.2 / 2.8 |
| [claude-opus-5-5](opus55_20261124/README.md) | 开启（自适应） | `low` | 936 / 266 / 298 | 93.6 / 2.0 | 98.0 / 0.8 | 99.4 / 0.0 |
| [claude-sonnet-5-5](sonnet55_20261124/README.md) | 请求关闭答题前思考（按响应筛选） | `high` | 1,197 / 0 / 303 | 79.2 / 14.8 | 98.2 / 1.6 | 99.4 / 0.2 |
| [DeepSeek V4 Pro](../llm/) | 请求关闭 | 未设 | 全未报告（1,500） | 31.8 / 67.4 | 66.0 / 34.0 | 94.2 / 5.6 |
| [DeepSeek V4 Pro（128/180）](deepseek_v4pro_thinking_20261124/README.md) | 开启 | `high` | 全报正数（68） | 100.0 / 0.0 | 100.0 / 0.0 | 100.0 / 0.0 |
| [DeepSeek V4.1 Flash 1017（180 题）](deepseek_v41flash_thinking_20261124/README.md) | 开启 | `high` | 全报正数（90） | 93.3 / 0.0 | 100.0 / 0.0 | 100.0 / 0.0 |
| [deepseek-v4.1-flash-1017](deepseek_v41_flash_20261020/README.md) | 请求关闭 | 未设 | 全未报告（1,500） | 58.4 / 38.2 | 81.2 / 18.6 | 100.0 / 0.0 |
| [gemini-3.6-flash](gemma4_gemini36_zenmux_20261124/full3000/README.md) | 最低推理档 | `minimal` | **全报 0**（1,500） | 87.2 / 10.8 | 97.4 / 0.2 | 100.0 / 0.0 |
| [gemma-4-31b-it](gemma4_gemini36_zenmux_20261124/full3000/README.md) | 请求关闭（按响应筛选） | 未设 | **全报 0**（1,500） | 79.4 / 18.4 | 98.8 / 1.0 | 100.0 / 0.0 |
| [hy4（暂定评分）](hy4_20261104/README.md) | 请求关闭 | 未设 | 1,498 / 0 / 2 | 54.4 / 44.6 | 90.4 / 9.6 | 99.6 / 0.2 |
| [HY4 preview](../llm/) | 请求关闭 | 未设 | **全报 0**（1,500） | 55.6 / 40.8 | 91.0 / 8.8 | 99.8 / 0.0 |
| [Kimi K3](../llm/) | 请求关闭 | `none` | **全报 0**（1,500） | 76.4 / 23.4 | 97.4 / 2.6 | 100.0 / 0.0 |
| [MiniMax/MiniMax-M3](minimax_m3_20261020/README.md) | 请求关闭 | 未设 | 484 / 0 / 1,016 | 20.6 / 38.8 | 46.2 / 38.2 | 59.2 / 27.8 |
| [gpt-5](gpt5_minimal_full_20261125/README.md) | 最低推理档 | `minimal` | **全报 0**（1,500） | 68.4 / 18.2 | 95.6 / 2.6 | 99.0 / 0.0 |
| [gpt-5.1](gpt51_none_full_20261125/README.md) | 请求关闭 | `none` | **全报 0**（1,500） | 76.0 / 16.0 | 97.8 / 2.0 | 100.0 / 0.0 |
| [gpt-5.4](gpt54_none_full_20261125/README.md) | 请求关闭 | `none` | **全报 0**（1,500） | 96.0 / 3.8 | 99.8 / 0.2 | 100.0 / 0.0 |
| [gpt-5.5](gpt55_none_full_20261125/README.md) | 请求关闭 | `none` | **全报 0**（1,500） | 99.4 / 0.4 | 100.0 / 0.0 | 100.0 / 0.0 |
| [gpt-5.6-luna](gpt56luna_none_full_20261125/README.md) | 请求关闭 | `none` | **全报 0**（1,500） | 76.4 / 20.0 | 99.0 / 1.0 | 99.8 / 0.2 |
| [gpt-5.6-sol](gpt56sol_none_full_20261125/README.md) | 请求关闭 | `none` | **全报 0**（1,500） | 98.2 / 1.6 | 99.6 / 0.0 | 100.0 / 0.0 |
| [gpt-5.6-terra](gpt56terra_none_full_20261125/README.md) | 请求关闭 | `none` | **全报 0**（1,500） | 80.0 / 18.6 | 99.4 / 0.6 | 100.0 / 0.0 |
| [gpt-6.1-terra](gpt61terra_zenmux_20261124/README.md) | 请求关闭 | `none` | **全报 0**（1,500） | 99.6 / 0.2 | 99.8 / 0.0 | 99.8 / 0.0 |
| [qwen3.7-flash](qwen37_flash/README.md) | 请求关闭 | 未设 | 全未报告（1,500） | 79.4 / 6.6 | 94.6 / 1.6 | 100.0 / 0.0 |
| [qwen3.8-max-0902](qwen38_max_20261020/README.md) | 请求关闭 | 未设 | 全未报告（1,500） | 89.2 / 7.4 | 96.8 / 2.6 | 99.0 / 0.8 |

### 英文

| 模型 | 思考模式（请求） | 等级 | 推理 token 计数（本语言） | 语义 | 独立随机 | 语义加逗号 |
|---|---|---|---|---:|---:|---:|
| [claude-opus-4-8](opus48_rightapi/README.md) | 请求关闭 | 未设 | 207 / 0 / 93 | 38.0 / 47.0 | 75.0 / 21.0 | 82.0 / 10.0 |
| [claude-opus-5](opus5_20261103/README.md) | 请求关闭（按响应筛选） | `low` | 1,185 / 0 / 314 | 69.7 / 26.5 | 86.6 / 11.0 | 79.4 / 13.2 |
| [claude-opus-5-5](opus55_20261124/README.md) | 开启（自适应） | `low` | 593 / 629 / 278 | 95.0 / 3.4 | 96.4 / 2.6 | 95.8 / 0.6 |
| [claude-sonnet-5-5](sonnet55_20261124/README.md) | 请求关闭答题前思考（按响应筛选） | `high` | 1,199 / 0 / 300 | 94.2 / 3.6 | 96.0 / 2.6 | 94.6 / 0.8 |
| [DeepSeek V4 Pro](../llm/) | 请求关闭 | 未设 | 全未报告（1,500） | 33.6 / 32.8 | 75.0 / 11.4 | 55.2 / 12.0 |
| [DeepSeek V4 Pro（128/180）](deepseek_v4pro_thinking_20261124/README.md) | 开启 | `high` | 全报正数（60） | 100.0 / 0.0 | 100.0 / 0.0 | 87.5 / 0.0 |
| [DeepSeek V4.1 Flash 1017（180 题）](deepseek_v41flash_thinking_20261124/README.md) | 开启 | `high` | 全报正数（90） | 100.0 / 0.0 | 100.0 / 0.0 | 100.0 / 0.0 |
| [deepseek-v4.1-flash-1017](deepseek_v41_flash_20261020/README.md) | 请求关闭 | 未设 | 全未报告（1,500） | 25.6 / 74.2 | 80.8 / 17.8 | 75.4 / 19.0 |
| [gemini-3.6-flash](gemma4_gemini36_zenmux_20261124/full3000/README.md) | 最低推理档 | `minimal` | **全报 0**（1,500） | 54.4 / 6.6 | 82.4 / 3.4 | 64.2 / 13.6 |
| [gemma-4-31b-it](gemma4_gemini36_zenmux_20261124/full3000/README.md) | 请求关闭（按响应筛选） | 未设 | **全报 0**（1,500） | 56.2 / 18.2 | 83.4 / 6.4 | 72.6 / 3.2 |
| [hy4（暂定评分）](hy4_20261104/README.md) | 请求关闭 | 未设 | **全报 0**（1,500） | 36.6 / 63.0 | 94.6 / 4.8 | 100.0 / 0.0 |
| [HY4 preview](../llm/) | 请求关闭 | 未设 | **全报 0**（1,500） | 65.8 / 30.8 | 89.0 / 7.8 | 93.2 / 6.0 |
| [Kimi K3](../llm/) | 请求关闭 | `none` | **全报 0**（1,500） | 83.8 / 12.6 | 94.4 / 5.0 | 100.0 / 0.0 |
| [MiniMax/MiniMax-M3](minimax_m3_20261020/README.md) | 请求关闭 | 未设 | 477 / 0 / 1,023 | 37.6 / 58.2 | 62.2 / 24.8 | 55.6 / 32.6 |
| [gpt-5](gpt5_minimal_full_20261125/README.md) | 最低推理档 | `minimal` | **全报 0**（1,500） | 79.4 / 16.0 | 96.0 / 3.2 | 89.6 / 2.0 |
| [gpt-5.1](gpt51_none_full_20261125/README.md) | 请求关闭 | `none` | **全报 0**（1,500） | 79.4 / 13.0 | 96.8 / 1.6 | 77.8 / 15.8 |
| [gpt-5.4](gpt54_none_full_20261125/README.md) | 请求关闭 | `none` | **全报 0**（1,500） | 88.8 / 3.6 | 99.6 / 0.2 | 44.8 / 3.0 |
| [gpt-5.5](gpt55_none_full_20261125/README.md) | 请求关闭 | `none` | **全报 0**（1,500） | 99.0 / 1.0 | 99.4 / 0.0 | 88.2 / 4.4 |
| [gpt-5.6-luna](gpt56luna_none_full_20261125/README.md) | 请求关闭 | `none` | **全报 0**（1,500） | 43.4 / 26.0 | 81.2 / 10.2 | 9.6 / 9.4 |
| [gpt-5.6-sol](gpt56sol_none_full_20261125/README.md) | 请求关闭 | `none` | **全报 0**（1,500） | 99.2 / 0.2 | 99.6 / 0.2 | 99.8 / 0.0 |
| [gpt-5.6-terra](gpt56terra_none_full_20261125/README.md) | 请求关闭 | `none` | **全报 0**（1,500） | 64.2 / 28.0 | 98.0 / 1.0 | 69.6 / 5.2 |
| [gpt-6.1-terra](gpt61terra_zenmux_20261124/README.md) | 请求关闭 | `none` | **全报 0**（1,500） | 100.0 / 0.0 | 99.6 / 0.4 | 100.0 / 0.0 |
| [qwen3.7-flash](qwen37_flash/README.md) | 请求关闭 | 未设 | 全未报告（1,500） | 37.6 / 56.6 | 81.4 / 6.2 | 58.2 / 33.4 |
| [qwen3.8-max-0902](qwen38_max_20261020/README.md) | 请求关闭 | 未设 | 全未报告（1,500） | 47.4 / 52.2 | 93.6 / 5.6 | 96.6 / 3.0 |

## 模型测试与对照

独立小测和全量实验可能复用相同回答，跨行相加会重复计数。逐题记录数可包含排除项；正确率分母以相应结果表为准。本表计数合并中英文。两个模型共用一页时分别列出，连通和参数诊断见各自记录。

| 实验 | 本轮思考设置 | 推理 token 计数（全组） | 保存的逐题记录 |
|---|---|---|---:|
| [DeepSeek V4.1 Flash 前序检索测试](deepseek_v41_flash_20261020/README.md) <br><sub>deepseek_v41_flash_20261020</sub> | 请求关闭，未设等级 | 全未报告（3,000） | [3,000](deepseek_v41_flash_20261020/answers.jsonl.gz) |
| [DeepSeek V4.1 Flash 1017 开启思考的前序检索测试](deepseek_v41flash_thinking_20261124/README.md) <br><sub>deepseek_v41flash_thinking_20261124</sub> | 开启，`high` | 全报正数（180） | [180](deepseek_v41flash_thinking_20261124/answers.jsonl.gz) |
| [DeepSeek V4 Pro 开启思考的前序检索测试](deepseek_v4pro_thinking_20261124/README.md) <br><sub>deepseek_v4pro_thinking_20261124</sub> | 开启，`high` | 全报正数（128） | [128](deepseek_v4pro_thinking_20261124/answers.jsonl.gz) |
| [Gemma 4 31B IT 与 Gemini 3.6 Flash 的前序检索](gemma4_gemini36_zenmux_20261124/fixed180/README.md) <br><sub>gemma4_gemini36_zenmux_20261124/fixed180</sub> | Gemma 请求关闭，未设等级；Gemini `minimal` | gemini-3.6-flash **全报 0**（180）<br>gemma-4-31b-it **全报 0**（180） | [360](gemma4_gemini36_zenmux_20261124/fixed180/answers.jsonl.gz) |
| [Gemma 4 31B IT 与 Gemini 3.6 Flash 的前序检索](gemma4_gemini36_zenmux_20261124/full3000/README.md) <br><sub>gemma4_gemini36_zenmux_20261124/full3000</sub> | Gemma 请求关闭，未设等级；Gemini `minimal` | gemini-3.6-flash **全报 0**（3,000）<br>gemma-4-31b-it **全报 0**（3,000） | [6,000](gemma4_gemini36_zenmux_20261124/full3000/answers.jsonl.gz) |
| [Gemma 4 31B IT 与 Gemini 3.6 Flash 单题对照](gemma4_gemini36_zenmux_20261124/README.md) <br><sub>gemma4_gemini36_zenmux_20261124</sub> | Gemma 请求关闭，未设等级；Gemini `minimal` | 见诊断记录 | 见报告 |
| [GLM-5.4 前序检索可用性核验](glm54_20261124/README.md) <br><sub>glm54_20261124</sub> | 请求关闭，未获回答 | 见诊断记录 | 见报告 |
| [GPT-5.1 中英文前序检索测试](gpt51_none_full_20261125/README.md) <br><sub>gpt51_none_full_20261125</sub> | 请求关闭，`none` | **全报 0**（3,000） | [3,000](gpt51_none_full_20261125/answers.jsonl.gz) |
| [GPT-5.1 中英文前序检索测试](gpt51_none_zenmux_20261125/README.md) <br><sub>gpt51_none_zenmux_20261125</sub> | 请求关闭，`none` | **全报 0**（180） | [180](gpt51_none_zenmux_20261125/answers.jsonl.gz) |
| [GPT-5.4 中英文前序检索测试](gpt54_none_full_20261125/README.md) <br><sub>gpt54_none_full_20261125</sub> | 请求关闭，`none` | **全报 0**（3,000） | [3,000](gpt54_none_full_20261125/answers.jsonl.gz) |
| [GPT-5.5：关闭思考连通性检查](gpt55_none_20261124/README.md) <br><sub>gpt55_none_20261124</sub> | 请求 `none`，未获回答 | 见诊断记录 | 见报告 |
| [GPT-5.5 中英文前序检索测试](gpt55_none_full_20261125/README.md) <br><sub>gpt55_none_full_20261125</sub> | 请求关闭，`none` | **全报 0**（3,000） | [3,000](gpt55_none_full_20261125/answers.jsonl.gz) |
| [GPT-5.6 Luna 中英文前序检索测试](gpt56luna_none_full_20261125/README.md) <br><sub>gpt56luna_none_full_20261125</sub> | 请求关闭，`none` | **全报 0**（3,000） | [3,000](gpt56luna_none_full_20261125/answers.jsonl.gz) |
| [GPT-5.6 Luna 中英文前序检索测试](gpt56luna_zenmux_20261124/README.md) <br><sub>gpt56luna_zenmux_20261124</sub> | 请求关闭，`none` | **全报 0**（180） | [180](gpt56luna_zenmux_20261124/answers.jsonl.gz) |
| [GPT-5.6 Sol 在其他模型错答首成分的题目上的表现](gpt56sol_hard_pairs_20261124/README.md) <br><sub>gpt56sol_hard_pairs_20261124</sub> | 请求关闭，`none` | **全报 0**（24） | [24](gpt56sol_hard_pairs_20261124/answers.jsonl.gz) |
| [GPT-5.6 Sol：请求关闭推理的小样本测试](gpt56sol_none_20261124/README.md) <br><sub>gpt56sol_none_20261124</sub> | 主测 `none`；另有档位诊断 | **全报 0**（180） | [180](gpt56sol_none_20261124/answers.jsonl.gz) |
| [GPT-5.6 Sol 中英文前序检索测试](gpt56sol_none_full_20261125/README.md) <br><sub>gpt56sol_none_full_20261125</sub> | 请求关闭，`none` | **全报 0**（3,000） | [3,000](gpt56sol_none_full_20261125/answers.jsonl.gz) |
| [GPT-5.6 Sol 位置对照](gpt56sol_position_control_20261124/README.md) <br><sub>gpt56sol_position_control_20261124</sub> | 请求关闭，`none` | **全报 0**（24） | [24](gpt56sol_position_control_20261124/answers.jsonl.gz) |
| [GPT-5.6 Terra 中英文前序检索测试](gpt56terra_none_full_20261125/README.md) <br><sub>gpt56terra_none_full_20261125</sub> | 请求关闭，`none` | **全报 0**（3,000） | [3,000](gpt56terra_none_full_20261125/answers.jsonl.gz) |
| [GPT-5 中英文前序检索测试](gpt5_minimal_full_20261125/README.md) <br><sub>gpt5_minimal_full_20261125</sub> | 最低推理档，`minimal` | **全报 0**（3,000） | [3,000](gpt5_minimal_full_20261125/answers.jsonl.gz) |
| [GPT-5 中英文前序检索测试](gpt5_zenmux_20261125/README.md) <br><sub>gpt5_zenmux_20261125</sub> | 最低推理档，`minimal` | **全报 0**（180） | [180](gpt5_zenmux_20261125/answers.jsonl.gz) |
| [GPT-6.1 Terra 中英文前序检索测试](gpt61terra_zenmux_20261124/README.md) <br><sub>gpt61terra_zenmux_20261124</sub> | 请求关闭，`none` | **全报 0**（3,000） | [3,000](gpt61terra_zenmux_20261124/answers.jsonl.gz) |
| [GPT-6 Luna 中英文前序检索测试](gpt6luna_zenmux_20261124/README.md) <br><sub>gpt6luna_zenmux_20261124</sub> | 请求关闭，`none` | **全报 0**（180） | [180](gpt6luna_zenmux_20261124/answers.jsonl.gz) |
| [GPT 转接通道：额外输入与请求隔离核验](gpt_channel_context_audit_20261124/README.md) <br><sub>gpt_channel_context_audit_20261124</sub> | 通道诊断，主请求 `none` | 见诊断记录 | 见报告 |
| [GPT 关闭推理通道试测](gpt_none_20261124/README.md) <br><sub>gpt_none_20261124</sub> | 主测 `none`；另有 `high` 诊断 | gpt-6-sol **全报 0**（180）<br>gpt-6.1-terra **全报 0**（180） | [360](gpt_none_20261124/answers.jsonl.gz) |
| [Grok 4.3 前序检索测试](grok43_none_20261124/README.md) <br><sub>grok43_none_20261124</sub> | 请求 `none`，未获回答 | 见诊断记录 | 见报告 |
| [HY4 正式版前序检索测试](hy4_20261104/README.md) <br><sub>hy4_20261104</sub> | 请求关闭，未设等级 | 2,998 / 0 / 2 | [3,000](hy4_20261104/answers.jsonl.gz) |
| [MiniMax M3 前序检索测试](minimax_m3_20261020/README.md) <br><sub>minimax_m3_20261020</sub> | 请求关闭，未设等级 | 961 / 0 / 2,039 | [3,000](minimax_m3_20261020/answers.jsonl.gz) |
| [可访问的无思考模型核验](no_thinking_model_audit_20261124/README.md) <br><sub>no_thinking_model_audit_20261124</sub> | 多种关闭设置，见各模型记录 | 见诊断记录 | 见报告 |
| [Opus 5.5 low 前序检索测试](opus55_20261124/README.md) <br><sub>opus55_20261124</sub> | 开启（自适应），`low` | 1,529 / 895 / 576 | [3,000](opus55_20261124/answers.jsonl.gz) |
| [Opus 5 前序检索测试](opus5_20261103/high_pilot/README.md) <br><sub>opus5_20261103/high_pilot</sub> | 请求关闭（按响应筛选），`high` | 91 / 0 / 28 | [120](opus5_20261103/high_pilot/answers.jsonl.gz) |
| [Opus 5 前序检索测试](opus5_20261103/README.md) <br><sub>opus5_20261103</sub> | 请求关闭（按响应筛选），`low` | 2,360 / 0 / 639 | [3,000](opus5_20261103/answers.jsonl.gz) |
| [Qwen3.8 Max-0902 前序检索测试](qwen38_max_20261020/README.md) <br><sub>qwen38_max_20261020</sub> | 请求关闭，未设等级 | 全未报告（3,000） | [3,000](qwen38_max_20261020/answers.jsonl.gz) |
| [Qwen4-Max-Preview 前序检索测试](qwen4_max_preview_20261112/README.md) <br><sub>qwen4_max_preview_20261112</sub> | 请求关闭，未获回答 | 见诊断记录 | 见报告 |
| [Sonnet 5.5 请求关闭思考的前序检索测试](sonnet55_20261124/README.md) <br><sub>sonnet55_20261124</sub> | 请求关闭答题前思考（按响应筛选），`high` | 2,396 / 0 / 603 | [3,000](sonnet55_20261124/answers.jsonl.gz) |
| [Sonnet 5.5 思考开关对照核验](sonnet55_20261124/thinking_control_audit/README.md) <br><sub>sonnet55_20261124/thinking_control_audit</sub> | 请求关闭 `low`、`high`；开启 `high` | 见诊断记录 | 见报告 |
| [GPT-6.2 Sol / Astra：none 实测](gpt62_diagnostics/README.md) <br><sub>gpt62_diagnostics</sub> | `none` 未获回答；`low` 对照成功 | 见诊断记录 | 见报告 |
| [Fable 5 关闭思考设置核验](fable5_20261103/README.md) <br><sub>fable5_20261103</sub> | 请求关闭，未设等级，返回思考块 | 见诊断记录 | 见报告 |
| [GPT-5.6 Sol 官方直连可用性检查](openai_direct_gpt56sol/README.md) <br><sub>openai_direct_gpt56sol</sub> | 请求 `none`，未获回答 | 见诊断记录 | 见报告 |
| [RightAPI 标称 Opus 4.8](opus48_rightapi/README.md) <br><sub>opus48_rightapi</sub> | 请求关闭，未设等级 | 406 / 0 / 194 | [600](opus48_rightapi/answers.jsonl.gz) |
| [Qwen3.7 Flash](qwen37_flash/README.md) <br><sub>qwen37_flash</sub> | 请求关闭，未设等级 | 全未报告（3,000） | [3,000](qwen37_flash/answers.jsonl.gz) |

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
