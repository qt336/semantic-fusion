# GPT-6.2 Sol / Astra：none 实测

本页记录 API 可用性或参数诊断。基准成绩和纳入数见[统一结果表](../README.md)。

> **2026-11-24 上下文核验更新：** 已确认转接通道存在三条重复/模板输入并改写历史、推理上下文和缓存相关参数。六轮随机标记测试未检测到前题串入，显式历史正对照成功；附加文本仍无法读取。这批结果保留为通道观察，暂不视为上下文已受控的无思维链实验。详见 完整核验报告。

使用本机已配置的 RightAPI GPT 通道。以下是实际请求结果，不是仅凭模型列表推断。

| 模型 | Responses：none | Chat Completions：none | Responses：low 对照 |
|---|---|---|---|
| GPT-6.2 Sol | 中文、英文各一次，均 HTTP 400 | HTTP 400，明确提示不支持 none | HTTP 200，回答 cream，返回 effort=low |
| GPT-6.2 Astra | 中文、英文各一次，均 HTTP 400 | HTTP 400，明确提示不支持 none | HTTP 200，回答 cream，返回 effort=low |

Responses 接口返回通用错误 `Upstream request failed`；Chat Completions 的具体错误如下：



英文题目：



中文题目：



两次 low 对照都返回 `reasoning_tokens=0`。这只说明这两次调用的接口统计为零，不表示 low 是关闭推理的模式，也不保证其他题目的推理计数为零。未将 low 结果混入 none 测试。

结论：在本次通道实测中，GPT-6.2 Sol 和 Astra 都不能通过 none 固定关闭推理；模型本身可以通过 low 正常连接。结果与 [OpenAI 官方推理文档](https://developers.openai.com/api/docs/guides/reasoning) 的支持范围一致。

通道仍存在附加输入上下文；这些诊断不构成官方直连或底层模型权重的独立验证。

原始请求和响应位于本目录的 `sol62_*.json`、`astra62_*.json`，固定请求清单为 `gpt62_none_jobs.json` 和 `gpt62_controls_jobs.json`。上述参数支持范围检查共 8 次调用，无重试。

## 补充：相同 low 档位下，更换任务并重复两次

为了检查 low 报告零是否意味着计数始终填零，两个模型分别在同一 low 档位、同一请求输出上限 1,024 下，回答检索题和多位数乘法，每种题重复两次。客户端均不提供工具、system 或对话历史。仍使用同一个 RightAPI 通道。

| 模型 | 检索题推理 token（两次） | 乘法题推理 token（两次） | 返回档位 |
|---|---|---|---|
| GPT-6.2 Sol | 0、0 | 183、123 | 全部 low |
| GPT-6.2 Astra | 0、0 | 137、136 | 全部 low |

检索题是上文的 `ice cream lake`，四次最终答案都是 `cream`。乘法题为：



四次最终答案均为 `46714782115586533`，经本地整数乘法核对正确。检索响应只含 message 输出项；乘法响应含 reasoning 与 message 输出项。所有八次调用均 HTTP 200、明确给出推理 token 统计，返回档位为 low，无重试。

这项对照排除了“该通道的 reasoning_tokens 在这次测试中一律填 0”，并显示同一个 low 档位下统计会随任务变化。它与官方描述的自适应推理用量相容，但不能独立证明转接平台没有省略其他上游步骤，也不能把 low 当作可控制的关闭模式。此前发现的额外输入上下文仍未核实。

原始记录为 `lowcheck_*.json`，预先固定的八条请求清单位于 `gpt62_low_adaptive_jobs.json`。包括本补充在内，共进行了 16 次 GPT-6.2 调用。
