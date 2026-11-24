# 可访问的无思考模型核验

核验日期：2026-11-24。范围：开发机已有的 RightAPI 和 DashScope API 配置。每个候选仅做中文、英文各一次连接与参数检查，没有启动完整基准测试。

| 模型 | 通道 | 请求设置 | 两次调用结果 | 思考计数 |
| --- | --- | --- | --- | --- |
| Claude Sonnet 5.5 | RightAPI | `thinking.type=between_tools`, `effort=high`，不提供工具 | 均 HTTP 200 | 均无 thinking block，`thinking_tokens=0` |
| Claude Opus 5 | RightAPI | `thinking.type=disabled`, `effort=high` | 均 HTTP 200 | 均无 thinking block，`thinking_tokens=0` |
| Qwen4-Max | DashScope | `enable_thinking=false` | 均 HTTP 403，`AccessDenied.Unpurchased` | 无模型回答 |
| Kimi K3.1 Pro | DashScope | `enable_thinking=false` | 均 HTTP 403，`AccessDenied.Unpurchased` | 无模型回答 |

## 能力与设置的区别

Sonnet 5.5 是本次发现的、值得补测的高能力候选。官方文档说明 `between_tools` 关闭回答前的思考；不调用工具时只返回文本。它不等同于在工具调用场景下完全关闭所有思考。Opus 5 则直接支持 `disabled`，但 effort 必须为 high 或更低。

目前没有查到这两个模型在关闭思考条件下足以判定综合能力高低的可比评测。Artificial Analysis 当前页面给出 Sonnet 5.5（Max，默认 fallback）55 分、Opus 5（Max）44 分；这两项开启思考，不能用来证明无思考条件下 Sonnet 5.5 更强。两条简单任务的输出也不足以比较模型能力。

Opus 5.5 和 Fable 5 系列不支持关闭思考，不能因设置 low 或未返回可见思考文本而归入无思考模型。

## 实际回答

英文请求：`ice cream lake. What word comes immediately before "lake"? Output only one word.`

中文请求：`猜中远。直接问“远”的前一个字是什么？只输出一个字即可。`

| 模型 | 英文原始回答 | 中文原始回答 |
| --- | --- | --- |
| Claude Sonnet 5.5 | cream | 不清楚你说的“远”出现在哪个词或句子里，所以没法确定前一个字。如果是指某个具体的词（比如“遥远”“长远”），把它发给我，我再回答。 |
| Claude Opus 5 | ice | 永 |

以上检查验证的是接口可调用与返回的思考计数，并不独立证明第三方通道的底层模型身份、内部完整输入或实际内部计算。没有验证这些模型的三 token 分词对齐。

## 来源与原始记录

- Claude 思考设置
- Sonnet 5.5 设置变更
- Sonnet 5.5 官方发布说明
- Artificial Analysis：Sonnet 5.5
- Artificial Analysis：Opus 5
- 同目录 `*_en.json`、`*_zh.json` 保存实际请求与原始响应；目录清单保存在 `rightapi_models.json` 和 `dashscope_models.json`。
