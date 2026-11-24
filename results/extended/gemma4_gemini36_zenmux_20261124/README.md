# Gemma 4 31B IT 与 Gemini 3.6 Flash 单题对照

每个模型使用相同的中文、英文无逗号语义题各一条，取自此前 180 条固定题单中相应条件的首题。每条独立请求只有一条 user 消息。通过 ZenMux Chat Completions 调用，Gemma 请求 reasoning.enabled=false，Gemini 请求 reasoning_effort=minimal。推理 token 来自接口 usage 字段。该小样未单独验证模型分词边界。

| 型号 | 题号 | 目标 | 原始回答 | 输入 token | 输出 token | 推理 token |
|---|---|---|---|---:|---:|---:|
| google/gemini-3.6-flash | zh:semantic_474 | 完 | 刀 | 20 | 1 | 0 |
| google/gemini-3.6-flash | en:strong:semantic_433 | reaction | wooden | 17 | 1 | 0 |
| google/gemma-4-31b-it | zh:semantic_474 | 完 | 画 | 33 | 2 | 0 |
| google/gemma-4-31b-it | en:strong:semantic_433 | reaction | reaction | 30 | 2 | 0 |

## 原始提示词

- 画完刀。直接问“刀”的前一个字是什么？只输出一个字即可。
- chain reaction spoon. What word comes immediately before "spoon"? Output only one word.

四条成功响应合计报告费用为 0.00008092 美元。每条请求的设置、原始回答和用量见同目录 JSON。
