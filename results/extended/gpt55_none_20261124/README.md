# GPT-5.5：关闭思考连通性检查

本页记录 API 可用性或参数诊断。基准成绩和纳入数见[统一结果表](../README.md)。

检查日期：2026-11-24。使用本机已经配置的 RightAPI GPT 通道。

官方 [GPT-5.5 模型文档](https://developers.openai.com/api/docs/models/gpt-5.5) 明确支持 `reasoning.effort: none`，并列出具体版本 `gpt-5.5-2026-04-23`。

## 实际请求

| 模型 | 接口 | 提示词 | HTTP | 错误 |
|---|---|---|---|---|
| gpt-5.5 | chat | ice cream lake. What word comes immediately before "lake"? Output only one word. | 404 | model not found |
| gpt-5.5 | responses | ice cream lake. What word comes immediately before "lake"? Output only one word. | 404 | model not found |
| gpt-5.5 | responses | 猜中远。直接问“远”的前一个字是什么？只输出一个字即可。 | 404 | model not found |
| gpt-5.5-2026-04-23 | responses | ice cream lake. What word comes immediately before "lake"? Output only one word. | 404 | model not found |

全部请求显式设置 `none`，输出上限为 128。客户端未提供 system、工具或对话历史。请求及原始响应保存在同目录的 `connect_*.json`。

## 可用性结论

刷新 `/models` 返回 HTTP 200，但列表中没有 GPT-5.5 或其具体版本。现有列表为：`codex-auto-review`, `gpt-5.6-sol`, `gpt-6-sol`, `gpt-6.1-terra`, `gpt-6.1-sol`, `gpt-6-astra`, `gpt-6.2-sol`, `gpt-6.2-astra`。

模型别名和官方具体版本名均返回 `404 model not found`。这是当前通道的模型可用性问题，不能据此判定 GPT-5.5 不支持关闭思考。

本轮没有成功生成答案，无法计算检索准确率，也没有可用的推理 token 统计；未将失败请求混入正式 benchmark。环境变量及现有 Hermes 配置中未找到可用于官方直连的 `OPENAI_API_KEY`。
