# Grok 4.3 前序检索测试

2026-11-24。连通性检查未取得模型回答，固定 180 题基准未启动。

试发原题：ice age door. What word comes immediately before "door"? Output only one word.

| 已配置入口 | 结果 |
|---|---|
| rightapi_unified | HTTP 404，model not found |
| rightapi_codex | HTTP 404，model not found |
| dashscope | HTTP 404，The model `grok-4.3` does not exist or you do not have access to it. |

Responses 请求显式指定 reasoning.effort=none，Chat Completions 使用 reasoning_effort=none。当前环境未配置 xAI 官方 API 密钥。

xAI 模型说明 · RightAPI 渠道与密钥
