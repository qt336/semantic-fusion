# GLM-5.4 前序检索可用性核验

2026 年 11 月 24 日。已检查三个本机配置通道的模型目录，在腾讯 TokenHub 和阿里云百炼分别发送了同一道原题。有效模型回答为 0，180 题基准标为未测。

| 渠道 | 目录列出 GLM-5.4 | 单题调用 |
|---|---|---|
| 腾讯 TokenHub | 否 | HTTP 400，模型或服务 ID 不存在 |
| 阿里云百炼 | 否 | HTTP 404，模型不存在或账户无权访问 |
| 本机另一个 Messages 通道 | 否 | 只检查目录 |

腾讯使用 `thinking.type=disabled`，百炼使用 `enable_thinking=false`。请求的模型 ID 均为 `glm-5.4`。服务在模型选择阶段即拒绝请求，无法据此判断是否支持关闭思考。

智谱发布公告 将通用 API 列为 Coming Soon，已列出的体验入口包括 GLM Coding Plan、ZCode 和 Z.ai Chat。ZCode 官方说明 列出 Low、High、Max 三档思考强度。
