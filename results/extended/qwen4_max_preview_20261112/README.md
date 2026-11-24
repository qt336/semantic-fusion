# Qwen4-Max-Preview 前序检索测试

更新：2026-11-12T14:41:00+08:00

**状态：未测。现有阿里云密钥调用返回 HTTP 403 `AccessDenied.Unpurchased`，尚未取得任何模型答案。** 接口拒绝不计作模型答错。

原测试题目已准备好，共 3,000 条，未重新抽样。语义与逗号条件逐题配对；随机条件沿用原独立抽样数据。

| 语言 | 条件 | 完成 | 正确 b | 错答 a | 错答 c | 其他 |
|---|---|---:|---:|---:|---:|---:|
| 中文 | 语义连写 | 0/500 | 未测 | 未测 | 未测 | 未测 |
| 中文 | 独立随机 | 0/500 | 未测 | 未测 | 未测 | 未测 |
| 中文 | 语义加逗号 | 0/500 | 未测 | 未测 | 未测 | 未测 |
| 英文 | 强语义空格分隔 | 0/500 | 未测 | 未测 | 未测 | 未测 |
| 英文 | 独立随机 | 0/500 | 未测 | 未测 | 未测 | 未测 |
| 英文 | 强语义加逗号 | 0/500 | 未测 | 未测 | 未测 | 未测 |

计划沿用首次实际答案评分，忽略引导语，后续更正不覆盖先前答案。单轮 user 消息，无系统提示、历史消息或工具；关闭思考、temperature=0、max_tokens=32。

## 接入核对

- 官方模型 ID：`qwen4-max-preview`。预览版仅 Token Plan 提供，普通按量 API 不提供该模型。
- 已配置的普通阿里云 API 实测：403，无模型输出；模型目录也未列出 Qwen4。
- 已配置的 RightAPI Claude 通道：目录请求成功，13 个模型中没有 Qwen 模型。
- 本机已检查的现有配置未发现 Token Plan 专用凭据。
- 腾讯私有目录未查询：从旧会话提取凭据的操作被自动审批拒绝；未执行该命令。
- 关闭思考在官方文档中受支持，但尚未通过成功调用核实。
- 精确 tokenizer 未核实。旧分词接口返回 404，不能据此断言这些序列在 Qwen4 下全部恰好为三个 token；没有用其他 Qwen 分词器替代。

题目文件：`items.jsonl`。SHA-256：`b059a390f208f6e58913af4020730f9e69b5d77fedf5b46087316d591b5789df`。

来源：[Qwen4 模型说明](https://help.aliyun.com/en/model-studio/qwen4-max-preview)、[思考模式](https://help.aliyun.com/en/model-studio/deep-thinking)、[Token Plan 接入](https://help.aliyun.com/en/model-studio/token-plan-personal-quick-start)。
