# GPT-5.6 Sol 在其他模型错答首成分的题目上的表现

[下载逐题结果（JSONL.gz）](answers.jsonl.gz)

原始无逗号语义题中排除三种 GPT 小测已用的三元组和位置对照中已用的序列。要求 Opus 5 与 DeepSeek V4.1 Flash 在关闭思考的历史请求中均直接回答首成分（忽略英文大小写），且 Opus 明确报告 0 个思考 token。符合条件的中文题 29 道，英文题 38 道。按固定的题号哈希顺序从每种语言各取 12 道，各语义组合只取一次。选择只使用其他模型的既有答案，发送给 GPT 的内容仅为原始提示词。

沿用 RightAPI 与原 GPT 协议，各题独立请求，reasoning.effort=none。题单 SHA-256 为 `bd758c263678b988448a714ee321af3df2056b0907e2a1724fecd5e02d502928`。

| 语言 | 正确 | 原样逐字答对 | 错答首成分 | 其他错误 |
|---|---:|---:|---:|---:|
| zh | 11/12 | 11/12 | 1 | 0 |
| en | 12/12 | 12/12 | 0 | 0 |

返回 none 且报告零推理 token 24/24。输入计数为 3763–3769 token。这是基于其他模型错误挑选的样本，只用于比较这些具体题目。通道附加上下文和上游模型身份没有独立核实。

| 序列 | 正确答案 | GPT-5.6 Sol | Opus 5 | DeepSeek V4.1 Flash |
|---|---|---|---|---|
| high school frog | school | school | high | high |
| hard drive tail | drive | drive | hard | hard |
| 打烂清 | 烂 | 烂 | 打 | 打 |
| stock market bee | market | market | stock | stock |
| 分完手 | 完 | 分 | 分 | 分 |
| ice cream kite | cream | cream | ice | ice |
| ice hockey queen | hockey | hockey | ice | ice |
| 缝补灯 | 补 | 补 | 缝 | 缝 |
| common sense bread | sense | sense | Common | common |
| 贴好瓦 | 好 | 好 | 贴 | 贴 |
| real estate bread | estate | estate | real | real |
| 跑完暖 | 完 | 完 | 跑 | 跑 |
| fast food train | food | food | Fast | fast |
| 冻住梅 | 住 | 住 | 冻 | 冻 |
| science fiction fish | fiction | fiction | science | science |
| natural gas face | gas | gas | Natural | natural |
| 拉倒哭 | 倒 | 倒 | 拉 | 拉 |
| dead end train | end | end | dead | dead |
| middle class brush | class | class | Middle | middle |
| 算错心 | 错 | 错 | 算 | 算 |
| 锯开耳 | 开 | 开 | 锯 | 锯 |
| 打翻灯 | 翻 | 翻 | 打 | 打 |
| 移开远 | 开 | 开 | 移 | 移 |
| 拧开夏 | 开 | 开 | 拧 | 拧 |
