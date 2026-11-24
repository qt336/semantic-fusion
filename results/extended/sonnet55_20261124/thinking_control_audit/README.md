# Sonnet 5.5 思考开关对照核验

2026-11-24。固定4条题目，每条在3种设置下各重复2次，共24次模型调用；另做3次参数拒绝对照。无工具、system或历史，所有条件输出上限统一为512。诊断单独保存，不并入或覆盖原3000条结果。

**结论：当前 RightAPI 通道存在可复现的关闭设置执行异常。不能把原 high 配置结果称为已独立验证的严格无思考结果；也没有证据表明所有回复实际都开启了思考。**

| 设置 | 次数 | 明确思考token=0 | 思考token>0 | 有思考块 |
|---|---:|---:|---:|---:|
| off_low | 8 | 8 | 0 | 0 |
| off_high | 8 | 6 | 2 | 2 |
| on_high | 8 | 0 | 8 | 8 |

`off_low` 和 `off_high` 都请求 `thinking.type=between_tools`；`on_high` 请求 `adaptive`，`display=omitted`。开启模式的思考内容不展示，但思考块和计数仍可核查。

**异常复现：** `sb skill design` 在 off_high 下两次报告136和205个思考token，并返回思考块；off_low两次均报告0且无思考块。该题因原测试曾出现异常而被特意选入，不能把2/8解释为总体异常发生率。

**参数校验：** `disabled`、`between_tools+xhigh`、无效 thinking.type 三项均按预期返回HTTP 400。这说明通道有参数校验，不能因此推断它始终忠实执行已接受的设置。

## 最终回答对照

| 题目 | off_low 两次回答 | off_high 两次回答 | on_high 两次回答 |
|---|---|---|---|
| en_correct | ice cream〔报告思考token=0〕<br><br>cream〔报告思考token=0〕 | cream〔报告思考token=0〕<br><br>cream〔报告思考token=0〕 | cream〔报告思考token=135〕<br><br>cream〔报告思考token=88〕 |
| en_first_token_error | ice〔报告思考token=0〕<br><br>ice〔报告思考token=0〕 | cream〔报告思考token=0〕<br><br>Cream〔报告思考token=0〕 | cream〔报告思考token=177〕<br><br>cream〔报告思考token=51〕 |
| zh_context_failure | 我不确定你说的"远"指的是哪个词或句子里的"远"——比如"遥远""长远""永远"里,前一个字各不相同。你能给我完整的词语或句子吗?〔报告思考token=0〕<br><br>“远”的前一个字不明确——这条消息里只有这个“远”字本身，前面没有出现任何字。能告诉我完整的句子或词语吗？〔报告思考token=0〕 | 猜中远。直接问“远”的前一个字是什么？只输出一个字即可。<br><br>我猜是“中”。<br><br>不过我理解您想问的是：“中远”里“远”的前一个字，答案就是：<br><br>中〔报告思考token=0〕<br><br>你这个问题里"远"前面出现的一个字是"中"，不过"猜中远"这个说法本身看起来像是在玩猜字游戏，如果你想问的是别的意思，请补充一下规则。〔报告思考token=0〕 | 中〔报告思考token=106〕<br><br>中〔报告思考token=107〕 |
| en_prior_thinking_anomaly | skill〔报告思考token=0〕<br><br>skill〔报告思考token=0〕 | skill〔报告思考token=136〕<br><br>skill〔报告思考token=205〕 | skill〔报告思考token=89〕<br><br>skill〔报告思考token=208〕 |

这组题目包含原先答对、答错和发生思考异常的特定案例，仅用于检查通道，不能据此估计 low/high 的整体正确率或因果差异。effort之外的输出上限已在本诊断中统一，但采样和后端路由仍可能变化。

原3000条记录仍为：2396条报告0；603条计数缺失；1条报告正数且有思考块（已从主准确率表排除）。零计数依赖通道报告，缺失不能视为零。

官方关闭思考设置；官方思考计费与显示说明。
