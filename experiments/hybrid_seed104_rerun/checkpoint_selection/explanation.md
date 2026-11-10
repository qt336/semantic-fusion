# 为什么论文使用 Hybrid seed104 的第1175轮

核对对象：`/home/user/qintian/iclr2027_submission/paper.pdf`附录D、Table12及Table13；原汇总`/home/user/qintian/toy_hybrid_20260912/results.md`；本次重训完整的1200轮history.jsonl。

论文中的seed104是第1175轮评估；此前聊天中给出的算术3.60%是本次重训第1200轮，比较轮次不同。论文中的语义检索79.65%还是五个seed的均值，并非seed104单独的正确率。

## 核验结果

本次重训与原模型文件、数据生成文件、seed104数据文件逐字节一致；训练计算及优化参数保持。记录频率增加到每轮评估，并额外保留权重快照。本次1175轮的算术、普通检索、三种语义组合检索、首成分错误、两个控制组都与原汇总到报告精度一致。原汇总留存的1150轮算术20.72%、1146轮loss2.61、1199/1200轮loss2.55/2.46、最终算术3.60%也全部吻合。核验明细及原文件哈希见verification.json。

完整曲线来自这次重新训练，不能标成找回的历史逐轮日志；多个已保留数值的吻合支持本次复验重现了原记录所描述的现象。

## 更换评估轮次的依据

| 轮次 | 训练loss | 算术正确率 | 普通检索正确率 | 语义检索正确率 | 首成分错误率 |
|---|---:|---:|---:|---:|---:|
| 1175，论文采用 | 0.249 | 99.93% | 97.40% | 68.82% | 19.49% |
| 1198，末尾spike前 | 0.243 | 99.95% | 97.05% | 65.72% | 20.90% |
| 1199 | 2.550 | 0.79% | 95.65% | 49.95% | 35.91% |
| 1200 | 2.456 | 3.60% | 97.30% | 68.80% | 22.80% |

第1169–1198轮连续30轮中，算术正确率保持99.85%–99.95%，训练loss保持0.243–0.254。这表明模型在末尾异常之前已稳定掌握算术。第1199轮loss突然增加约10倍，算术正确率同步从99.95%跌至0.79%；第1200轮仍未恢复。因而，仅用最后一轮不能代表此前已经达到的稳定算术表现。

第1175轮位于该稳定区间，也是原来每25轮评估记录中最后一个未受末尾spike影响的轮次。它的语义检索正确率68.82%并非稳定区间最高值（最高70.05%），与第1200轮68.80%也几乎相同。因此，这个选择有避开训练不稳定状态的实证依据，而不是选取语义检索最高点。它仍是观察到末尾异常后的事后调整；报告时应保留轮次标记和最终轮次的对照，不能称为预先规定的选择规则。当前论文附录D已明确说明这一点。

原训练没有保留1175轮权重，历史结果来自当时评估记录；本次重训已实际保留1175、1198、1199和1200轮的权重，可供后续复核。本次权重不能称为找回的原训练权重。

[末尾曲线](checkpoint_selection.png) · [PDF](checkpoint_selection.pdf) · [末尾逐轮数据](tail_metrics.csv) · [核验明细](verification.json)

## 可用于论文的说明草稿

Seed 104 exhibits a terminal optimization spike. Over epochs 1,169–1,198, arithmetic accuracy stays between 99.85% and 99.95%, with training loss between 0.243 and 0.254. At epoch 1,199, the loss rises to 2.55 and arithmetic accuracy drops to 0.79%; the final epoch remains affected, with loss 2.46 and arithmetic accuracy 3.60%. We therefore report the earlier epoch-1,175 evaluation, the last scheduled evaluation before the terminal spike. Its semantic retrieval accuracy is 68.82%, compared with 68.80% at epoch 1,200, and is not the highest within the stable interval. This is a post-hoc adjustment for training instability; the reporting epoch and the final-epoch results are disclosed explicitly. A fresh rerun matches the retained original metrics at their reported precision and provides the complete per-epoch trace shown here.
