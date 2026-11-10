from pathlib import Path
from statistics import mean
import json
from summarize import main as summarize,score,OPS,MATCH,OLD
R=Path(__file__).parent
D=[json.loads((R/f'results_seed{s}.json').read_text()) for s in [101,102,103]]
F=[json.loads((R/f'factorial_results_seed{s}.json').read_text()) for s in [101,102,103]]
summary={'families':{},'factorial':{},'optimizer_convergence':[],'negative_control':[],'softmax_config_range':[]}
for family in ['adamw','adamw_large','ridge','lbfgs']:
 selected=[next(r for r in d['readers'] if r['name']==d['selected'][family]) for d in D]
 summary['families'][family]={key:mean(vals) for key,vals in {
  'ordinary_fresh':[score(r,['ordinary_fresh_test']) for r in selected],
  'operator':[score(r,OPS) for r in selected],
  'operator_reads_A':[score(r,OPS,field='reads_A') for r in selected],
  'operator_masked':[score(r,OPS,'masked_layer0') for r in selected],
  'B_matched_controls':[score(r,MATCH) for r in selected],
  'ordinary_heldout':[score(r,OLD) for r in selected]}.items()}
 fr=[r for d in F for r in d['rows'] if r['family']==family and r['condition']=='baseline']
 summary['factorial'][family]={k:mean(r[k] for r in fr)*(1 if k=='interaction_pp' else 100) for k in ['semantic','replace_A','replace_B','replace_both','interaction_pp']}
for d in D:
 for row in d['readers']:
  if row['family']=='lbfgs':summary['optimizer_convergence'].append({'seed':d['seed'],'lambda':row['lambda'],'converged':row['converged'],'gradient_inf':row['trace'][-1]['gradient_inf']})
  if row['name']=='shuffled':summary['negative_control'].append({'seed':d['seed'],'ordinary_fresh':score(row,['ordinary_fresh_test'])})
 configrows=[r for r in d['readers'] if r['family']=='adamw' and r['budget']==2000]
 vv=[score(r,OPS) for r in configrows]
 summary['softmax_config_range'].append({'seed':d['seed'],'min_operator':min(vv),'max_operator':max(vv)})
(R/'robustness_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
summarize()
lines=['','## 四格匹配对照（补充分析）','',
'每个原上下文联合抽取八对 A′、B′。保持位置及其他上下文不变，三个普通反事实组合都不出现在任何探针训练集或验证集中。此分析是在发现 A 身份影响后补做，所有三个模型均报告。','',
'| 解码器 | 原运算 AB | 只换 A：A′B | 只换 B：AB′ | 两者都换：A′B′ | 四格交互差（百分点） |','|---|---:|---:|---:|---:|---:|']
for f,vals in summary['factorial'].items():
 lines.append('| '+f+' | '+' | '.join(f'{vals[k]:.2f}' for k in ['semantic','replace_A','replace_B','replace_both','interaction_pp'])+' |')
lines+=['','交互差定义为 acc(A′B)+acc(AB′)−acc(A′B′)−acc(AB)，是描述性的概率尺度差分，不是正确率，因此可能超过 100 个百分点。正值表示四格结果存在额外组合交互；不把这一尺度上的交互唯一归因于语义。','',
'| 模型 seed | 解码器 | 原 AB | A′B | AB′ | A′B′ | 交互差 |','|---:|---|---:|---:|---:|---:|---:|']
for seed,fd in zip([101,102,103],F):
 for fam in ['adamw','adamw_large','ridge','lbfgs']:
  rows=[r for r in fd['rows'] if r['family']==fam and r['condition']=='baseline']
  vals=[mean(r[k] for r in rows)*(1 if k=='interaction_pp' else 100) for k in ['semantic','replace_A','replace_B','replace_both','interaction_pp']]
  lines.append(f'| {seed} | {fam} | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
lines+=['','保持 A 的初次补充对照仍保留在 afixed_results_seed*.json；四格对照使用新抽取且四格一致的 A′、B′，不将两个对照批次的数值混为一组。','']
with (R/'results.md').open('a') as f:f.write('\n'.join(lines))
intro='''## 完成情况与主要结论

本轮全部完成：三个冻结模型，每个模型九组 Softmax 探针配置训练至 2000 轮预算；另做四倍训练数据、Ridge、独立 L-BFGS 优化、标签打乱及四格匹配对照。所有配置和轮数仅按普通验证集选择。L-BFGS 的九个拟合均达到梯度无穷范数不超过 1e-6 的停止标准。训练数据、验证数据、测试数据无输入重叠；主实验 4134 行指标及补充对照 144 行指标均从保存的预测独立复核通过。

**较扎实的结论是：在这个人为去掉 embedding 残差的表示上，普通组合上学到的 Softmax 线性读取规则很难迁移到运算组合；这不只是原先训练 100 轮不充分造成的。** 四倍数据版本普通新测试平均 77.44%，运算组合平均 5.68%；保持 B 而替换 A 的匹配对照为 99.93%。固定探针后屏蔽第一层 B→A 注意力，运算组合升至 70.82%，三个 seed 均提高（0.00→55.34、0.00→57.11、17.03→100.00）。

**仍不能把这个结果等同于“B 信息消失”或“已经证明首 token 崩塌”。** Ridge 在运算组合上平均为 27.73%；探针的错误也多数不是 A。保留 A、只换 B 的四格对照同样较难（四倍数据版本 27.15%），说明 A 的身份也有影响。原模型检索训练中未出现的普通组合 9,10 / 15,20 平均只有 46.18%，所以“仅语义组合会失败”并不成立。四格对照显示组合相关的额外差异，但不能唯一归因于语义子空间机制。

这里的 Softmax 指**线性分类探针的训练方式**；主模型始终是普通 Softmax-attention Transformer。所有表格是**探针解码 B 的准确率，不是主模型生成答案的准确率**。adamw 是原数据 Softmax 探针，adamw_large 是四倍数据版本，lbfgs 是使用另一优化器拟合的线性多类逻辑回归。屏蔽只删除第一层全部注意力头中 B 位置对 A 位置的注意力连接，其余连接重新归一化；不删除整层，且屏蔽后不重新拟合探针。

此处 a_B 仍由原输入正常计算；去掉的是块内相加的 e_B，以及 MLP 输入中的 e_B。因此不能把该实验描述成“注意力完全没有接触过 B”。本轮使用重建模型的三个 seed，不是论文原始五个 checkpoint。原始短程结果保留，不用它代替本轮稳健性结果。

![拟合曲线、配置敏感性与屏蔽结果](robustness.png)

'''
report=R/'results.md'
body=report.read_text()
body=body.replace('## 验证集选出的解码器',intro+'## 验证集选出的解码器',1)
report.write_text(body)
print(json.dumps(summary,ensure_ascii=False,indent=2))
