from pathlib import Path
from statistics import mean
import json,csv
import numpy as np
R=Path(__file__).parent
OPS=['operator_0_1','operator_0_2','operator_2_1'];MATCH=['matched8_0_1','matched8_0_2','matched8_2_1'];OLD=['ordinary_9_10','ordinary_15_20']

def score(row,groups,cond='baseline',field='accuracy'):
 return 100*mean(row['results'][f'{cond}/{g}'][field] for g in groups)

def main():
 data=[json.loads(p.read_text()) for p in sorted(R.glob('results_seed*.json'))]
 lines=['# 第一层去 embedding 后的线性探针稳健性检查','',
'表示固定为 $z_B=a_B+\\mathrm{MLP}(\\mathrm{LN}_2(a_B))$，目标为从 331 个类别中解码 B。三个普通 Transformer 权重冻结，不进行模型训练。所有选择仅用普通验证集。','',
 f'完整模型 seed：{[d["seed"] for d in data]}；目标为 101、102、103。','',
'## 验证集选出的解码器','',
'每列均为正确率（%）。匹配普通对照保持 B、位置、上下文不变，每条运算题使用八个不同的普通 A，且这些 A,B 对在大小两种探针训练集和验证集中均未出现。','',
'| 模型 seed | 解码器 | 普通验证 | 原普通测试 | 新普通测试 | 运算组合 | 探针误读为 A | 屏蔽后运算组合 | 八替代 A 匹配普通对照 | 留出普通组合 9,10 / 15,20 |','|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 selected_rows=[];flat=[]
 for d in data:
  byname={q['name']:q for q in d['readers']}
  for fam,name in d['selected'].items():
   row=byname[name];selected_rows.append((d['seed'],fam,row))
   vals=[100*row['val_acc'],score(row,['ordinary_test']),score(row,['ordinary_fresh_test']),score(row,OPS),score(row,OPS,field='reads_A'),score(row,OPS,'masked_layer0'),score(row,MATCH),score(row,OLD)]
   lines.append(f'| {d["seed"]} | {fam} | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
  for row in d['readers']:
   for key,val in row['results'].items():
    cond,group=key.split('/');flat.append({'seed':d['seed'],'reader':row['name'],'family':row['family'],'selected':d['selected'].get(row['family'])==row['name'],'condition':cond,'group':group,'val_acc':row['val_acc'],**val})
 if len(data)==3:
  lines+=['','## 三模型均值','',
  '| 解码器 | 新普通测试 | 运算组合 | 探针误读为 A | 屏蔽后运算组合 | 八替代 A 匹配普通对照 | 留出普通组合 |','|---|---:|---:|---:|---:|---:|---:|']
  for fam in ['adamw','adamw_large','ridge','lbfgs']:
   rs=[q for _,f,q in selected_rows if f==fam]
   vals=[mean(score(q,['ordinary_fresh_test']) for q in rs),mean(score(q,OPS) for q in rs),mean(score(q,OPS,field='reads_A') for q in rs),mean(score(q,OPS,'masked_layer0') for q in rs),mean(score(q,MATCH) for q in rs),mean(score(q,OLD) for q in rs)]
   lines.append('| '+fam+' | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
 lines+=['','## 固定基准配置：延长训练','',
 '以下固定 lr=0.01、weight decay=0.01、零初始化，避免把延长训练和更换配置混为一谈。每个预算取其范围内普通验证集选出的轮数。','',
 '| 模型 seed | 训练预算 | 所选轮数 | 普通验证 | 新普通测试 | 运算组合 | 屏蔽后运算组合 |','|---:|---:|---:|---:|---:|---:|---:|']
 for d in data:
  for row in d['readers']:
   if not row['name'].startswith('adam_base_zero_budget'):continue
   vals=[100*row['val_acc'],score(row,['ordinary_fresh_test']),score(row,OPS),score(row,OPS,'masked_layer0')]
   lines.append(f'| {d["seed"]} | {row["budget"]} | {row["epoch"]} | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
 lines+=['','## 全部 AdamW 配置（2000 轮预算）','',
 '| 模型 seed | 配置 | 普通验证 | 新普通测试 | 运算组合 | 屏蔽后运算组合 | 匹配普通对照 |','|---:|---|---:|---:|---:|---:|---:|']
 for d in data:
  for row in d['readers']:
   if row['family']!='adamw' or row['budget']!=2000:continue
   vals=[100*row['val_acc'],score(row,['ordinary_fresh_test']),score(row,OPS),score(row,OPS,'masked_layer0'),score(row,MATCH)]
   lines.append(f'| {d["seed"]} | {row["config"]["name"]} | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
 lines+=['','## L-BFGS 收敛检查','',
 '目标为完整训练集上的多类交叉熵加显式 L2；使用 double precision，记录最终梯度无穷范数。阈值为 1e-6。','',
 '| 模型 seed | L2 | 迭代次数 | 最终梯度 | 达到阈值 | 普通验证 | 运算组合 |','|---:|---:|---:|---:|---|---:|---:|']
 for d in data:
  for row in d['readers']:
   if row['family']!='lbfgs':continue
   last=row['trace'][-1]
   lines.append(f'| {d["seed"]} | {row["lambda"]} | {last["iterations"]} | {last["gradient_inf"]:.3g} | {row["converged"]} | {100*row["val_acc"]:.2f} | {score(row,OPS):.2f} |')
 lines+=['','## 打乱标签的负对照','',
 '| 模型 seed | 原普通测试 | 新普通测试 |','|---:|---:|---:|']
 for d in data:
  row=next(q for q in d['readers'] if q['name']=='shuffled')
  lines.append(f'| {d["seed"]} | {score(row,["ordinary_test"]):.2f} | {score(row,["ordinary_fresh_test"]):.2f} |')
 lines+=['','机会水平为 1/331≈0.30%。负对照不参与解码器选择。','',
 '## 数据和解释范围','',
 '- 探针训练：21184 条；扩大后 84736 条。固定普通验证 2648 条，原普通测试 2648 条，新独立普通测试 10592 条；各类别平衡，新增训练不包含验证或测试输入。',
 '- 三个模型 seed、三类运算符是固定的研究对象；上下文重复测量不等于独立模型或独立运算符。',
 '- 匹配普通对照仅保证未进入探针训练，其组合可能进入过原 Transformer 的检索训练。留出的 9,10 / 15,20 对照始终同时报告。',
 '- AdamW 权重衰减与 L-BFGS 显式 L2 不是同一优化目标。比较用于检查解码器敏感性，不宣称它们必须产生相同权重。',
 '- 这个表示来自去掉 e_B 后重新计算 MLP 的人为干预，不是模型正常前向的隐藏状态。低分不自动证明信息擦除；解码器从普通组合迁移到运算组合失败也可能反映分布变化。',
 '- 尚未直接测量共享语义子空间，也没有据此修改论文结论。','',
 '完整配置见 PROTOCOL.md；全部指标见 all_metrics.csv；逐样本预测及拟合权重均保留。','']
 (R/'results.md').write_text('\n'.join(lines))
 if flat:
  with (R/'all_metrics.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 print('Reported seeds',[d['seed'] for d in data])

if __name__=='__main__':main()
