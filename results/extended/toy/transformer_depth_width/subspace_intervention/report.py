from pathlib import Path
import json,csv
root=Path(__file__).resolve().parent
rows=[]
for folder in sorted(root.glob('w[0-9]*')):
 if not folder.is_dir() or not (folder/'baseline.json').exists():continue
 base=json.loads((folder/'baseline.json').read_text())
 for path in sorted(folder.glob('layer*.json')):
  d=json.loads(path.read_text())
  if not d.get('complete') or not d.get('double_orthogonal'):continue
  for cond,m in d['conditions'].items():
   r={'width':d['width'],'epoch':d['epoch'],'layer':d['layer'],'condition':cond}
   for k in ['arithmetic','standard','semantic']:
    r[k]=100*m[k]['correct']/m[k]['n']
    r[k+'_drop']=100*base['metrics'][k]['correct']/base['metrics'][k]['n']-r[k]
   rows.append(r)
with (root/'metrics.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['width'],lineterminator='\n')
 writer.writeheader();writer.writerows(rows)
lines=['# 成分读取和算术方向的逐层干预','','## 宽度假说的比较标准','','宽度假说预期窄模型中成分读取与组合运算的功能重叠较多，宽模型中两者较容易分别访问。第一项检验是移除读取方向后，窄模型算术明显下降，而宽模型算术保持。第二项检验是移除宽模型的算术方向后，算术确实受到破坏，同时检索保持。应对照同层、随机干预和探针读出变化。','','两种宽度的八层均已完成，共 90 个定向或随机干预条件及两份基线。现有数据在第 2 层符合第一项预测的方向，尚未完成两项预测的共同验证。','','第 2 层移除读取方向，256 维算术从 99.97% 降至 87.68%，下降 12.29 个百分点；两组匹配随机干预下降 4.92 和 5.23 个百分点。1024 维算术维持 99.99%。第 3 层分别下降 5.18 和 0.00 个百分点，但窄模型的随机对照也下降 7.30 和 3.08 个百分点。第一层方向相反，窄模型下降 3.25 个百分点，宽模型下降 20.12 个百分点。层间效果尚不一致。','','第二项检验尚未获得支持。1024 维第一层移除算术敏感方向，算术从 99.99% 降到 46.48%，检索也从 55.74% 降到 24.56%。其他层的这项干预几乎没有降低算术正确率，因此检索保持也不能满足该检验的前提。','','原先突出窄模型的算术受损而检索保持，偏离了宽度假说所要求的宽模型判据。窄模型该方向的结果仍保留于下表。另需注意，本轮算术方向在构造时被强制投影到成分方向的正交补空间，其正交性来自构造。','','每层移除读取方向后的算术准确率下降如下，单位为百分点。','','| 层 | 256 维 | 1024 维 |','|---:|---:|---:|','| 1 | 3.25 | 20.12 |','| 2 | 12.29 | 0.00 |','| 3 | 5.18 | 0.00 |','| 4 | 0.15 | 0.00 |','| 5 | 0.07 | 0.00 |','| 6 | 0.20 | 0.01 |','| 7 | 0.02 | 0.00 |','| 8 | 0.00 | 0.00 |','','## 实验设置','','冻结 seed 106 的八层模型。256 维取第 1,200 轮，1024 维取第 250 轮。每层在 b 位置读取注意力输出加正常 MLP 输出，方向确定后只干预该层，随后继续正常前向计算。','','普通组合的 Softmax 探针使用 21,184 条训练输入、2,648 条验证输入，拟合协议沿用逐层实验。把标准化坐标下的探针权重转换为原始坐标，并减去类别均值，取前 16 个右奇异向量构成成分读取方向。探针目标为所有 331 个数值身份。','','算术敏感方向使用训练集的 768 道算术题，每个运算 256 题，计算正确答案 logit 相对全部数值 logit 均值的梯度，取 b 位置在该层输出处的梯度。先投影掉与成分读取方向的重叠，再取前 16 个右奇异向量。两组方向的正交由构造保证。算术梯度方向反映这批训练例上的输出敏感性。','','对于方向基 Q，从本层更新量 u 中移除 QQᵀu，并从完整状态 h 中减去同一项。每项干预都有两组固定随机正交方向对照。每个样本的随机扰动被缩放到与对应定向扰动相同的欧氏长度。所有方向和层均在看测试结果前确定；全部八层逐一报告。第八层 b 位置无法再向后续位置传递信息，算术梯度为零，保留其成分方向干预作为结构性对照。','','## 结果','','以下均为测试集正确率（%）。算术 12,288 题，普通检索 2,000 题，运算组合检索 7,680 题。b 为成分方向，arithmetic 为算术敏感方向；random0/1 为同秩、逐题同扰动长度的随机对照。','','| 宽度 | 层 | 干预 | 算术 | 普通检索 | 运算组合检索 |','|---:|---:|---|---:|---:|---:|']
for w in (256,1024):
 p=root/f'w{w}/baseline.json'
 if p.exists():
  d=json.loads(p.read_text())['metrics']
  lines.append('| '+str(w)+' | — | baseline | '+' | '.join(f"{100*d[k]['correct']/d[k]['n']:.2f}" for k in ['arithmetic','standard','semantic'])+' |')
for r in rows:lines.append('| '+' | '.join(map(str,[r['width'],r['layer'],r['condition'],*[f"{r[k]:.2f}" for k in ['arithmetic','standard','semantic']]]))+' |')
probe_lines=['','## 原探针对成分投影的响应','','只在送入原探针的向量中移除相同方向，复用原权重与标准化。以下测量针对探针的分类输出。','','| 宽度 | 层 | 普通组合原始 | 普通组合投影后 | 运算组合原始 | 运算组合投影后 |','|---:|---:|---:|---:|---:|---:|']
for w in (256,1024):
    p=root/f'w{w}/probe_projection_effect.json'
    if not p.exists():continue
    for layer,d in json.loads(p.read_text()).items():
        vals=[]
        for group in ['ordinary','semantic']:
            for mode in ['baseline','removed_b']:
                selected=[v[mode] for k,v in d.items() if k.startswith(group)]
                vals.append(100*sum(v['correct'] for v in selected)/sum(v['n'] for v in selected))
        probe_lines.append('| '+' | '.join(map(str,[w,layer,*[f'{v:.2f}' for v in vals]]))+' |')
lines+=probe_lines
lines+=['','256 维和 1024 维的轮数不同，方向维数都为 16，占各自宽度的比例不同。本实验首先解释每份冻结权重内部的干预效应。算术任务同时要求保存操作数和执行运算，因此算术受损本身无法把原因唯一归于组合语义。所有生成答案的测试百分数由保存的逐题预测复核。256 维基线运算检索为 897/7,680，与原训练日志 896/7,680 相差一题，各干预均与本次相同前向计算的基线比较。','',
'[各任务的下降幅度](metrics.csv) · [方向正交和逐题计数复核](audit.json)']
(root/'README.md').write_text('\n'.join(lines)+'\n')
for w in (256,1024):
 print(w)
 for cond in ['b','arithmetic']:
  chosen=[r for r in rows if r['width']==w and r['condition']==cond]
  for r in chosen: print(r['layer'],cond,*[round(r[k+'_drop'],2) for k in ['arithmetic','standard','semantic']])
