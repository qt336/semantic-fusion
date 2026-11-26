"""One paired layout for every seed; unavailable checkpoints stay explicit."""
import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text()) if p.exists() else {}
def value(m):return 100*m['correct']/m['n']
def rate(m):return f'{value(m):.2f}%'
lines=['# 所有 seed：逐层宽窄探针对比','','统一比较同 seed、同层、同为 8 层 Transformer 的 256 维与 1024 维。读取 b 位置的注意力输出 + 正常 MLP 输出，去掉本层输入的直连残差，MLP 输入保持正常。',
'', '探针只用普通组合训练：21,184 训练 / 2,648 验证 / 2,648 测试，331 类；零初始化，AdamW lr=0.001、wd=0.01、batch=256，100 轮，按普通验证准确率选模（平手选交叉熵）。运算组合测试共 7,680 题。没有按运算组合准确率挑选模型或读取器。不同宽度检查点轮数不同。',
'', '**基础任务筛选：** 算术与普通检索均 ≥99% 为达标。108 的窄模型没有符合条件的可用权重，完整展示配对数值，但标记未达标，不混入达标模型汇总。109、110 尚无对应的 8 层、256 维训练模型，不能拿 4 层权重替代。',
'', '106 复用先前完成的同口径探针；107、108 为本次追加实验。',
'', '## 各 seed 的逐层运算组合读出准确率','','每格依次为 **256维 / 1024维**，均为探针预测 b 的准确率，而非模型生成答案的准确率。',
'', '| 层 | Seed 106 | Seed 107 | Seed 108（窄模型未达标） | Seed 109 | Seed 110 |','|---:|---:|---:|---:|---|---|']
for layer in range(1,9):
 row=[str(layer)]
 for seed in range(106,111):
  if seed>=109:row.append('缺少8层窄模型');continue
  vals=[]
  for width in [256,1024]:
   r=read(ROOT/f'seed{seed}_w{width}/layer{layer:02}.json');vals.append(rate(r['semantic']) if r else '运行中')
  row.append(' / '.join(vals))
 lines.append('| '+' | '.join(row)+' |')
lines+=['','## 检查点与模型实际任务成绩','','| Seed | 宽度 | 模型轮数 | 算术 | 普通检索 | 运算组合检索 | 基础任务达标 | 探针进度 |','|---:|---:|---:|---:|---:|---:|---|---:|']
for seed in range(106,109):
 for width in [256,1024]:
  d=ROOT/f'seed{seed}_w{width}';meta=read(d/'metadata.json');base=read(d/'baseline.json')
  qualified=all(value(base[k])>=99 for k in ['arithmetic','standard']) if base else None
  lines.append('| '+' | '.join(map(str,[seed,width,meta.get('epoch','待启动'),*[rate(base[k]) if k in base else '—' for k in ['arithmetic','standard','semantic']],('是' if qualified else '否') if base else '待核验',f"{len(list(d.glob('layer??.json')))}/8"]))+' |')
for seed in range(106,109):
 lines+=['',f'## Seed {seed}：完整配对','','差值 = 1024维 − 256维，单位百分点。','','| 层 | 256维普通 | 1024维普通 | 256维运算 | 1024维运算 | 运算差值 |','|---:|---:|---:|---:|---:|---:|']
 for layer in range(1,9):
  pair={w:read(ROOT/f'seed{seed}_w{w}/layer{layer:02}.json') for w in [256,1024]}
  row=[str(layer)]+[rate(pair[w][key]) if pair[w] else '运行中' for key in ['test','semantic'] for w in [256,1024]]
  row.append(f"{value(pair[1024]['semantic'])-value(pair[256]['semantic']):+.2f}" if all(pair.values()) else '运行中')
  lines.append('| '+' | '.join(row)+' |')
tmp=ROOT/f'results.{os.getpid()}.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(ROOT/'results.md')
