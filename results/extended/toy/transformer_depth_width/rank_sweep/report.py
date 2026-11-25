from pathlib import Path
import csv,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
root=Path(__file__).resolve().parent
rows=[];baselines={}
for w in (256,1024):
 p=root/f'w{w}/baseline.json'
 if not p.exists():continue
 baselines[w]=json.loads(p.read_text())['metrics']
 for path in sorted((root/f'w{w}').glob('layer*.json')):
  d=json.loads(path.read_text())
  r={k:d[k] for k in ('width','epoch','layer','kind','rank','variant')}
  for k in ('arithmetic','standard','semantic'):
   m=d['metrics'][k];r[k]=100*m['counts']['correct']/m['n']
   r[k+'_first']=100*m['counts']['first']/m['n']
  rows.append(r)
if not rows:raise SystemExit()
with (root/'metrics.csv').open('w') as f:
 wr=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');wr.writeheader();wr.writerows(rows)
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['axes.unicode_minus']=False
for kind,title in [('b','移除成分读取方向'),('arithmetic','移除独立估计的算术方向')]:
 fig,axes=plt.subplots(4,2,figsize=(12,14),sharex=True,sharey=True)
 for layer,ax in enumerate(axes.flat,1):
  for w,c in [(256,'#b65333'),(1024,'#167d9a')]:
   for task,style,label in [('arithmetic','-','算术'),('semantic','--','运算组合检索')]:
    sub=sorted([r for r in rows if r['width']==w and r['layer']==layer and r['kind']==kind and r['variant']=='target'],key=lambda r:r['rank'])
    if not sub:continue
    m=baselines[w][task];x=[0]+[r['rank'] for r in sub];y=[100*m['counts']['correct']/m['n']]+[r[task] for r in sub]
    ax.plot(x,y,style,color=c,marker='o',markersize=3,label=f'{w} 维 · {label}')
  ax.set_xscale('symlog',linthresh=1,base=2);ax.set_xticks([0,1,4,16,64,128,256]);ax.set_xticklabels(['0','1','4','16','64','128','256'],fontsize=8)
  ax.set_ylim(-3,103);ax.grid(alpha=.15);ax.set_title(f'第 {layer} 层',fontproperties=font)
  if layer in (7,8):ax.set_xlabel('移除方向数',fontproperties=font)
  if layer%2:ax.set_ylabel('正确率（%）',fontproperties=font)
  if kind=='arithmetic' and layer==8:ax.text(.5,.5,'此位置的算术梯度为零',transform=ax.transAxes,ha='center',fontproperties=font)
 handles,labels=axes[0,0].get_legend_handles_labels()
 fig.legend(handles,labels,loc='lower center',ncol=2,prop=font)
 fig.suptitle(title+' · seed 106',fontproperties=font,fontsize=16)
 fig.tight_layout(rect=[0,.055,1,.96])
 fig.savefig(root/f'{kind}_curves.png',dpi=145)
 plt.close(fig)
lines=['# 移除方向数量与算术、检索准确率','','## 观察结果','','第二层移除读取方向呈现宽度差异。256 维模型移除 16、64、128、256 个方向后，算术准确率依次为 87.68%、53.95%、35.74%、22.08%；1024 维在这些条件下均为 99.99%。匹配宽度比例时，窄模型 16/256 对应宽模型 64/1024，窄模型 64/256 对应宽模型 256/1024，差异仍然存在。','','第二层窄模型的方向具有一定针对性。移除 64 个读取方向后算术为 53.95%，两组同秩、逐题同扰动长度的随机对照为 65.97% 和 63.07%。宽模型的这些随机对照也保持 99.99%。','','宽模型第一层在四维尺度上出现功能区分。原算术为 99.99%，运算组合检索为 55.74%。移除四个读取方向后分别为 99.99% 和 42.11%；移除四个独立估计的算术方向后分别为 45.66% 和 59.79%。后者的两组匹配随机对照算术均为 99.99%。继续扩大移除维数，两项任务可同时受损。','','层间效果不同。第一层移除 16 个读取方向时，窄模型算术为 96.72%，宽模型为 79.88%，与第二层的相对敏感性不同。当前结果支持部分层、部分干预规模下功能可以分别访问。宽度效应的普遍性和子空间几何解释仍需更多匹配检查点和随机种子验证。','','本轮依据全扫描后报告这些有代表性的点，完整表格保留全部方向数、层和随机对照。算术方向的定义也更新为独立估计，因此与前轮固定 16 维且强制正交的算术方向干预存在方法差异。','','## 实验设置','','seed 106，八层模型，256 维固定第 1,200 轮，1024 维固定第 250 轮。与前轮相同的冻结权重、输入和测试题。当前完成 '+str(len(rows))+'/540 个干预条件。','','读取方向来自普通组合探针，恢复到原始坐标后按奇异值排序。算术方向来自算术训练样本上的输出梯度，独立进行 SVD，本轮不再施加与读取方向正交的约束。每组取前 1、4、16、64、128、256 个方向，因此同一类型的子空间随方向数递增而嵌套。','','每次移除 b 位置本层“注意力＋正常 MLP”更新量在这些方向上的投影，随后继续计算。每个目标条件附两组随机子空间对照，逐题匹配移除向量的欧氏长度。保留外侧残差。全部八层都测试；最后一层 b 位置无法再影响后面的答案位置，算术梯度为零。','','方向数和宽度比例均需对照。例如 16/256 对应 64/1024，64/256 对应 256/1024。训练轮数不同，当前结果描述两份固定权重的内部干预效应。','','## 曲线','','![移除读取方向](b_curves.png)','','![移除算术方向](arithmetic_curves.png)','','[全部条件与随机对照 CSV](metrics.csv) · [复核记录](audit.json)','','## 同方向数量的定向干预','','| 层 | 方向 | 移除数 | 256 算术 | 1024 算术 | 256 普通检索 | 1024 普通检索 | 256 运算组合检索 | 1024 运算组合检索 |','|---:|---|---:|---:|---:|---:|---:|---:|---:|']
for layer in range(1,9):
 for kind in ['b','arithmetic']:
  for rank in [1,4,16,64,128,256]:
   pair={r['width']:r for r in rows if r['layer']==layer and r['kind']==kind and r['rank']==rank and r['variant']=='target'}
   if not pair:continue
   cells=[str(layer),'读取' if kind=='b' else '算术',str(rank)]
   for task in ['arithmetic','standard','semantic']:
    cells += [f"{pair[w][task]:.2f}" if w in pair else '—' for w in [256,1024]]
   lines.append('| '+' | '.join(cells)+' |')
lines+=['','## 同宽度比例的比较','','| 层 | 方向 | 移除比例 | 窄模型方向数 | 宽模型方向数 | 窄模型算术 | 宽模型算术 | 窄模型运算检索 | 宽模型运算检索 |','|---:|---|---:|---:|---:|---:|---:|---:|---:|']
for layer in range(1,9):
 for kind in ['b','arithmetic']:
  for k in [1,4,16,64]:
   pair={r['width']:r for r in rows if r['layer']==layer and r['kind']==kind and r['variant']=='target' and ((r['width']==256 and r['rank']==k) or (r['width']==1024 and r['rank']==4*k))}
   if len(pair)!=2:continue
   cells=[str(layer),'读取' if kind=='b' else '算术',f'{100*k/256:.2f}%',str(k),str(4*k)]
   cells += [f"{pair[w][task]:.2f}" for task in ['arithmetic','semantic'] for w in [256,1024]]
   lines.append('| '+' | '.join(cells)+' |')

lines+=['','算术每条件 12,288 题，普通检索 2,000 题，运算组合检索 7,680 题。原始逐题输出及分类计数位于 w256、w1024 目录。']
(root/'README.md').write_text('\n'.join(lines)+'\n')
print('Rendered',len(rows),'conditions')
