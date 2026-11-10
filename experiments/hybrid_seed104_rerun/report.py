from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).parent
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':11,'pdf.fonttype':42})

def read(p):
 try:return json.loads(p.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return {}

def render():
 d=ROOT/'runs/hybrid_seed104';rows=[];p=d/'history.jsonl'
 if p.exists():
  for line in p.read_text().splitlines():
   try:row=json.loads(line)
   except json.JSONDecodeError:continue
   rows.append(row)
 rows=sorted({row['epoch']:row for row in rows}.values(),key=lambda r:r['epoch'])
 ev=[row for row in rows if 'evaluation' in row]
 status=read(d/'status.json');hb=read(d/'heartbeat.json');supervisor=read(ROOT/'supervisor_status.json')
 state='完成' if (d/'result.json').exists() else supervisor.get('state',status.get('state','准备中'))
 lines=['# Hybrid seed104 新复验：完整训练曲线','',
 '更新时间：'+datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(timespec='seconds'),'',
 '**这是 2026-11-07 重新训练的曲线，不是旧实验日志。**','',
 f'状态：{state}；训练进度：第 {hb.get("epoch",status.get("epoch",0))}/1200 轮；已完成评估 {len(ev)} 次。','',
 '使用原来的 seed104 数据和 hybrid 模型，每轮评估，完整保留 spike；固定报告最终第 1200 轮。','',
 '[完整准确率曲线](accuracy_curve.png) · [PDF](accuracy_curve.pdf) · [逐轮数据](accuracy_curve.csv)','']
 if ev:
  m=ev[-1]['evaluation'];lines += [f'## 最新完整评估：第 {ev[-1]["epoch"]} 轮','',
   '| 指标 | 正确率 | 首成分错误率 |','|---|---:|---:|']
  for key,label in [('arithmetic','真实算术'),('standard','普通检索'),('semantic','语义组合平均'),('semantic_0','01 检索'),('semantic_1','02 检索'),('semantic_2','21 检索'),('control_a1b','x1y 对照'),('control_a2b','x2y 对照')]:
   mm=m[key];f=f'{100*mm["rates"]["first"]:.2f}%' if key.startswith('semantic') else '—'
   lines.append(f'| {label} | {100*mm["accuracy"]:.2f}% | {f} |')
  keys=['arithmetic','standard','semantic','semantic_0','semantic_1','semantic_2','control_a1b','control_a2b']
  with (ROOT/'accuracy_curve.csv').open('w') as f:
   writer=csv.writer(f);writer.writerow(['epoch','loss']+[k+'_accuracy_percent' for k in keys]+['semantic_first_error_percent'])
   for row in ev:
    mm=row['evaluation'];writer.writerow([row['epoch'],row['loss']]+[100*mm[k]['accuracy'] for k in keys]+[100*mm['semantic']['rates']['first']])
  fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True)
  colors=['#087e8b','#7b8493','#bd5c3b']
  x=[row['epoch'] for row in ev]
  for key,label,c in zip(['arithmetic','standard','semantic'],['算术正确率','普通检索正确率','语义检索平均正确率'],colors):
   axes[0].plot(x,[100*row['evaluation'][key]['accuracy'] for row in ev],color=c,label=label,lw=1.7)
  for key,label,c in zip(['semantic_0','semantic_1','semantic_2'],['01 检索','02 检索','21 检索'],['#2468a0','#ca7d21','#7552a0']):
   axes[1].plot(x,[100*row['evaluation'][key]['accuracy'] for row in ev],color=c,label=label,lw=1.7)
  axes[2].plot([row['epoch'] for row in rows],[row['loss'] for row in rows],color='#4b586c',label='逐轮训练 loss',lw=1.5)
  for ax in axes[:2]:ax.set_ylim(-2,102);ax.set_ylabel('准确率（%）')
  axes[2].set_ylabel('训练 loss');axes[2].set_xlabel('训练轮数（epoch）')
  for ax in axes:
   ax.legend(loc='upper left',ncol=3,frameon=False);ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
  axes[-1].set_xlim(0,max(10,x[-1]))
  fig.suptitle(f'Hybrid seed104 新复验：第 1–{x[-1]} 轮实测曲线',fontsize=17,y=.98)
  fig.text(.09,.025,'每轮评估；当前图只包含已经完成的轮次。新训练不能视为恢复历史 seed104 的轨迹。',fontsize=10,color='#4a5568')
  fig.subplots_adjust(top=.91,bottom=.09,left=.09,right=.97,hspace=.17)
  for ext in ['png','pdf']:
   temp=ROOT/f'accuracy_curve.tmp.{ext}';fig.savefig(temp,dpi=160,facecolor='white');temp.replace(ROOT/f'accuracy_curve.{ext}')
  plt.close(fig)
 lines+=['','检查点：每 25 轮保留一次模型权重，第 1175–1200 轮每轮保留；latest.pt 含恢复训练所需优化器状态。','']
 tmp=ROOT/'results.md.tmp';tmp.write_text('\n'.join(lines));tmp.replace(ROOT/'results.md')

if __name__=='__main__':render()
