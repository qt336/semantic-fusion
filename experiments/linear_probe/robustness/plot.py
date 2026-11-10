from pathlib import Path
import json
from statistics import mean
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
R=Path(__file__).parent
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'pdf.fonttype':42,'font.size':10})
colors={101:'#087e8b',102:'#b65b37',103:'#8059a3'}
fig,axes=plt.subplots(1,3,figsize=(15,4.6))
for seed in [101,102,103]:
 for tag,style in [('grid','-'),('large','--')]:
  p=R/f'{tag}_seed{seed}_history.json'
  if not p.exists():continue
  h=json.loads(p.read_text());y=[100*max(r['best_val_acc']) for r in h]
  axes[0].plot([r['epoch'] for r in h],y,style,color=colors[seed],label=f'{seed} '+('原数据' if tag=='grid' else '4倍数据'))
axes[0].set_title('普通验证集：最佳已观察正确率');axes[0].set_xlabel('探针训练 epoch');axes[0].set_ylabel('B 解码正确率（%）');axes[0].legend(fontsize=8,ncol=2,loc='lower right')
ops=['operator_0_1','operator_0_2','operator_2_1']
def score(row,group,condition='baseline'):
 return 100*mean(row['results'][condition+'/'+g]['accuracy'] for g in group)
rows=[]
for seed in [101,102,103]:
 p=R/f'results_seed{seed}.json'
 if not p.exists():continue
 d=json.loads(p.read_text());by={r['name']:r for r in d['readers']}
 grid=[r for r in d['readers'] if r['family']=='adamw' and r['budget']==2000]
 axes[1].scatter([score(r,['ordinary_fresh_test']) for r in grid],[score(r,ops) for r in grid],color=colors[seed],s=30,alpha=.7,label=str(seed))
 for family,marker in [('adamw','o'),('adamw_large','s'),('ridge','^'),('lbfgs','D')]:
  r=by[d['selected'][family]]
  rows.append((seed,family,score(r,ops),score(r,ops,'masked_layer0')))
axes[1].plot([0,100],[0,100],':',color='#999999');axes[1].set_xlabel('新普通测试 B 正确率（%）');axes[1].set_ylabel('运算组合 B 正确率（%）');axes[1].set_title('9组配置 × 3个模型：2000轮预算');axes[1].legend(title='模型 seed')
family_names=['adamw','adamw_large','ridge','lbfgs'];xx=np.arange(4);width=.34
for offset,ind,color,label in [(-width/2,2,'#b65b37','未屏蔽'),(width/2,3,'#087e8b','屏蔽 B→A')]:
 vals=[mean(r[ind] for r in rows if r[1]==f) if any(r[1]==f for r in rows) else np.nan for f in family_names]
 axes[2].bar(xx+offset,vals,width,color=color,label=label)
 for x,y in zip(xx+offset,vals):
  if np.isfinite(y):axes[2].text(x,y+1,f'{y:.1f}',ha='center',fontsize=8)
axes[2].set_xticks(xx,['Softmax','4倍数据','Ridge','L-BFGS']);axes[2].set_ylabel('运算组合 B 正确率（%）');axes[2].set_title('普通验证集选出的解码器：3模型均值');axes[2].legend(fontsize=8)
for ax in axes:
 ax.set_ylim(0,105);ax.grid(axis='y',alpha=.17);ax.spines[['right','top']].set_visible(False)
axes[1].set_xlim(0,105)
fig.suptitle('去掉两处 embedding、保留 attention＋MLP：线性探针稳健性',fontsize=15)
fig.text(.5,.015,'所有配置和轮数仅按普通验证集选择。原数据 21,184 条，4倍数据 84,736 条；探针目标均为 B。',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.05,1,.92])
for ext in ['png','pdf']:fig.savefig(R/f'robustness.{ext}',dpi=180,facecolor='white')
