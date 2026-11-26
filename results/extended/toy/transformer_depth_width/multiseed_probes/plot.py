import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False})
fig,axes=plt.subplots(1,3,figsize=(14,4.8),sharey=True,layout='constrained')
for seed,ax in zip([106,107,108],axes):
    for width,color in [(256,'#3985ba'),(1024,'#c45535')]:
        x=[];y=[]
        for layer in range(1,9):
            p=ROOT/f'seed{seed}_w{width}/layer{layer:02}.json'
            if p.exists():
                m=json.loads(p.read_text())['semantic'];x.append(layer);y.append(100*m['correct']/m['n'])
        ax.plot(x,y,'o-',color=color,label=f'{width} 维',lw=2,ms=5)
    ax.set_title(f'Seed {seed}'+ ('（窄模型基础任务未达标）' if seed==108 else ''))
    ax.set_xticks(range(1,9));ax.set_ylim(-3,105);ax.set_xlabel('读取层');ax.grid(alpha=.2);ax.legend()
axes[0].set_ylabel('运算组合中 b 的探针读出准确率（%）')
fig.suptitle('同 seed、同层宽窄对比：8 层 Transformer',fontsize=17)
fig.get_layout_engine().set(rect=(0,.06,1,.95))
fig.text(.5,.015,'检查点轮数不同；109、110 缺少 8 层窄模型，未画入配对图。',ha='center',fontsize=10,color='#555555')
for ext in ['png','pdf']:fig.savefig(ROOT/f'paired_all_seeds.{ext}',dpi=180)
