from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

root=Path(__file__).resolve().parents[1]
rows=[json.loads(l) for l in (root/'runs/random_operator_seed103/history.jsonl').read_text().splitlines() if l.strip()]
data=[]
for r in rows:
    if 'evaluation' not in r: continue
    m=r['evaluation']['ordinary_9_10']
    counts=m['counts']; n=m['n']
    assert sum(counts.values())==n
    correct=100*counts['correct']/n
    first=100*counts['first']/n
    other=100*sum(counts[k] for k in ['query','eos','other'])/n
    assert abs(correct+first+other-100)<1e-8
    data.append([r['epoch'],correct,100-correct,first,other])
with (root/'figures/seed103_pair9_10_curves.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['epoch','correct_percent','all_errors_percent','first_token_error_percent','remaining_errors_percent']);w.writerows(data)
a=np.array(data);x=a[:,0]
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':12,'axes.unicode_minus':False,'pdf.fonttype':42})
fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True,gridspec_kw={'height_ratios':[1,1]})
colors=['#167d8d','#c94343','#d77a16','#758296']
series=[[(1,'检索正确：回答 10',colors[0]),(2,'全部错误',colors[1])],[(3,'首 token 错误：回答 9',colors[2]),(4,'其余错误（查询 token / EOS / 其他）',colors[3])]]
for ax,ss in zip(axes,series):
    for col,label,color in ss:
        ax.plot(x,a[:,col],label=label,color=color,lw=2,marker='o',ms=3)
        ax.annotate(f'{a[-1,col]:.2f}%',xy=(x[-1],a[-1,col]),xytext=(9,0),textcoords='offset points',va='center',color=color,fontweight='bold',fontsize=11)
    ax.set_ylim(-2,104);ax.set_yticks(range(0,101,20));ax.set_xlim(0,1350)
    ax.set_ylabel('占全部测试样本的比例（%）')
    ax.grid(axis='both',alpha=.18);ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper left',bbox_to_anchor=(0,1.18),frameon=False,ncol=2,fontsize=11)
axes[-1].set_xticks(range(0,1201,200));axes[-1].set_xlabel('训练轮数（epoch）')
fig.suptitle('旧实验 seed103：随机运算符 9,10 的检索曲线',fontsize=18,y=.98)
fig.text(.5,.93,'随机输出样本频率 2×；9,10 从检索训练中留出；每次评估 2,560 条',ha='center',fontsize=11,color='#4a5568')
fig.text(.08,.025,'第 1 轮及之后每 25 轮评估，共 49 个实测点；折线仅连接采样点，未平滑。\n下图两类错误之和等于上图“全部错误”；均以全部样本为分母。',fontsize=10,color='#4a5568')
fig.subplots_adjust(top=.83,bottom=.14,left=.09,right=.94,hspace=.40)
for ext in ['png','pdf']:
    fig.savefig(root/f'figures/seed103_pair9_10_curves.{ext}',dpi=180,facecolor='white')
print('Final:',data[-1])
print('Saved:',root/'figures/seed103_pair9_10_curves.png')
