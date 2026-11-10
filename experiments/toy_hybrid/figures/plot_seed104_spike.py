from pathlib import Path
import csv,json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).parent
SOURCE=ROOT.parent/'results.md'
text=SOURCE.read_text()
for needed in ['20.72%','99.93%','3.60%','1175','1150','1200','2.55','2.46']:
    assert needed in text
# Only explicitly retained summary measurements: no synthesized epochs.
points=[(1150,20.72,'spike 后的评估'),(1175,99.93,'汇总采用的评估轮次'),(1200,3.60,'最后一轮：末尾 spike')]
with (ROOT/'hybrid_seed104_retained_accuracy_points.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['epoch','arithmetic_accuracy_percent','note'])
    w.writerows(points)
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':12,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(11,5),gridspec_kw={'width_ratios':[1.45,1]})
ax=axes[0]
ax.scatter([p[0] for p in points],[p[1] for p in points],s=75,color=['#c65447','#087e8b','#c65447'],zorder=3)
for epoch,acc,note in points:
    offset=(0,12) if acc<30 else (0,-22)
    ax.annotate(f'{acc:.2f}%',(epoch,acc),xytext=offset,textcoords='offset points',ha='center',fontweight='bold',color='#087e8b' if epoch==1175 else '#c65447')
ax.set_xlim(1140,1210);ax.set_ylim(-3,106);ax.set_xticks([1150,1175,1200]);ax.set_xlabel('评估轮数（epoch）');ax.set_ylabel('算术测试正确率（%）')
ax.set_title('留存的后期准确率评估点',pad=14)
ax.axvline(1175,color='#087e8b',alpha=.25,ls='--')
ax=axes[1]
# Summary retains the final three rounded epoch losses. No other losses are invented.
loss_epochs=[1198,1199,1200];losses=[.24,2.55,2.46]
ax.scatter(loss_epochs,losses,s=65,color=['#7c8796','#c65447','#c65447'],zorder=3)
for ep,loss in zip(loss_epochs,losses):
    ax.annotate(f'{loss:.2f}',(ep,loss),xytext=(0,10),textcoords='offset points',ha='center',fontweight='bold')
ax.set_xlim(1197.5,1200.5);ax.set_ylim(0,3);ax.set_xticks(loss_epochs);ax.set_xlabel('训练轮数（epoch）');ax.set_ylabel('训练损失（loss）');ax.set_title('最后三轮：loss 突增',pad=14)
for ax in axes:
    ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Hybrid seed104：末尾 spike 与汇总轮次调整',fontsize=17,y=.98)
fig.text(.08,.055,'仅绘制 results.md 中明确留存的数值；完整逐轮日志未找到，因此不连接或插值。\n1175 轮仅留存评估记录，未保存该轮权重；1200 轮是最终模型状态。',fontsize=10,color='#4a5568')
fig.subplots_adjust(top=.80,bottom=.25,left=.08,right=.97,wspace=.35)
for ext in ['png','pdf']:fig.savefig(ROOT/f'hybrid_seed104_retained_spike_points.{ext}',dpi=180,facecolor='white')
(ROOT/'hybrid_seed104_plot_provenance.json').write_text(json.dumps({'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'raw_training_log_found':False,'arithmetic_points':points,'final_loss_points':list(zip(loss_epochs,losses)),'interpolation':False,'epoch1175_weights_saved':False},ensure_ascii=False,indent=2)+'\n')
print(ROOT/'hybrid_seed104_retained_spike_points.png')
