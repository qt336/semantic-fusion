from pathlib import Path
import json,csv
import numpy as np
from statistics import mean

root=Path(__file__).resolve().parent
with (root/'paired_epochs.csv').open() as f:
    rows=list(csv.DictReader(f))
groups=[r for r in rows if min(float(r[f'w{w}_{k}']) for k in ['standard','arithmetic'] for w in (256,1024))>=99]
lines=['# 加宽后为什么更容易检索运算组合中的第二个符号？','','## 同 seed、同训练轮数的比较','','两种模型均为八层、八头。使用相同 seed 的固定数据和评估题。快照涵盖 seed 106 至第 250 轮，107、108 至第 350 轮，共 41 对检查点，其中 31 对由宽模型取得较高的运算组合准确率。各轮重复测同一批题，不能视为独立模型重复。','','在这 41 对中，要求两模型的算术和普通检索均达到 99%，符合条件的只有 seed 107 的第 325、350 轮。以下各数均为全部运算组合题的百分比。','','| 轮数 | 宽度 | 答对 b | 误答 a | 输出结束符 | 其他 |','|---:|---:|---:|---:|---:|---:|']
for r in groups:
    for w in (256,1024):
        vals=[float(r[f'w{w}_{k}']) for k in ('correct','first','eos','other')]
        lines.append(f"| {r['epoch']} | {w} | "+' | '.join(f'{v:.2f}' for v in vals)+' |')
lines+=['','两轮平均，宽模型正确率高 69.79 个百分点，结束符错误少 47.01 个百分点，误答首项少 7.04 个百分点，其他错误少 15.74 个百分点。第 350 轮三个运算对均有改善。较宽模型在运算输入上更常给出有效答案。', '', '## 冻结模型上的输入替换与注意力干预','','seed 106，256 维用第 1,200 轮权重，1024 维用第 250 轮权重。同一批 7,680 道题中，显示的三个符号为 a、b、q，末尾再出现一次 q 要求模型输出 b。固定每题上下文，随机抽取序列外的普通符号只替换 a；答案 b 和查询 q 保持相同。测试所有三种运算组合，每种 2,560 题。比较时保持模型参数冻结。两种宽度使用不同训练轮数，这部分只用于解释每个模型内部的变化。','','表中数字为答对 b 的百分比。注意力遮挡在所有头的 softmax 前进行，其余连接重新归一化。除明确标为第一层的条件外，在全部八层操作。','','| 干预 | 256 维运算组合 | 256 维替换 a 后 | 1024 维运算组合 | 1024 维替换 a 后 |','|---|---:|---:|---:|---:|']
edge=json.loads((root/'edge_results.json').read_text())
labels={'baseline':'正常运行','block_b_from_a_first':'第一层，b 不能看 a','block_displayed_q_from_b':'显示序列中的 q 不能看 b','block_displayed_q_from_a':'显示序列中的 q 不能看 a','block_final_query_from_b':'末尾查询 q 不能看 b','block_final_query_from_a':'末尾查询 q 不能看 a'}
with np.load(root/'predictions.npz') as z, (root/'paired_changes.csv').open('w',newline='') as changes:
    writer=csv.writer(changes,lineterminator='\n');writer.writerow(['width','condition','both_correct','rescued','lost','both_wrong'])
    for mode,label in labels.items():
        vals=[]
        for w in ('256','1024'):
            for ordinary in (False,True):
                key=('ordinary_a' if mode=='baseline' else 'ordinary_a+'+mode) if ordinary else mode
                m=edge[w]['conditions'][key];p=z['w'+w+'_'+key];ok=p==z['gold']
                base=z['w'+w+('_ordinary_a' if ordinary else '_baseline')]==z['gold']
                assert m['n']==len(p)==7680 and ok.sum()==m['correct']
                assert m['first']==sum(p==z['ordinary_a' if ordinary else 'first'])
                writer.writerow([w,key,*[int(x.sum()) for x in (base&ok,~base&ok,base&~ok,~base&~ok)]])
                vals.append(100*ok.mean())
        lines.append('| '+label+' | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
lines+=['','只替换 a，两份模型都接近满分，说明查询和答案的身份不足以解释原题困难。屏蔽末尾查询对 b 的访问使两份模型在运算组合上几乎完全失去正确答案；屏蔽末尾查询对 a 的访问则提高了正确率。遮挡会同时改变剩余位置的注意力权重。这个对照支持查询时从不同位置取信息会影响答案。它没有直接测量子空间，也不能区分加宽残差流、注意力头或 MLP 各自的作用。','','## 数据与复核','','[同轮检查点与各运算对](paired_epochs.csv) · [干预计数](edge_results.json) · [逐题输出](predictions.npz) · [配对正确与错误转换](paired_changes.csv)。[输入替换与遮挡代码](test_edges.py) 调用本实验使用的 model.py。256 维基线各运算对计数与训练记录完全一致。1024 维本次基线为 4,280/7,680，训练日志为 4,281/7,680，差一题，两者在 (0,1) 上的正确数分别为 2,035 和 2,036。各项干预均与本次重新计算的基线比较。']
(root/'README.md').write_text('\n'.join(lines)+'\n')
