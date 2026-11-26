import json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
lines=['# 多 seed 逐层 Softmax 探针','','只选算术与普通检索均 ≥99% 的冻结模型。256 维 seed 108 因普通检索不足而排除；1024 维 108 单独报告，不作同 seed 配对。',
       '', '读取每层显示 b 位置的注意力输出 + 正常 MLP 输出，不包含本层输入直连项；模型冻结，MLP 输入保持正常。探针只用普通组合训练：21,184 训练 / 2,648 验证 / 2,648 测试，331 类；零初始化，AdamW lr=0.001、wd=0.01、batch=256，100 轮，按普通验证准确率选模（平手选交叉熵）。三种运算组合共 7,680 题；不按运算组合测试成绩选权重或探针。',
       '', '新实验共 24 个逐层读取器：107 的两个宽度及 108 的 1024 维。宽度间使用相同普通探针数据；同 seed 使用相同运算组合测试题。不同宽度检查点轮数不同。',
       '', '| Seed | 宽度 | 模型轮数 | 算术 | 普通检索 | 运算组合检索 | 探针进度 |','|---:|---:|---:|---:|---:|---:|---:|']
def rate(m):return f"{100*m['correct']/m['n']:.2f}%"
for seed,width in [(107,256),(107,1024),(108,1024)]:
    d=ROOT/f'seed{seed}_w{width}'
    meta=json.loads((d/'metadata.json').read_text()) if (d/'metadata.json').exists() else {}
    base=json.loads((d/'baseline.json').read_text()) if (d/'baseline.json').exists() else {}
    lines.append('| '+' | '.join(map(str,[seed,width,meta.get('epoch','待启动'),*[rate(base[k]) if k in base else '—' for k in ['arithmetic','standard','semantic']],f"{len(list(d.glob('layer??.json')))}/8"]))+' |')
lines+=['','## 逐层解码准确率','','百分比为探针读出 b 的准确率，不是模型生成回答的准确率。','','| 层 | 107·256 普通 | 107·256 运算 | 107·1024 普通 | 107·1024 运算 | 108·1024 普通 | 108·1024 运算 |','|---:|---:|---:|---:|---:|---:|---:|']
for layer in range(1,9):
    vals=[str(layer)]
    for seed,width in [(107,256),(107,1024),(108,1024)]:
        p=ROOT/f'seed{seed}_w{width}/layer{layer:02}.json'
        if p.exists():
            r=json.loads(p.read_text());vals += [rate(r['test']),rate(r['semantic'])]
        else:vals+=['待完成','待完成']
    lines.append('| '+' | '.join(vals)+' |')
tmp=ROOT/f'results.{os.getpid()}.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(ROOT/'results.md')
