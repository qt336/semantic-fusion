"""Refresh a human-readable dashboard; incomplete runs never enter final means."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import statistics
from data import CONDITIONS

LABELS = {'heldout_pairs':'未见普通组合','random_operator':'随机运算符 9,10（2×）'}
PAIR_KEYS = [('operator_0_1','01'),('operator_0_2','02'),('operator_2_1','21'),
             ('ordinary_9_10','9,10'),('ordinary_15_20','15,20')]


def read_json(path):
    try: return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError): return {}


def pct(metric,key='correct'):
    return f'{100*metric["rates"][key]:.2f}' if metric else '—'


def render(root):
    root = Path(root)
    manifest = read_json(root/'run_manifest.json')
    seeds = manifest.get('seeds',[101,102,103])
    target = manifest.get('epochs',1200)
    scheduler = read_json(root/'scheduler_status.json')
    active = {(v['condition'],v['seed']):v['physical_gpu'] for v in scheduler.get('active',[])}
    lines = ['# 普通 Transformer：未见组合与随机运算符对照','',
       '更新时间：'+datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y-%m-%d %H:%M:%S %Z'),'',
       '**只有完整训练到目标轮数的结果才进入最终汇总；训练中的准确率不能作为结论。**','',
       '- 每组 3 个 seed（101、102、103）；仅使用清单中指定的 4 张 GPU，每卡一个独立运行。',
       '- 未见组合组：检索训练排除 01、02、21、9,10、15,20；个体 token 均在检索训练出现。',
       '- 随机运算符组：共享上述基础数据，另加 180,000 条/轮 `r 9 10 s → 随机值`；真实运算每种 90,000 条/轮。',
       '- 随机结果每次呈现独立重采样，均匀取 0–330；不构成可记忆的固定输入输出表。',
       '- 所有组合测试查询紧随组合的第三个 token，正确答案是组合的第二个 token。','',
       '## 运行进度','',
       '| 组别 | Seed | GPU | 状态 | 已完成 epoch | 最新评估 epoch |',
       '|---|---:|---:|---|---:|---:|']
    records = {}
    for condition in CONDITIONS:
        for seed in seeds:
            d = root/'runs'/f'{condition}_seed{seed}'
            status, result, latest = (read_json(d/n) for n in ['status.json','result.json','latest_evaluation.json'])
            failed = read_json(d/'failure.json')
            state = '完成' if result else '失败' if failed else {'preparing':'准备数据','training':'训练中'}.get(status.get('state'),'排队')
            epoch = result.get('completed_epochs',status.get('epoch',0))
            metrics = result.get('metrics',latest.get('metrics',{}))
            records[(condition,seed)] = {'result':result,'metrics':metrics,'epoch':latest.get('epoch',0)}
            lines.append(f'| {LABELS[condition]} | {seed} | {active.get((condition,seed),"—")} | {state} | {epoch}/{target} | {latest.get("epoch","—")} |')
    lines += ['', '## 最新评估：准确率（%）','',
       '| 组别 | Seed | 真实算术 | 普通检索 | 01 | 02 | 21 | 9,10 | 15,20 |',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for (condition,seed),r in records.items():
        m = r['metrics']
        lines.append('| '+ ' | '.join([LABELS[condition],str(seed),pct(m.get('arithmetic')),pct(m.get('standard'))]
                         + [pct(m.get(k)) for k,_ in PAIR_KEYS])+' |')
    lines += ['', '## 最新评估：错误回答成组合首 token 的比例（%）','',
       '分母为对应测试集的全部样本，不是仅在错误回答中计算。','',
       '| 组别 | Seed | 01 → 0 | 02 → 0 | 21 → 2 | 9,10 → 9 | 15,20 → 15 |',
       '|---|---:|---:|---:|---:|---:|---:|']
    for (condition,seed),r in records.items():
        lines.append('| '+' | '.join([LABELS[condition],str(seed)]+[pct(r['metrics'].get(k),'first') for k,_ in PAIR_KEYS])+' |')
    lines += ['', '## 最终三种子汇总','',
       '数值为均值 ± 样本标准差；未完成三次训练的组显示“待完成”。','',
       '| 组别 | 指标 | 真实算术 | 普通检索 | 01 | 02 | 21 | 9,10 | 15,20 |',
       '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        results = [records[(condition,s)]['result'] for s in seeds]
        ready = all(r and r['completed_epochs']==target and not r.get('small_smoke_test') for r in results)
        for rate, label in [('correct','正确率'),('first','首 token 错误率')]:
            values = []
            for key in ['arithmetic','standard']+[k for k,_ in PAIR_KEYS]:
                if rate=='first' and key in ['arithmetic','standard']: values.append('不适用')
                elif not ready: values.append('待完成')
                else:
                    vv = [100*r['metrics'][key]['rates'][rate] for r in results]
                    values.append(f'{statistics.mean(vv):.2f} ± {statistics.stdev(vv):.2f}')
            lines.append('| '+' | '.join([LABELS[condition],label]+values)+' |')
    lines += ['', '## 随机输出任务检查','',
       '该任务的目标不固定，随机猜测准确率约为 0.30%（1/331）。此项不计入“真实算术”准确率。','',
       '| 组别 | Seed | 随机输出任务准确率（%） |', '|---|---:|---:|']
    for (condition,seed),r in records.items():
        if condition=='random_operator': lines.append(f'| {LABELS[condition]} | {seed} | {pct(r["metrics"].get("noise_arithmetic_9_10"))} |')
    lines += ['', '## 如何解释','',
       '1. 第一组比较未见普通组合 9,10 / 15,20 与未见运算组合 01 / 02 / 21。普通组合若仍能检索，而运算组合出现首 token 错误，则“未在检索训练中出现”本身不足以解释差异。',
       '2. 第二组检查高频随机运算符 9,10 的检索是否也受损，并对照没有新增曝光的 15,20。不能预先假定任一方向。',
       '3. 同时检查真实算术与普通检索是否学好。若随机标签使整体训练失效，不能把组合检索下降单独归因于频率或语义。',
       '4. 两组共享基础样本，但第二组增加随机算术样本，因此每轮总样本数和优化器步数不同。这不是严格等训练步数的比较；本轮不会据此宣称单独证明某一内部机制。','',
       '协议与运行方式见 README.md；逐样本预测在各运行目录 predictions.npz；数据排除审计在 datasets/seed*.json。','']
    tmp = root/'results.md.tmp'
    tmp.write_text('\n'.join(lines))
    tmp.replace(root/'results.md')


if __name__ == '__main__': render(Path(__file__).parent)
