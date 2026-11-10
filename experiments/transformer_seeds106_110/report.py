"""Live Markdown from saved metrics; no model calls."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import statistics

ROOT = Path(__file__).parent.resolve()
SEEDS = list(range(106, 111))


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def render():
    schedule = read(ROOT / 'scheduler_status.json')
    active = {v['seed']: v['physical_gpu'] for v in schedule.get('active', [])}
    entries = []
    for seed in SEEDS:
        directory = ROOT / 'runs' / f'transformer_seed{seed}'
        status, final = read(directory / 'status.json'), read(directory / 'result.json')
        evaluation, evaluation_epoch = {}, None
        # Last line for an epoch wins if a worker resumed from an older checkpoint.
        if (directory / 'history.jsonl').exists():
            for line in (directory / 'history.jsonl').read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if 'evaluation' in row:
                    evaluation, evaluation_epoch = row['evaluation'], row['epoch']
        metrics = final.get('baseline', evaluation)
        state = '已完成' if final else '训练中' if seed in active else '失败' if seed in schedule.get('failed', []) else '排队中'
        epoch = final.get('completed_epochs', status.get('epoch', 0))
        entries.append((seed, epoch, evaluation_epoch, state, metrics, final))
    pct = lambda val: f'{100 * val:.2f}'
    lines = ['# 原始算术＋检索任务：Transformer seed106–110', '',
             '更新时间：' + datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'), '',
             '普通4层Transformer；每轮270000条算术、135000条检索，训练1200轮，batch8190。检索训练仅排除01、02、21；没有额外随机输出或共现任务。',
             '沿用2026-09-12按论文重建的实现；固定第1200轮统计，不挑选中间高点。', '',
             '| Seed | GPU | 已完成轮数 | 最近评估轮数 | 状态 |',
             '|---|---:|---:|---:|---|']
    for seed, epoch, ev, state, _, _ in entries:
        lines.append(f'| {seed} | {active.get(seed, "—")} | {epoch}/1200 | {ev or "—"} | {state} |')
    lines += ['', '## 算术与检索正确率（%）', '',
              '| Seed | 算术测试 | 普通检索 | 01检索 | 02检索 | 21检索 | 语义检索均值 | x1y | x2y |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    names = ['arithmetic', 'standard', 'semantic_0', 'semantic_1', 'semantic_2', 'semantic', 'control_a1b', 'control_a2b']
    for seed, _, _, _, metrics, _ in entries:
        values = [pct(metrics[k]['accuracy']) if k in metrics else '—' for k in names]
        lines.append('| ' + ' | '.join([str(seed)] + values) + ' |')
    complete = all(v[5].get('completed_epochs') == 1200 for v in entries)
    values = [pct(statistics.mean(v[5]['baseline'][k]['accuracy'] for v in entries)) if complete else '待完成' for k in names]
    lines.append('| ' + ' | '.join(['五seed最终均值'] + values) + ' |')
    lines += ['', '## 语义检索输出分布（%）', '',
              '| Seed | 组合 | 正确第二成分 | 错答第一成分 | EOS | 其他 |',
              '|---|---|---:|---:|---:|---:|']
    for seed, _, _, _, metrics, _ in entries:
        for name, label in [('semantic_0', '01'), ('semantic_1', '02'), ('semantic_2', '21'), ('semantic', '合计')]:
            rates = metrics.get(name, {}).get('rates', {})
            values = [pct(rates[k]) if k in rates else '—' for k in ['correct', 'first', 'eos', 'other']]
            lines.append('| ' + ' | '.join([str(seed), label] + values) + ' |')
    lines += ['', '每种运算组合2560条测试，记忆长度3、4、5、6各640条；普通检索2000条；算术每种运算4096条留出测试。四类互斥，分母为全部对应测试样本。',
              '训练中显示最近一次评估；每25轮评估一次。完整逐轮loss、评估和检查点位于runs/transformer_seed<seed>/。',
              '任务在screen qt1中运行。GPU4、5按原队列接续；GPU1、6分别等随机输出对照104、105完成并释放后，可接续尚未启动的108、110。同一seed只运行一次，每张卡同时一个本组worker。',
              '数据与源文件核验见protocol_manifest.json；完成后自动根据逐样本预测复核所有结果。', '']
    tmp = ROOT / 'results.md.tmp'
    tmp.write_text('\n'.join(lines))
    tmp.replace(ROOT / 'results.md')


if __name__ == '__main__':
    render()
