"""Five-seed report with historical runs and the two explicitly added seeds."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import statistics
import tempfile

ROOT = Path(__file__).parent.resolve()
SOURCE = ROOT.parent


def read(path):
    try: return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError): return {}


def render():
    schedule = read(ROOT/'scheduler_status.json')
    active = {a['seed']: a['physical_gpu'] for a in schedule.get('active', [])}
    pc = lambda v: f'{100*v:.2f}'
    rows = []
    for seed in [101, 102, 103, 104, 105]:
        run = SOURCE/'runs'/f'random_operator_seed{seed}'
        status, final, latest = [read(run/n) for n in ['status.json', 'result.json', 'latest_evaluation.json']]
        metrics = final.get('metrics', latest.get('metrics', {}))
        rows.append((seed, status, final, latest, metrics))
    lines = ['# 随机输出9,10：五seed结果', '',
             '更新：'+datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'), '',
             '原seed101/102/103保留；只补跑104/105。所有seed使用完全相同的数据生成、普通Transformer和训练程序，固定第1200轮结果，不选择中间高点。',
             '每轮真实算术270000条、检索135000条、随机输出任务180000条；每种真实算符90000次。随机标签每轮重采样；检索训练中留出9,10。', '',
             '| Seed | GPU | 已完成轮数 | 最近评估轮数 | 状态 |', '|---|---|---:|---:|---|']
    for seed, status, final, latest, metrics in rows:
        state = '完成' if final else '训练中' if seed in active else status.get('state', '等待启动')
        lines.append(f'| {seed} | {active.get(seed,"—")} | {final.get("completed_epochs",status.get("epoch",0))}/1200 | {latest.get("epoch","—")} | {state} |')
    lines += ['', '## 9,10检索：原始输出（%）', '',
              '| Seed | 正确10 | 错答9 | 错答查询 | EOS | 其他 |', '|---|---:|---:|---:|---:|---:|']
    keys = ['correct', 'first', 'query', 'eos', 'other']
    for seed, _, _, _, metrics in rows:
        rates = metrics.get('ordinary_9_10', {}).get('rates', {})
        lines.append('| '+' | '.join([str(seed)]+[pc(rates[k]) if k in rates else '—' for k in keys])+' |')
    for selected, label in [([101,102,103], '原3seed最终均值'), ([101,102,103,104,105], '全部5seed最终均值')]:
        entries = [final for seed, _, final, _, _ in rows if seed in selected]
        ready = all(x.get('completed_epochs') == 1200 and x.get('small_smoke_test') is False for x in entries)
        values = [pc(statistics.mean(x['metrics']['ordinary_9_10']['rates'][k] for x in entries)) if ready else '待完成' for k in keys]
        lines.append('| '+' | '.join([label]+values)+' |')
    lines += ['', '每seed固定2560条检索测试，记忆长度3–6；五类互斥，分母是全部测试题。', '',
              '## 算术与检索对照（%）', '',
              '| Seed | 真实算术 | 普通检索 | 01检索 | 02检索 | 21检索 | 低频15,20 | 随机输出任务 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for seed, _, _, _, m in rows:
        names = ['arithmetic', 'standard', 'operator_0_1', 'operator_0_2', 'operator_2_1', 'ordinary_15_20', 'noise_arithmetic_9_10']
        vals = [pc(m[k]['accuracy']) if k in m else '—' for k in names]
        lines.append('| '+' | '.join([str(seed)]+vals)+' |')
    lines += ['', '随机输出任务的标签不可预测，机会水平约0.30%，不混入真实算术指标。不同seed按原协议同时改变模型初始化、数据采样和随机标签；并非只改变初始化。',
              '补充训练在screen qt1/randomop_104105，使用共享GPU 1、6；每个worker的日志、检查点和逐样本预测仍保存在原runs/random_operator_seed104或105。',
              '新训练的设置、数据验证、历史文件校验和见extension_104105/。原三seed完整对照表见results.md。', '']
    with tempfile.NamedTemporaryFile(mode='w', dir=SOURCE, prefix='.randomop5-', delete=False) as f:
        f.write('\n'.join(lines)); tmp = Path(f.name)
    tmp.replace(SOURCE/'random_operator_5seeds_results.md')


if __name__ == '__main__':
    render()
