"""Verify retained historical numbers and show the rerun's terminal spike."""
from pathlib import Path
import csv
import hashlib
import json
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).parent.resolve()
RUN = ROOT.parent
ORIGINAL = RUN.parent / 'qintian/toy_hybrid_20260912'


def main():
    history = RUN / 'runs/hybrid_seed104/history.jsonl'
    rows = [json.loads(line) for line in history.read_text().splitlines()]
    assert [row['epoch'] for row in rows] == list(range(1, 1201))
    assert all('evaluation' in row for row in rows)
    by = {row['epoch']: row for row in rows}
    # Published/retained values have limited precision; compare at that precision.
    expected = {'arithmetic': (99.93, 2), 'standard': (97.40, 2),
                'semantic_0': (50.0, 1), 'semantic_1': (79.6, 1),
                'semantic_2': (76.9, 1), 'control_a1b': (99.8, 1),
                'control_a2b': (97.7, 1)}
    checks = []
    for key, (value, precision) in expected.items():
        actual = 100 * by[1175]['evaluation'][key]['accuracy']
        assert round(actual, precision) == value
        checks.append({'epoch': 1175, 'metric': key, 'retained_value_percent': value,
                       'rerun_value_percent': actual, 'reported_decimal_places': precision})
    actual_first = 100 * by[1175]['evaluation']['semantic']['rates']['first']
    assert round(actual_first, 2) == 19.49
    checks.append({'epoch': 1175, 'metric': 'semantic_first',
                   'retained_value_percent': 19.49, 'rerun_value_percent': actual_first})
    for epoch, value in [(1150, 20.72), (1200, 3.60)]:
        actual = 100 * by[epoch]['evaluation']['arithmetic']['accuracy']
        assert round(actual, 2) == value
        checks.append({'epoch': epoch, 'metric': 'arithmetic',
                       'retained_value_percent': value, 'rerun_value_percent': actual})
    for epoch, value in [(1146, 2.61), (1199, 2.55), (1200, 2.46)]:
        actual = by[epoch]['loss']
        assert round(actual, 2) == value
        checks.append({'epoch': epoch, 'metric': 'loss', 'retained_value': value, 'rerun_value': actual})
    hashes = {}
    for name in ['data.py', 'model.py', 'datasets/seed_104.npz']:
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest(RUN / name) == digest(ORIGINAL / name)
        hashes[name] = digest(RUN / name)
    window = [r for r in rows if 1169 <= r['epoch'] <= 1198]
    ranges = {}
    for key in ['arithmetic', 'standard', 'semantic']:
        values = [100 * r['evaluation'][key]['accuracy'] for r in window]
        ranges[key] = {'min_percent': min(values), 'max_percent': max(values),
                       'mean_percent': statistics.mean(values)}
    ranges['loss'] = {'min': min(r['loss'] for r in window), 'max': max(r['loss'] for r in window)}
    manifest = {'passed': True, 'source': str(history), 'history_sha256': hashlib.sha256(history.read_bytes()).hexdigest(),
                'historical_reference': str(ORIGINAL / 'results.md'),
                'historical_reference_sha256': hashlib.sha256((ORIGINAL / 'results.md').read_bytes()).hexdigest(),
                'historical_anchors_match_at_retained_precision': checks,
                'identical_source_and_dataset_sha256': hashes, 'stable_window_1169_1198': ranges,
                'curve_provenance': 'Fresh seed104 rerun; not recovered historical per-epoch logs.',
                'selection_status': 'Historical post-hoc replacement of epoch1200 evaluation by epoch1175; not a preregistered rule.'}
    (ROOT / 'verification.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    tail = [r for r in rows if r['epoch'] >= 1140]
    with (ROOT / 'tail_metrics.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'training_loss', 'arithmetic_percent', 'standard_percent',
                         'semantic_correct_percent', 'semantic_first_error_percent'])
        for r in tail:
            e = r['evaluation']
            writer.writerow([r['epoch'], r['loss'], *[100 * e[k]['accuracy'] for k in ['arithmetic', 'standard', 'semantic']],
                             100 * e['semantic']['rates']['first']])
    font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family': 'Noto Sans CJK JP', 'axes.unicode_minus': False,
                         'pdf.fonttype': 42, 'font.size': 10})
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9), sharex=True)
    x = [r['epoch'] for r in tail]
    green, orange, blue, grey = '#087e8b', '#c76a31', '#366ab1', '#738092'
    axes[0].plot(x, [100 * r['evaluation']['arithmetic']['accuracy'] for r in tail], color=green, lw=1.8, label='算术正确率')
    axes[0].plot(x, [100 * r['evaluation']['standard']['accuracy'] for r in tail], color=grey, lw=1.2, label='普通检索正确率')
    axes[1].plot(x, [100 * r['evaluation']['semantic']['accuracy'] for r in tail], color=orange, lw=1.8, label='语义检索正确率')
    axes[1].plot(x, [100 * r['evaluation']['semantic']['rates']['first'] for r in tail], color=blue, lw=1.5, label='首成分错误率')
    axes[2].plot(x, [r['loss'] for r in tail], color='#414c5c', lw=1.8, label='逐轮训练 loss')
    for ax in axes:
        ax.axvspan(1169, 1198, color=green, alpha=.055)
        ax.axvspan(1198.5, 1200.5, color='#c84646', alpha=.12)
        ax.axvline(1175, color=green, ls='--', lw=1.2)
        ax.grid(alpha=.16)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(loc='lower left' if ax is axes[0] else 'upper left', frameon=False, ncol=2)
    for ax in axes[:2]:
        ax.set_ylim(-4, 108)
        ax.set_ylabel('比例（%）')
    axes[2].set_ylabel('训练 loss')
    axes[2].set_ylim(0, 3.4)
    axes[2].set_xlim(1139, 1202)
    axes[2].set_xlabel('训练轮数（epoch）')
    axes[2].set_xticks([1140, 1150, 1160, 1170, 1175, 1180, 1190, 1198, 1200])
    axes[2].tick_params(axis='x', rotation=35)
    for epoch in [1175, 1198, 1199, 1200]:
        r = by[epoch]
        axes[0].scatter(epoch, 100 * r['evaluation']['arithmetic']['accuracy'], color=green, s=30, zorder=4)
    axes[0].annotate('论文采用第1175轮：99.93%', (1175, 99.9349), (1163, 67),
                     arrowprops={'arrowstyle': '->', 'color': green}, color=green)
    axes[0].annotate('1198轮：99.95%', (1198, 99.9512), (1181, 82),
                     arrowprops={'arrowstyle': '->', 'color': green}, color=green)
    axes[0].annotate('1199轮：0.79%\n1200轮：3.60%', (1200, 3.597), (1180, 22),
                     arrowprops={'arrowstyle': '->', 'color': '#ac3737'}, color='#ac3737')
    for epoch, offset in [(1175, (1159, 45)), (1200, (1181, 91))]:
        acc = 100 * by[epoch]['evaluation']['semantic']['accuracy']
        axes[1].scatter(epoch, acc, color=orange, zorder=4, s=30)
        axes[1].annotate(f'{epoch}轮：{acc:.2f}%', (epoch, acc), offset,
                         arrowprops={'arrowstyle': '->', 'color': orange}, color=orange)
    axes[2].scatter([1175, 1198, 1199, 1200], [by[e]['loss'] for e in [1175, 1198, 1199, 1200]], color='#414c5c', s=25, zorder=4)
    axes[2].annotate('稳定区间1169–1198：loss约0.24–0.25', (1187, by[1187]['loss']), (1154, .85),
                     arrowprops={'arrowstyle': '->', 'color': green}, color=green)
    axes[2].annotate('末尾 spike\n1199轮：2.55；1200轮：2.46', (1199, by[1199]['loss']), (1178, 2.85),
                     arrowprops={'arrowstyle': '->', 'color': '#ac3737'}, color='#ac3737')
    fig.suptitle('Hybrid seed104：稳定评估轮次与末尾 loss spike', fontsize=16, y=.98)
    fig.text(.09, .025, '曲线来自重新训练的逐轮日志；已核对1175轮各指标及末尾spike与原记录到报告精度一致。\n虚线：论文采用的1175轮；浅绿色：稳定区间；浅红色：最后两轮。原始选择为事后调整。', fontsize=9, color='#495566')
    fig.subplots_adjust(top=.93, bottom=.13, left=.09, right=.98, hspace=.18)
    for ext in ['png', 'pdf']:
        fig.savefig(ROOT / f'checkpoint_selection.{ext}', dpi=180, facecolor='white')
    plt.close(fig)
    print('Verified historical anchors and wrote checkpoint-selection plot.')


if __name__ == '__main__':
    main()
