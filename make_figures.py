import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import fig_data as D

BLUE, GREEN, YELLOW, PURPLE, GREY = ('#DAE8FC', '#D5E8D4', '#FFF2CC',
                                     '#E1D5E7', '#EEEEEE')
PANEL_COLOR = {'direct': BLUE, 'ntk_x14': GREEN, 'yarn_8': PURPLE, 'pi_x8': YELLOW}
PANEL_TITLE = {
    'direct': 'Direct (no scaling)',
    'ntk_x14': r'NTK, $\lambda$=14',
    'yarn_8': 'YaRN, causal default',
    'pi_x8': 'Positional interpolation (all cells 0-3%)',
}
ERROR_COLORS = {'exact': GREEN, 'near': BLUE, 'wrong_number': YELLOW,
                'no_number': PURPLE, 'empty': GREY}
ERROR_ORDER = ['exact', 'near', 'wrong_number', 'no_number', 'empty']
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9.0,
    'axes.linewidth': 0.7, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def _ramp(hex_color):
    rgb = np.array(matplotlib.colors.to_rgb(hex_color))
    light = 1.0 - (1.0 - rgb) * 0.18
    full = rgb
    return LinearSegmentedColormap.from_list('ramp', [light, full])


BAR_8K = GREEN
BAR_24K = PURPLE


def _cell_text_color(value, vmax=100.0):
    return '#222222'


def load_matrix(results_dir):
    path = os.path.join(results_dir, 'expM_matrix.json') if results_dir else None
    if path and os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            rows = json.load(f)['rows']
        label = {'direct': 'direct', 'ntk': 'ntk_x14', 'pi': 'pi_x8',
                 'yarn': 'yarn_8'}
        out = {k: {} for k in label.values()}
        for r in rows:
            if r['factor'] == 14 and r['method'] == 'ntk':
                key = 'ntk_x14'
            elif r['method'] == 'pi':
                key = 'pi_x8'
            elif r['method'] == 'yarn':
                key = 'yarn_8'
            elif r['method'] == 'direct':
                key = 'direct'
            else:
                continue
            out[key].setdefault(r['context_tokens'], {}).setdefault(r['depth'], []).append(r['score'])
        lengths = sorted({r['context_tokens'] for r in rows})
        depths = sorted({r['depth'] for r in rows})
        m = {k: {L: [float(np.mean(v.get(d, [np.nan]))) for d in depths]
                 for L, v in out[k].items()} for k in out}
        print(f'[data] matrix from {path}')
        return m, lengths, depths
    print('[data] matrix from fig_data.py (transcribed fallback)')
    return D.MATRIX, D.LENGTHS, D.DEPTHS


def load_band(results_dir):
    path = os.path.join(results_dir, 'expB2_freqband.json') if results_dir else None
    if path and os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            rows = json.load(f)['rows']
        agg = {}
        for r in rows:
            key = r['variant']
            if key.startswith('band_'):
                key = key[len('band_'):]
            if key == 'yarn':
                s = r.get('spec') or {}
                try:
                    bf = float(s.get('beta_fast', 32))
                except (TypeError, ValueError):
                    bf = 32.0
                key = 'yarn_tuned' if bf < 32 else 'yarn_default'
            agg.setdefault(key, {}).setdefault(r['context_tokens'], []).append(
                r['score'])
        band = {k: {L: 100.0 * sum(1 for s in v if s >= 100.0) / len(v)
                    for L, v in d.items()} for k, d in agg.items()}
        print(f'[data] band from {path}')
        return band
    print('[data] band from fig_data.py (transcribed fallback)')
    return D.BAND


def load_errors(results_dir):
    path = os.path.join(results_dir, 'analysis_matrix.json') if results_dir else None
    if path and os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            rows = json.load(f)['rows']
        cats = ['exact', 'near', 'wrong_number', 'no_number', 'empty']
        modes = {}
        for r in rows:
            key = r['method']
            if key == 'dynamic_ntk':
                key = 'dynamic'
            counts = modes.setdefault(key, {c: 0 for c in cats})
            counts[r['category']] = counts.get(r['category'], 0) + 1
        print(f'[data] error modes from {path}')
        return {k: tuple(v[c] for c in cats) for k, v in modes.items()}
    print('[data] error modes from fig_data.py (transcribed fallback)')
    return D.ERROR_MODES


def load_order(results_dir, variants=('direct', 'ntk_x14', 'pi_x8')):
    if not results_dir:
        return None
    path = os.path.join(results_dir, 'expOrder_8k.json')
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as f:
        doc = json.load(f)
    ranks = {}
    for r in doc['rows']:
        if r['variant'] not in variants:
            continue
        key = (r['variant'], r['question'])
        ranks.setdefault(key, []).append(r['emitted_rank'])
    out = {}
    for key, vals in ranks.items():
        uniq = set(vals)
        if len(uniq) != 1:
            raise SystemExit(
                f'[order] {key} is not unanimous ({sorted(map(str, uniq))}); '
                'the slope chart assumes one rank per cell -- switch to a '
                'distribution plot before using this figure.')
        out[key] = vals[0]
    print(f'[data] ordering probe from {path}')
    return out


def load_order_dist(results_dir, variants=('band_low_only',)):
    if not results_dir:
        return None
    path = os.path.join(results_dir, 'expOrder_8k.json')
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as f:
        doc = json.load(f)
    out = {}
    for r in doc['rows']:
        if r['variant'] not in variants:
            continue
        v = out.setdefault(r['variant'], {'n': 0})
        v.setdefault(r['question'], []).append(r['emitted_rank'])
    for v in out.values():
        v['n'] = max((len(x) for k, x in v.items() if k != 'n'), default=0)
    print(f'[data] ordering distribution from {path}')
    return out


def figure_order(order, outdir, dist=None):
    variants = [('direct', 'direct', BLUE),
                ('ntk_x14', r'NTK $\lambda$=14', GREEN),
                ('pi_x8', 'PI $s$=8', YELLOW)]
    questions = ['first', 'last']

    fig = plt.figure(figsize=(3.386, 2.25))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.20, right=0.775, top=0.80, bottom=0.245)

    NOANS = -0.9
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-1.55, 2.75)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['asked\n"first"', 'asked\n"last"'])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['first\nneedle', 'middle', 'last'])
    ax.set_ylabel('needle reported', labelpad=2)

    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', length=0, pad=3)

    ax.plot([0, 1], [0, 2], color='#BBBBBB', lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(0.60, 0.50, 'correct', color='#888888', fontsize=8.4,
            ha='left', va='center', zorder=1)

    NAME_X = 1.055
    for key, label, color in variants:
        ys = []
        for q in questions:
            v = order.get((key, q))
            ys.append(NOANS if not isinstance(v, int) else v)
        if all(not isinstance(order.get((key, q)), int) for q in questions):
            ax.plot([0, 1], ys, color=color, lw=1.6, marker='x', ms=5.5,
                    mew=1.3, zorder=3, clip_on=False)
            ax.text(NAME_X, NOANS, label, fontsize=8.4, va='center', ha='left',
                    color='black')
            ax.text(0.5, NOANS - 0.42, 'no answer', fontsize=8.4, va='top',
                    ha='center', color='#666666')
        else:
            ax.plot([0, 1], ys, color=color, lw=1.8, marker='o', ms=5.2,
                    mec='black', mew=0.5, zorder=4)
            ax.text(NAME_X, ys[0], label, fontsize=8.4, va='center', ha='left',
                    color='black')

    fig.savefig(os.path.join(outdir, 'fig4_order.pdf'), dpi=300,
                facecolor='white')
    fig.savefig(os.path.join(outdir, 'fig4_order.png'), dpi=300,
                facecolor='white')
    plt.close(fig)
    print('[save]', os.path.join(outdir, 'fig4_order.pdf'))


def figure_matrix(matrix, lengths, depths, outdir, single_column=False,
                  row_depths=(0.0, 0.5, 1.0)):
    order = ['direct', 'ntk_x14', 'pi_x8', 'yarn_8']
    names_sc = {'direct': 'Direct (no scaling)', 'ntk_x14': r'NTK, $\lambda$=14',
                'pi_x8': 'Positional interpolation', 'yarn_8': 'YaRN, causal default'}
    keep = [min(range(len(depths)), key=lambda i: abs(depths[i] - d))
            for d in row_depths]
    depths = [depths[i] for i in keep]
    if single_column:
        fig, axes = plt.subplots(4, 1, figsize=(3.4, 4.15))
        axes = axes.reshape(4, 1)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(7.008, 2.15))
    for pi, (ax, key) in enumerate(zip(axes.ravel(), order)):
        data = np.array([[matrix[key].get(L, [np.nan] * len(depths))[i]
                          for L in lengths] for i in keep])
        cmap = _ramp(PANEL_COLOR[key])
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect='auto',
                       origin='lower')
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                        fontsize=9.0 if not single_column else 8,
                        color=_cell_text_color(v))
        ax.set_xticks(range(len(lengths)))
        ax.set_xticklabels([f'{L // 1024}K' for L in lengths])
        ax.set_yticks(range(len(depths)))
        ax.set_yticklabels([f'{d:g}' for d in depths])
        if single_column:
            ax.set_title(names_sc[key], fontsize=9.0, pad=1.5)
            ax.tick_params(labelsize=7.5)
            if pi == 3:
                ax.set_xlabel('context length', fontsize=9.0)
            else:
                ax.set_xticklabels([])
            if pi == 1:
                ax.set_ylabel('needle depth', fontsize=9.0)
        else:
            ax.set_title(PANEL_TITLE[key], fontsize=9.0, pad=2)
            ax.tick_params(labelsize=10.5)
            if ax in axes[:, 0]:
                ax.set_ylabel('needle depth', fontsize=9.0)
        for s in ax.spines.values():
            s.set_visible(False)

    if single_column:
        fig.subplots_adjust(left=0.175, right=0.985, top=0.965, bottom=0.090,
                            hspace=0.30)
        return _save(fig, outdir, 'fig2_matrix_col')

    lax = fig.add_axes([0.08, 0.004, 0.84, 0.10])
    lax.set_xlim(0, 4)
    lax.set_ylim(0, 1)
    lax.axis('off')
    grad = np.linspace(0, 1, 128).reshape(1, -1)
    sw_bot, sw_top = 0.58, 0.98
    sw_mid = 0.5 * (sw_bot + sw_top)
    leg_label = {'direct': 'Direct', 'ntk_x14': r'NTK $\lambda$=14',
                 'pi_x8': 'Interpolation', 'yarn_8': 'YaRN'}
    for i, key in enumerate(order):
        lax.imshow(grad, cmap=_ramp(PANEL_COLOR[key]), aspect='auto',
                   extent=(i + 0.06, i + 0.94, sw_bot, sw_top))
        lax.text(i + 0.5, 0.44, leg_label[key], ha='center', va='top', fontsize=9.0)
    lax.text(0.02, sw_mid, '0', ha='right', va='center', fontsize=9.0)
    lax.text(3.98, sw_mid, '100%', ha='left', va='center', fontsize=9.0)
    fig.subplots_adjust(left=0.082, right=0.995, top=0.885, bottom=0.255,
                        wspace=0.16, hspace=0.75)
    return _save(fig, outdir, 'fig2_method_matrix')


def figure_band_errors(band, modes, outdir, panel_b=True):
    if panel_b:
        fig = plt.figure(figsize=(7.008, 2.10))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.30,
                              left=0.115, right=0.99, top=0.91, bottom=0.27)
    else:
        _LABEL_FRAC = 0.19
        _PAD = 0.035
        fig = plt.figure(figsize=(3.386, 1.86))
        gs = fig.add_gridspec(1, 1, left=_LABEL_FRAC + _PAD, right=1.0 - _PAD,
                              top=0.855, bottom=0.175)

    ax = fig.add_subplot(gs[0, 0])
    order = ['direct', 'ntk_x14', 'full', 'high_only', 'low_only', 'smooth',
             'yarn_default', 'yarn_tuned']
    names = {'direct': 'direct', 'ntk_x14': r'NTK $\lambda$=14', 'full': 'full band',
             'high_only': 'high-freq', 'low_only': 'low-freq',
             'smooth': 'smooth ramp', 'yarn_default': 'YaRN',
             'yarn_tuned': 'YaRN tuned'}

    if not panel_b:
        v8 = [band[k].get(8192, np.nan) for k in order]
        v24 = [band[k].get(24576, np.nan) for k in order]
        data = np.array([[v8[i], v24[i]] for i in range(len(order))])
        cmap = _ramp(GREEN)
        ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect='auto',
                  origin='upper')
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isnan(val):
                    continue
                ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                        fontsize=9.0,
                        color='#222222' if val < 60 else '#1d3d1c')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['8K', '24K'], fontsize=9.0)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([names[k] for k in order], fontsize=9.0)
        ax.tick_params(length=0, labelsize=9.0)
        for s in ax.spines.values():
            s.set_visible(False)
        for i in range(1, len(order)):
            ax.axhline(i - 0.5, color='white', lw=1.2)
        ax.axvline(0.5, color='white', lw=1.2)
        ax.set_title('exact retrieval (%), depth 0.5', fontsize=9.0, pad=3)
        return _save(fig, outdir, 'fig3_band')

    y = np.arange(len(order))
    v8 = [band[k].get(8192, np.nan) for k in order]
    v24 = [band[k].get(24576, np.nan) for k in order]
    ax.barh(y + 0.12, v8, height=0.22, color=BAR_8K, label='8K')
    ax.barh(y - 0.12, v24, height=0.22, color=BAR_24K, label='24K')
    for yi, (a8, a24) in enumerate(zip(v8, v24)):
        near_zero = (not np.isnan(a8) and not np.isnan(a24) and abs(a8 - a24) < 6.0)
        if not np.isnan(a8):
            ax.text(a8 + 2.0, yi + 0.12, f'{a8:.0f}', va='center', ha='left',
                    fontsize=9.0, color='#222222')
        if not np.isnan(a24):
            ax.text(a24 + 2.0 + (7.5 if near_zero else 0.0), yi - 0.12,
                    f'{a24:.0f}', va='center', ha='left',
                    fontsize=9.0, color='#222222')
    ax.set_yticks(y)
    ax.set_yticklabels([names[k] for k in order], fontsize=9.0)
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xlabel('exact retrieval accuracy (%)', fontsize=9.0)
    ax.set_title('(a) frequency-band ablations, depth 0.5', fontsize=9.0, pad=3)
    ax.tick_params(labelsize=9.0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

    ax2 = fig.add_subplot(gs[0, 1])
    method_order = ['ntk', 'dynamic', 'direct', 'yarn', 'pi']
    labels = {'ntk': 'NTK (matched)', 'dynamic': 'dynamic NTK', 'direct': 'direct',
              'yarn': 'YaRN (default)', 'pi': 'PI'}
    y2 = np.arange(len(method_order))
    left = np.zeros(len(method_order))
    for cat in ERROR_ORDER:
        vals = []
        for k in method_order:
            counts = modes[k]
            total = sum(counts)
            vals.append(100.0 * counts[ERROR_ORDER.index(cat)] / total if total else 0)
        ax2.barh(y2, vals, left=left, height=0.6, color=ERROR_COLORS[cat],
                 label=cat.replace('_', ' '))
        left += np.array(vals)
    ax2.set_yticks(y2)
    ax2.set_yticklabels([labels[k] for k in method_order], fontsize=9.0)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 100)
    ax2.set_xlabel('share of cells (%)', fontsize=9.0)
    ax2.set_title('(b) answer-failure composition', fontsize=9.0, pad=3)
    ax2.tick_params(labelsize=7.5)
    for s in ('top', 'right'):
        ax2.spines[s].set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1, l1, loc='lower left', bbox_to_anchor=(0.10, 0.005), ncol=2,
               frameon=False, fontsize=9.0)
    fig.legend(h2, l2, loc='lower right', bbox_to_anchor=(0.985, 0.005), ncol=3,
               frameon=False, fontsize=9.0)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.92, bottom=0.24,
                        wspace=0.34)
    return _save(fig, outdir, 'fig3_band_errors')


def verify(results_dir):
    import numpy as np

    def band_exact(path):
        rows = json.load(open(path, encoding='utf-8'))['rows']
        agg = {}
        for r in rows:
            k = r['variant']
            if k.startswith('band_'):
                k = k[len('band_'):]
            if k == 'yarn':
                s = r.get('spec') or {}
                bf = float(s.get('beta_fast', 32)) if s.get('beta_fast') is not None else 32.0
                k = 'yarn_tuned' if bf < 32 else 'yarn_default'
            exact = 100.0 if str(r['keyword']) in r['prediction'] else 0.0
            agg.setdefault(k, {}).setdefault(r['context_tokens'], []).append(exact)
        return {k: {L: float(np.mean(v)) for L, v in d.items()} for k, d in agg.items()}

    def matrix_mean(path):
        rows = json.load(open(path, encoding='utf-8'))['rows']
        agg = {}
        for r in rows:
            agg.setdefault((r['method'], r['factor'], r['context_tokens']), []).append(r['score'])
        return {k: float(np.mean(v)) for k, v in agg.items()}

    problems = []
    print('--- matrix (Fig. 2) ---')
    real = matrix_mean(os.path.join(results_dir, 'expM_matrix.json'))
    for disp, (meth, fac) in {'direct': ('direct', 1.0), 'ntk_x14': ('ntk', 14.0),
                              'pi_x8': ('pi', 8.0), 'yarn_8': ('yarn', 8.0),
                              'dyn_16': ('dynamic_ntk', 16.0)}.items():
        for L in D.LENGTHS:
            tr = float(np.mean(D.MATRIX[disp][L]))
            rv = real.get((meth, fac, float(L)))
            if rv is None or abs(tr - rv) > 0.05:
                problems.append(f'matrix {disp} {L}: transcribed={tr} real={rv}')
    print(f'  {len(D.MATRIX) * len(D.LENGTHS)} cells checked')

    print('--- band (Fig. 3) ---')
    rb = band_exact(os.path.join(results_dir, 'expB2_freqband.json'))
    n = 0
    for disp, d in D.BAND.items():
        for L, tr in d.items():
            rv = rb.get(disp, {}).get(L)
            n += 1
            if rv is None or abs(tr - rv) > 0.05:
                problems.append(f'band {disp} {L}: transcribed={tr} real={rv}')
    print(f'  {n} cells checked')

    print('--- error modes ---')
    rows = json.load(open(os.path.join(results_dir, 'analysis_matrix.json'),
                          encoding='utf-8'))['rows']
    cats = ['exact', 'near', 'wrong_number', 'no_number', 'empty']
    m = {}
    for r in rows:
        k = 'dynamic' if r['method'] == 'dynamic_ntk' else r['method']
        m.setdefault(k, {c: 0 for c in cats})[r['category']] += 1
    for k, tr in D.ERROR_MODES.items():
        rv = tuple(m.get(k, {}).get(c, 0) for c in cats)
        if tr != rv:
            problems.append(f'errors {k}: transcribed={tr} real={rv}')
    print(f'  {len(D.ERROR_MODES)} families checked')

    print('--- ceiling (Table I) ---')
    def cell(path, meth, fac, L, depth):
        rows = json.load(open(path, encoding='utf-8'))['rows']
        v = [r['score'] for r in rows
             if r['method'] == meth and abs(r['factor'] - fac) < 1e-6
             and r['context_tokens'] == L and abs(r['depth'] - depth) < 1e-9]
        return float(np.mean(v)) if v else None
    ceil_p = os.path.join(results_dir, 'expM_ceiling.json')
    n10_p = os.path.join(results_dir, 'expM_ceiling24k_lam55_n10.json')
    for label, meth, fac, tr in [('direct 24K', 'direct', 1.0, D.CEILING['direct'][24576]),
                                 ('ntk31 24K', 'ntk', 31.0, D.CEILING['ntk_x31'][24576]),
                                 ('ntk55 24K', 'ntk', 55.0, D.CEILING['ntk_x55'][24576])]:
        for i, depth in enumerate((0.25, 0.5, 0.75)):
            src = n10_p if (meth == 'ntk' and fac == 55.0 and depth == 0.25
                            and os.path.isfile(n10_p)) else ceil_p
            rv = cell(src, meth, fac, 24576, depth)
            if rv is None or abs(tr[i] - rv) > 0.05:
                problems.append(f'ceiling {label} d={depth}: transcribed={tr[i]} real={rv} ({os.path.basename(src)})')
    print('  9 cells checked')

    print()
    if problems:
        print(f'[verify] {len(problems)} MISMATCH(ES):')
        for p in problems:
            print('   *', p)
        return 1
    print('[verify] all transcribed values agree with the run JSONs')
    return 0


def _save(fig, outdir, stem):
    os.makedirs(outdir, exist_ok=True)
    outs = []
    for ext in ('pdf', 'png'):
        path = os.path.join(outdir, f'{stem}.{ext}')
        fig.savefig(path, dpi=300, facecolor='white')
        outs.append(path)
    plt.close(fig)
    print('[save]', ', '.join(outs))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=None)
    ap.add_argument('--outdir', default='figures')
    ap.add_argument('--no-panel-b', dest='panel_b', action='store_false')
    ap.add_argument('--fig2-single-column', dest='fig2_col', action='store_true')
    ap.add_argument('--verify', metavar='RESULTS_DIR', default=None)
    args = ap.parse_args()

    if args.verify:
        raise SystemExit(verify(args.verify))

    matrix, lengths, depths = load_matrix(args.results)
    band = load_band(args.results)
    modes = load_errors(args.results)

    figure_matrix(matrix, lengths, depths, args.outdir,
                  single_column=args.fig2_col)
    figure_band_errors(band, modes, args.outdir, panel_b=args.panel_b)

    order = load_order(args.results)
    if order:
        figure_order(order, args.outdir,
                     dist=load_order_dist(args.results))


if __name__ == '__main__':
    main()
