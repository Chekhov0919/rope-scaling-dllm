import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BLUE, GREEN, YELLOW, PURPLE, GREY = ('#DAE8FC', '#D5E8D4', '#FFF2CC',
                                     '#E1D5E7', '#EEEEEE')
EDGE, TEXT, MARK_BAD, MARK_OK = '#555555', '#222222', '#B03A2E', '#1E7B34'
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9.0,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'axes.linewidth': 0.6,
})


def token_row(ax, y, n, x0, dx, fill, query_idx, attend, size=0.62,
              label_dy=1.32):
    xs = [x0 + i * dx for i in range(n)]
    for i, x in enumerate(xs):
        is_q = (i == query_idx)
        ax.add_patch(Rectangle((x - size / 2, y - size / 2), size, size,
                               facecolor=fill, edgecolor=EDGE,
                               linewidth=1.5 if is_q else 0.8, zorder=3))
        if is_q:
            ax.add_patch(Rectangle((x - size / 2 - 0.09, y - size / 2 - 0.09),
                                   size + 0.18, size + 0.18, facecolor='none',
                                   edgecolor=EDGE, linewidth=0.7, zorder=4))
    qx = xs[query_idx]
    for i in attend:
        if i == query_idx:
            continue
        dist = min(1.0, abs(i - query_idx) / 4.0)
        rad = 0.30 * dist * (1.0 if xs[i] < qx else -1.0)
        ax.annotate('', xy=(xs[i], y + size / 2 + 0.02),
                    xytext=(qx, y + size / 2 + 0.02),
                    arrowprops=dict(arrowstyle='-|>', color=TEXT, lw=0.8,
                                    shrinkA=1.5, shrinkB=1.5,
                                    connectionstyle=f'arc3,rad={rad}'),
                    zorder=2)
    ax.text(qx, y + label_dy, r'query at $m$', ha='center', va='bottom',
            fontsize=9.0, color=TEXT)


def offset_axis(ax, y, x0, x1, ticks, label=None):
    ax.add_patch(Rectangle((x0, y + 0.10), x1 - x0, 0.26, facecolor=GREY,
                           edgecolor='none', zorder=1))
    ax.annotate('', xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle='-|>', color=EDGE, lw=0.9))
    for x, t in ticks:
        ax.plot([x, x], [y - 0.09, y + 0.09], color=EDGE, lw=0.7)
        ax.text(x, y - 0.26, t, ha='center', va='top', fontsize=9.0, color=TEXT)
    if label:
        ax.text((x0 + x1) / 2, y - 1.15, label, ha='center', va='top',
                fontsize=9.0, color=TEXT)


def panel_a(ax):
    ax.set_xlim(-4.9, 11.4)
    ax.set_ylim(-1.4, 11.6)
    ax.axis('off')

    x0, dx, n, q = 0.9, 0.96, 9, 4
    xlast = x0 + (n - 1) * dx

    token_row(ax, 8.7, n, x0, dx, BLUE, q, attend=range(0, q + 1))
    offset_axis(ax, 7.2, x0 - 0.2, xlast + 0.2,
                [(x0 - 0.2, r'$0$'), (xlast + 0.2, r'$T-1$')])
    ax.text(-4.6, 8.7, 'Causal\n(auto-regressive)', ha='left', va='center',
            fontsize=9.0, color=TEXT)

    token_row(ax, 4.5, n, x0, dx, PURPLE, q, attend=range(0, n), label_dy=1.02)
    offset_axis(ax, 3.1, x0 - 0.2, xlast + 0.2,
                [(x0 - 0.2, r'$-(T-1)$'), ((x0 + xlast) / 2, r'$0$'),
                 (xlast + 0.2, r'$T-1$')],
                label=r'relative offset $m-n$')
    ax.text(-4.6, 4.5, 'Bidirectional\n(diffusion)', ha='left', va='center',
            fontsize=9.0, color=TEXT)
    ax.annotate('', xy=(x0 - 0.55, 0.55), xytext=(x0 + 0.95, 0.25),
                arrowprops=dict(arrowstyle='-|>', color=EDGE, lw=0.7))
    ax.text(x0 + 1.05, 0.18, r'sign of $m-n$ = left / right direction',
            ha='left', va='center', fontsize=9.0, color=TEXT)
    ax.text(2.2, -1.00, '(a) relative position range', ha='center', va='top',
            fontsize=9.0, color=TEXT)


def mark_cross(ax, x, y, w=2.0, h=1.4):
    ax.plot([x - w, x + w], [y - h, y + h], color=MARK_BAD, lw=2.0,
            solid_capstyle='round', zorder=5)
    ax.plot([x - w, x + w], [y + h, y - h], color=MARK_BAD, lw=2.0,
            solid_capstyle='round', zorder=5)


def mark_check(ax, x, y, w=2.2, h=1.5):
    ax.plot([x - w, x - 0.45 * w, x + w], [y, y - h, y + 1.35 * h],
            color=MARK_OK, lw=2.0, solid_capstyle='round',
            solid_joinstyle='round', zorder=5)


def panel_b(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(-16.0, 48)
    ax.axis('off')

    XL, XDIV, XR = 36.0, 68.0, 99.0

    ax.add_patch(Rectangle((XL, 40), XDIV - XL, 6.4, facecolor=YELLOW,
                           edgecolor='none'))
    ax.add_patch(Rectangle((XDIV, 40), XR - XDIV, 6.4, facecolor=GREEN,
                           edgecolor='none'))
    ax.text((XL + XDIV) / 2, 43.2, 'direction-sensitive',
            ha='center', va='center', fontsize=8.4, color=TEXT, linespacing=1.25)
    ax.text((XDIV + XR) / 2, 43.2, 'long-range coverage',
            ha='center', va='center', fontsize=8.4, color=TEXT, linespacing=1.25)
    ax.plot([XDIV, XDIV], [6.0, 40.0], color=EDGE, lw=0.6, ls=(0, (3, 2)),
            zorder=1)

    rows = [
        (r'PI: positions $\div\,s$', 'full', 'bad'),
        (r'NTK: base $\times\,s$', 'full', 'bad'),
        ('YaRN: low band interpolated', 'ramp', None),
        ('Target: stretch low band only', 'low', 'ok'),
    ]
    y = 34.0
    for label, glyph, mark in rows:
        ax.text(XL - 1.5, y, label, ha='right', va='center', fontsize=9.0,
                color=TEXT)
        if glyph == 'full':
            ax.annotate('', xy=(XR, y), xytext=(XL, y),
                        arrowprops=dict(arrowstyle='-|>', color=TEXT, lw=1.0))
        elif glyph == 'ramp':
            ax.plot([XL, XL + 18, XL + 32, XR], [y, y, y + 2.4, y + 2.4],
                    color=TEXT, lw=1.0, solid_capstyle='round')
            ax.annotate('', xy=(XR, y + 2.4), xytext=(XR - 2.0, y + 2.4),
                        arrowprops=dict(arrowstyle='-|>', color=TEXT, lw=1.0))
        else:
            ax.annotate('', xy=(XR, y), xytext=(XDIV + 1.0, y),
                        arrowprops=dict(arrowstyle='-|>', color=GREEN, lw=2.8))
        if mark == 'bad':
            mark_cross(ax, (XL + XDIV) / 2, y)
        elif mark == 'ok':
            mark_check(ax, (XL + XDIV) / 2, y)
        y -= 7.0

    ax.annotate('', xy=(XR, 5.0), xytext=(XL, 5.0),
                arrowprops=dict(arrowstyle='-|>', color=EDGE, lw=0.9))
    ax.text((XL + XR) / 2, 3.4, 'rotary frequency dimension',
            ha='center', va='top', fontsize=9.0, color=TEXT, linespacing=1.25)
    ax.text(XL + 1.0, -5.2, 'higher frequency', ha='left', va='center',
            fontsize=8.4, color=TEXT)
    ax.text(XR - 1.0, -5.2, 'lower frequency', ha='right', va='center',
            fontsize=8.4, color=TEXT)
    ax.annotate('', xy=(XR - 1.0, -8.2), xytext=(XL + 1.0, -8.2),
                arrowprops=dict(arrowstyle='-|>', color=EDGE, lw=0.6))
    ax.text(50, -11.4, '(b) scaling effects on the rotary spectrum',
            ha='center', va='top', fontsize=9.0, color=TEXT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='figures')
    args = ap.parse_args()

    fig = plt.figure(figsize=(7.008, 1.95))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.50, 0.50], wspace=0.08,
                          left=0.015, right=0.99, top=0.97, bottom=0.12)
    panel_a(fig.add_subplot(gs[0, 0]))
    panel_b(fig.add_subplot(gs[0, 1]))

    os.makedirs(args.outdir, exist_ok=True)
    outs = []
    for ext in ('pdf', 'svg', 'png'):
        path = os.path.join(args.outdir, f'fig1_mechanism.{ext}')
        fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.02, facecolor='white')
        outs.append(path)
    plt.close(fig)
    print('[save]', ', '.join(outs))


if __name__ == '__main__':
    main()
