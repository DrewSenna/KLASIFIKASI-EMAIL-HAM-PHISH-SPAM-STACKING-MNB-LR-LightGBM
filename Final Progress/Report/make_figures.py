"""Build the UTS report figures into Final Progress/Report/figures/.

Usage: python3 make_figures.py [fig_name ...]
Gambar 3.1 (fig_wordcount) also needs the raw Hugging Face dataset as CSV:
    DATA_CSV=/path/to/df.csv python3 make_figures.py fig_wordcount
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../Final Progress
OUT = f'{ROOT}/Report/figures'
DATA_CSV = os.environ.get('DATA_CSV', f'{ROOT}/data/df.csv')  # raw dataset (df.csv from Hugging Face)

CLASS_COLORS = {'Ham': '#2a78d6', 'Phish': '#eb6834', 'Spam': '#1baf7a'}
MODEL_COLORS = {'MNB': '#eda100', 'LR': '#6250d6', 'LightGBM': '#e87ba4',
                'Stack A': '#52514e', 'Stack B': '#e34948',
                'Stack MNB+LGBM': '#eda100', 'Stack MNB+LR': '#a65e2e'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'legend.frameon': False, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
})


def idn(x, dec=None):
    """Indonesian number format: 1.234,5"""
    s = f'{x:,.{dec}f}' if dec is not None else f'{x:,}'
    return s.replace(',', '#').replace('.', ',').replace('#', '.')


def fmt(dec):
    return FuncFormatter(lambda v, _: idn(v, dec))


def save(fig, name):
    fig.savefig(f'{OUT}/{name}', facecolor='white')
    plt.close(fig)
    print('saved', name)


# ---------------------------------------------------------------- Gambar 1: pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(8.2, 9.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12.6); ax.axis('off')

    def box(x, y, w, h, title, body='', fc='#f4f3ef', ec=INK2):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.02,rounding_size=0.12',
                                    fc=fc, ec=ec, lw=1))
        if body:
            ax.text(x, y + 0.17, title, ha='center', va='center', fontsize=9.5, weight='bold', color=INK)
            ax.text(x, y - 0.2, body, ha='center', va='center', fontsize=8.5, color=INK2)
        else:
            ax.text(x, y, title, ha='center', va='center', fontsize=9.5, weight='bold', color=INK)

    def arrow(x1, y1, x2, y2, style='-|>', ls='-'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=INK2, lw=1, ls=ls, shrinkA=0, shrinkB=0))

    box(5, 12.0, 6.4, 0.8, 'Data mentah', '365.448 pesan (Ham, Phish, Spam)')
    arrow(5, 11.6, 5, 11.2)
    box(5, 10.8, 6.4, 0.8, 'Pembersihan dan deduplikasi', 'teks kosong, label berbeda, duplikat → 278.941 pesan')
    arrow(5, 10.4, 5, 10.0)
    box(5, 9.6, 6.4, 0.8, 'Pembagian stratified 80/20', 'random_state = 42')
    # training branch (left) and testing branch (right)
    arrow(3.6, 9.2, 3.0, 8.7); arrow(6.4, 9.2, 8.0, 8.7)
    box(3.0, 8.3, 4.4, 0.8, 'Data training', '223.152 pesan, 5-fold cross-validation')
    box(8.0, 8.3, 3.2, 0.8, 'Data testing', '55.789 pesan')
    arrow(3.0, 7.9, 3.0, 7.5)
    box(3.0, 7.1, 4.4, 0.8, 'Teks V3', 'format sumber diseragamkan')
    for x, name, sub in [(1.0, 'MNB', 'count uni+bigram'), (3.0, 'LR', 'TF-IDF unigram'), (5.0, 'LightGBM', 'count unigram')]:
        arrow(3.0, 6.7, x, 6.15)
        box(x, 5.75, 1.8, 0.8, name, sub, fc='white', ec=MODEL_COLORS[name if name != 'MNB' else 'MNB'])
        arrow(x, 5.35, 3.0, 4.85)
    box(3.0, 4.45, 4.4, 0.8, 'Meta-model', 'Logistic Regression atas log-probabilitas')
    arrow(3.0, 4.05, 3.0, 3.65)
    box(3.0, 3.25, 4.4, 0.8, 'Threshold Ham', 'log P(Ham) + b, b = 1,777')
    # testing path: used only at the end
    ax.plot([8.0, 8.0], [7.9, 3.25], color=INK2, lw=1, ls=(0, (4, 3)))
    arrow(8.0, 3.25, 5.25, 3.25, ls='--')
    ax.text(8.15, 5.6, 'dipakai satu kali,\nsetelah model\ndan b ditetapkan', fontsize=8.5, color=INK2, va='center')
    arrow(3.0, 2.85, 3.0, 2.35)
    for x, label, cls in [(1.4, 'Kotak masuk', 'Ham'), (4.0, 'Folder spam', 'Spam'),
                          (6.8, 'Folder spam +\nperingatan phishing', 'Phish')]:
        arrow(3.0, 2.35, x, 1.75)
        ax.add_patch(FancyBboxPatch((x - 1.15, 0.85), 2.3, 0.85, boxstyle='round,pad=0.02,rounding_size=0.12',
                                    fc='white', ec=CLASS_COLORS[cls], lw=1.6))
        ax.text(x, 1.45, f'Prediksi {cls}', ha='center', va='center', fontsize=8.5, color=INK2)
        ax.text(x, 1.12, label, ha='center', va='center', fontsize=9, weight='bold', color=INK,
                linespacing=1.0)
    save(fig, 'gambar01_pipeline.png')


# ---------------------------------------------------------------- Gambar 2: word count boxplot (crop of M03 B.6.2)
def fig_wordcount():
    import csv, sys
    csv.field_size_limit(sys.maxsize)
    labels, counts = [], []
    with open(DATA_CSV, newline='', encoding='utf-8') as f:
        r = csv.reader(f); head = next(r); li, ti = head.index('label'), head.index('text')
        for row in r:
            labels.append(int(row[li])); counts.append(len(row[ti].split()))
    print('rows', len(labels))
    labels = np.array(labels); wc = np.array(counts)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    names = ['Ham', 'Phish', 'Spam']
    data = [np.log10(1 + wc[labels == i]) for i in range(3)]
    bp = ax.boxplot(data, widths=0.5, showfliers=False, patch_artist=True,
                    medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=INK2),
                    capprops=dict(color=INK2), boxprops=dict(lw=0))
    for patch, n in zip(bp['boxes'], names):
        patch.set_facecolor(CLASS_COLORS[n])
    for i, n in enumerate(names):
        med = int(np.median(wc[labels == i]))
        print(n, 'median', med)
        ax.text(i + 1.29, np.log10(1 + med), f'median\n{idn(med)} kata', va='center', fontsize=8.5, color=INK)
    ticks = [1, 10, 100, 1000, 10000]
    ax.set_yticks(np.log10(1 + np.array(ticks)))
    ax.set_yticklabels([idn(t) for t in ticks])
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(names)
    ax.set_xlim(0.5, 3.85)
    ax.set_ylabel('Jumlah kata per pesan (skala log)')
    ax.grid(axis='x', visible=False)
    thr = 590.5  # batas pencilan Subbab 3.4: Q3 + 1,5 x IQR jumlah kata seluruh data mentah
    ax.axhline(np.log10(1 + thr), color=INK2, lw=1.1, ls='--', zorder=1)
    ax.text(3.83, np.log10(1 + thr) + 0.04, 'batas pencilan (Subbab 3.4): 590,5 kata', ha='right',
            va='bottom', fontsize=8.5, color=INK2,
            bbox=dict(facecolor='white', edgecolor='none', pad=1.5))
    ax.set_title('Panjang pesan per kelas (data mentah)', loc='left')
    save(fig, 'gambar02_panjang_pesan.png')


# ---------------------------------------------------------------- Gambar 3: source-format traits
def fig_format():
    # Data training. Angka asli dan escape…: M06 (B.6); ciri lain: M07 R2 (audit penyeragaman, V0).
    traits = ['Memuat angka asli', 'Memuat placeholder escape…', 'Memuat timestamp header',
              'Memuat kata "the"', 'Memuat tanda baca', 'Tautan menyatu\n(httpwwwsitecom)']
    vals = {'Ham': [12.2, 75.1, 41.1, 65.6, 54.4, 0.0],
            'Phish': [93.4, 0.3, 67.6, 0.0, 0.0, 39.1],
            'Spam': [10.2, 67.0, 10.8, 51.8, 49.2, 0.0]}
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    n = len(traits); bh = 0.26; gap = 0.02
    for i, cls in enumerate(['Ham', 'Phish', 'Spam']):
        ys = np.arange(n) + (i - 1) * (bh + gap)
        ax.barh(ys, vals[cls], height=bh, color=CLASS_COLORS[cls], label=cls)
        for yv, v in zip(ys, vals[cls]):
            ax.text(v + 1, yv, idn(v, 1) + '%', va='center', fontsize=8, color=INK)
    ax.set_yticks(np.arange(n)); ax.set_yticklabels(traits)
    ax.invert_yaxis()
    ax.set_xlim(0, 105); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax.set_xlabel('Persentase pesan pada kelas tersebut (%)')
    ax.grid(axis='y', visible=False)
    ax.legend(loc='lower right', frameon=False)
    ax.set_title('Ciri format sumber data per kelas (data training)', loc='left')
    save(fig, 'gambar03_format_sumber.png')


# ---------------------------------------------------------------- Gambar 4: text versions V0-V3
def fig_versions():
    x = np.arange(4)
    mnb = [0.9344, 0.8956, 0.8900, 0.8016]
    lr = [0.9687, 0.9476, 0.9423, 0.8658]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.axvspan(-0.25, 2, color='#f4f3ef', zorder=0)
    ax.axvspan(2, 3.25, color='#eceae4', zorder=0)
    ax.text(1.0, 0.985, 'Hipotesis 4: angka dan tanggal', ha='center', fontsize=8.5, color=INK2)
    ax.text(2.62, 0.985, 'Hipotesis 5: stopword\ndan tautan', ha='center', fontsize=8.5, color=INK2, va='center')
    for name, v in [('LR', lr), ('MNB', mnb)]:
        ax.plot(x, v, color=MODEL_COLORS[name], lw=2, marker='o', ms=7, zorder=3)
        full = 'Logistic Regression' if name == 'LR' else 'Multinomial Naive Bayes'
        ax.text(x[0] - 0.08, v[0], f'{full}\n{idn(v[0], 4)}', ha='right', va='center', fontsize=8.5, color=INK)
        ax.text(x[-1] + 0.08, v[-1], idn(v[-1], 4), ha='left', va='center', fontsize=8.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(['V0\nteks asli', 'V1\nangka', 'V2\ntimestamp', 'V3\nstopword,\ntautan'])
    ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_xlim(-1.35, 3.5); ax.set_ylim(0.78, 1.0)
    ax.yaxis.set_major_formatter(fmt(2))
    ax.set_ylabel('Macro F1 (rata-rata 5 fold)')
    ax.grid(axis='x', visible=False)
    ax.set_title('Macro F1 setelah format sumber diseragamkan', loc='left')
    save(fig, 'gambar04_versi_teks.png')


# ---------------------------------------------------------------- Gambar 5: tuning curves
def fig_tuning():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    alpha = [0.25, 0.5, 1, 2, 4, 8]
    f1 = [0.8169, 0.8201, 0.8283, 0.8356, 0.8251, 0.7811]
    a1.plot(alpha, f1, color=MODEL_COLORS['MNB'], lw=2, marker='o', ms=7)
    a1.set_xscale('log')
    a1.xaxis.set_major_locator(FixedLocator(alpha)); a1.xaxis.set_minor_locator(FixedLocator([]))
    a1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: idn(v, 2).rstrip('0').rstrip(',')))
    a1.yaxis.set_major_formatter(fmt(2))
    a1.annotate(f'terbaik: α = 2\n{idn(0.8356, 4)}', xy=(2, 0.8356), xytext=(1.35, 0.806), fontsize=8.5,
                color=INK, arrowprops=dict(arrowstyle='-', color=INK2, lw=0.8))
    a1.set_xlabel('α (skala log)'); a1.set_ylabel('Macro F1')
    a1.set_title('(a) Multinomial Naive Bayes: α, 5 fold', loc='left')

    trees = [25, 50, 75, 100, 125, 150]
    tr = [0.8481, 0.8804, 0.8943, 0.9064, 0.9161, 0.9208]
    va = [0.8446, 0.8668, 0.8756, 0.8816, 0.8875, 0.8868]
    a2.plot(trees, tr, color='#8a8984', lw=2, ls='--', marker='o', ms=6, label='Data training')
    a2.plot(trees, va, color=INK, lw=2, marker='o', ms=7, label='Data validasi')
    a2.axvline(125, color=INK2, lw=0.8, ls=':')
    a2.text(150 + 4, tr[-1], 'training', va='center', fontsize=8.5, color=INK2)
    a2.text(150 + 4, va[-1], 'validasi', va='center', fontsize=8.5, color=INK)
    a2.text(126, 0.848, 'dipilih: 125 pohon', fontsize=8.5, color=INK2)
    a2.set_xticks(trees); a2.set_xlim(18, 172)
    a2.yaxis.set_major_formatter(fmt(2))
    a2.set_xlabel('Jumlah pohon'); a2.set_ylabel('Macro F1')
    a2.legend(loc='upper left')
    a2.set_title('(b) LightGBM: jumlah pohon, fold 1', loc='left')
    fig.tight_layout(w_pad=3)
    save(fig, 'gambar07_tuning.png')


# ---------------------------------------------------------------- Gambar 6: learning curves
def fig_learning():
    n = [17852, 44630, 89260, 178521]
    data = {'Multinomial Naive Bayes': ([0.8181, 0.8529, 0.8853, 0.8816], [0.7185, 0.7757, 0.8204, 0.8344]),
            'Logistic Regression': ([0.9948, 0.9825, 0.9751, 0.9533], [0.8363, 0.8493, 0.8554, 0.8630])}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for ax, (name, (tr, va)), tag in zip(axes, data.items(), ['(a)', '(b)']):
        ax.plot(n, tr, color='#8a8984', lw=2, ls='--', marker='o', ms=6, label='Data training')
        ax.plot(n, va, color=INK, lw=2, marker='o', ms=7, label='Data validasi')
        gap = tr[-1] - va[-1]
        ax.annotate('', xy=(n[-1] + 6000, va[-1]), xytext=(n[-1] + 6000, tr[-1]),
                    arrowprops=dict(arrowstyle='<->', color=INK2, lw=0.9))
        ax.text(n[-1] + 9000, (tr[-1] + va[-1]) / 2, f'selisih\n{idn(gap, 3)}', va='center', fontsize=8.5, color=INK2)
        ax.set_xlim(0, 215000)
        ax.set_xticks([0, 50000, 100000, 150000, 200000])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: idn(int(v / 1000))))
        ax.yaxis.set_major_formatter(fmt(2))
        ax.set_xlabel('Jumlah pesan data training, fold 1 (ribu)')
        ax.set_title(f'{tag} {name}', loc='left')
    axes[0].set_ylabel('Macro F1')
    axes[0].legend(loc='lower right')
    fig.tight_layout(w_pad=3)
    save(fig, 'gambar06_learning_curve.png')


# ---------------------------------------------------------------- Gambar 7: Ham-threshold trade-off on V3 OOF
def fig_tradeoff():
    d = np.load(f'{ROOT}/M07/m07_artifacts/oof_log_proba.npz')
    y = d['y_train']
    models = [('LR', 'LR_V3', 'Logistic Regression'),
              ('LightGBM', 'LightGBM_V3', 'LightGBM'),
              ('Stack MNB+LGBM', 'Stacking_MNB_+_LightGBM', 'Stacking MNB + LightGBM'),
              ('Stack MNB+LR', 'Stacking_MNB_+_LR', 'Stacking MNB + LR'),
              ('Stack A', 'Stacking_LR_+_LightGBM', 'Stacking LR + LightGBM'),
              ('Stack B', 'Stacking_MNB_+_LR_+_LightGBM', 'Stacking MNB + LR + LightGBM (model akhir)')]
    lost = np.linspace(0.002, 0.015, 60)  # share of Ham lost
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for key, arr, label in models:
        L = d[arr]
        m = np.maximum(L[:, 1], L[:, 2]) - L[:, 0]
        dh = m[y == 0]
        bs = np.quantile(dh, 1 - lost)
        ph = np.array([(m[y == 1] <= b).sum() for b in bs])
        sh = np.array([(m[y == 2] <= b).sum() for b in bs])
        bold = key.startswith('Stack')
        for ax, v in zip(axes, [ph, sh]):
            ax.plot(lost * 100, v, color=MODEL_COLORS[key], lw=2.4 if key == 'Stack B' else 1.6,
                    ls={'Stack A': '--', 'Stack MNB+LGBM': (0, (6, 2, 1, 2)), 'Stack MNB+LR': (0, (1.5, 1.5))}.get(key, '-'), label=label, zorder=3 if bold else 2)
    for ax, title, tag in zip(axes, ['Phishing masuk kotak masuk (Phish → Ham)', 'Spam masuk kotak masuk (Spam → Ham)'],
                              ['(a)', '(b)']):
        ax.axvline(0.5, color=INK2, lw=0.9, ls=':')
        ax.text(0.52, ax.get_ylim()[1] * 0.96, 'target: Ham Recall 99,5%', fontsize=8.5, color=INK2, va='top')
        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: idn(int(v))))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: idn(int(v)) if str(int(v))[0] in '25' else ''))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: idn(v, 1) + '%'))
        ax.set_xlabel('Email sah hilang (% dari Ham)')
        ax.set_ylabel('Jumlah pesan (skala log)')
        ax.set_title(f'{tag} {title}', loc='left', fontsize=10)
        ax.grid(which='minor', color=GRID, linewidth=0.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout(w_pad=3)
    save(fig, 'gambar08_tradeoff_threshold.png')


# ---------------------------------------------------------------- Gambar 8: confusion matrices on test
def fig_confusion():
    cms = {'(a) Baseline': np.array([[25374, 7, 167], [538, 4555, 3475], [1991, 65, 19617]]),
           '(b) Model akhir, b = 1,777': np.array([[25454, 5, 89], [107, 5913, 2548], [781, 612, 20280]])}
    cls = ['Ham', 'Phish', 'Spam']
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for ax, (title, cm) in zip(axes, cms.items()):
        pct = cm / cm.sum(axis=1, keepdims=True) * 100
        ax.imshow(pct, cmap='Blues', vmin=0, vmax=100)
        for i in range(3):
            for j in range(3):
                c = 'white' if pct[i, j] > 55 else INK
                ax.text(j, i, f'{idn(pct[i, j], 2)}%\n({idn(int(cm[i, j]))})', ha='center', va='center',
                        fontsize=9, color=c)
        ax.set_xticks(range(3)); ax.set_xticklabels(cls); ax.set_yticks(range(3)); ax.set_yticklabels(cls)
        ax.set_xlabel('Prediksi'); ax.set_ylabel('Aktual')
        ax.grid(False)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(title, loc='left')
    fig.tight_layout(w_pad=3)
    save(fig, 'gambar05_confusion_matrix.png')


if __name__ == '__main__':
    ALL = [fig_pipeline, fig_wordcount, fig_format, fig_versions, fig_tuning, fig_learning, fig_tradeoff, fig_confusion]
    chosen = sys.argv[1:]  # contoh: python make_figures.py fig_format
    for f in ALL:
        if not chosen or f.__name__ in chosen:
            f()
