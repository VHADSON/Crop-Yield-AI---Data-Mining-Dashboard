import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ── Palette — Light & Colorful ─────────────────────────────────────────────
BG     = '#F4F6FB'
CARD   = '#FFFFFF'
BORDER = '#D4DAE8'
INDIGO = '#4F6EF7'
TEAL   = '#0EA88C'
AMBER  = '#E8960A'
ROSE   = '#E53E3E'
VIOLET = '#7C4DFF'
SKY    = '#0EA5E9'
ORANGE = '#F97316'
MUTED  = '#64748B'
TEXT   = '#1A2340'
GRID   = '#E8ECF3'

CLASS_COLORS = {'Rendah': ROSE, 'Sedang': AMBER, 'Tinggi': TEAL}
MODEL_COLORS = {'Decision Tree': INDIGO, 'Naive Bayes': TEAL, 'XGBoost': ORANGE}


def _apply_theme(fig, axes):
    fig.patch.set_facecolor(BG)
    if not hasattr(axes, '__iter__'):
        axes = [axes]
    for ax in np.array(axes).flat:
        ax.set_facecolor(CARD)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
            spine.set_linewidth(1.2)
        ax.xaxis.label.set_color(MUTED)
        ax.yaxis.label.set_color(MUTED)
        ax.title.set_color(TEXT)
        ax.title.set_fontsize(12)
        ax.title.set_fontweight('bold')
        ax.tick_params(colors=MUTED, which='both', labelsize=9)
        ax.grid(True, color=GRID, linestyle='-', linewidth=0.8, alpha=1.0)
        ax.set_axisbelow(True)


# ── Boxplots ───────────────────────────────────────────────────────────────
def plot_boxplots(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    cols = [
        ('average_rain_fall_mm_per_year', 'Curah Hujan (mm/tahun)', SKY),
        ('pesticides_tonnes',             'Penggunaan (tonnes)',     VIOLET),
        ('avg_temp',                      'Suhu (°C)',               AMBER),
    ]
    titles = [
        'Vis 1 — Boxplot Curah Hujan',
        'Vis 2 — Boxplot Pestisida',
        'Vis 3 — Boxplot Rata-rata Suhu',
    ]
    for ax, (col, ylabel, color), title in zip(axes, cols, titles):
        bp = ax.boxplot(
            df[col].dropna(), vert=True, patch_artist=True, widths=0.5,
            medianprops=dict(color=color, linewidth=2.5),
            flierprops=dict(marker='o', markerfacecolor=color,
                            markeredgecolor='none', alpha=0.35, markersize=3.5),
            whiskerprops=dict(color=MUTED, linewidth=1.3),
            capprops=dict(color=MUTED, linewidth=1.5),
        )
        for patch in bp['boxes']:
            patch.set_facecolor(color + '22')
            patch.set_edgecolor(color)
            patch.set_linewidth(1.8)
        ax.set_title(title, pad=14)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks([])
    _apply_theme(fig, axes)
    plt.tight_layout(pad=2.5)
    return fig


# ── Distribusi Yield ──────────────────────────────────────────────────────
def plot_distribution(df):
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.histplot(df['hg/ha_yield'], kde=True, ax=ax,
                 color=INDIGO, edgecolor='white', linewidth=0.6, alpha=0.65,
                 line_kws={'color': TEAL, 'linewidth': 2.5})
    ax.set_title('Vis 4 — Distribusi Hasil Panen (Target Awal)', pad=14)
    ax.set_xlabel('hg/ha Yield')
    ax.set_ylabel('Frekuensi')
    _apply_theme(fig, ax)
    plt.tight_layout(pad=2.5)
    return fig


# ── Keseimbangan Kelas ────────────────────────────────────────────────────
def plot_target_balance(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    order  = ['Rendah', 'Sedang', 'Tinggi']
    counts = df['Yield_Class'].value_counts().reindex(order)
    colors = [CLASS_COLORS[c] for c in order]
    bars   = ax.bar(order, counts.values,
                    color=[c + 'CC' for c in colors],
                    edgecolor=colors, linewidth=2, width=0.5, zorder=3)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
                f'{val:,}', ha='center', va='bottom',
                color=TEXT, fontsize=10, fontweight='bold')
    ax.set_title('Vis 5 — Keseimbangan Kelas Target', pad=14)
    ax.set_xlabel('Kelas Yield')
    ax.set_ylabel('Jumlah Data')
    _apply_theme(fig, ax)
    plt.tight_layout(pad=2.5)
    return fig


# ── Confusion Matrix ──────────────────────────────────────────────────────
def plot_confusion_matrix(cm, labels, title, cmap='Blues'):
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, color=TEXT)
    ax.set_yticklabels(labels, color=TEXT)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title, pad=14)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    color='white' if cm[i, j] > thresh else TEXT)
    _apply_theme(fig, ax)
    plt.tight_layout(pad=2.5)
    return fig


# ── Perbandingan 4 Metrik (support 2 atau 3 model) ───────────────────────
def plot_metrics_comparison(dt_metrics, nb_metrics, xgb_metrics=None):
    labels    = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    models    = [('Decision Tree', dt_metrics, INDIGO),
                 ('Naive Bayes',   nb_metrics, TEAL)]
    if xgb_metrics is not None:
        models.append(('XGBoost', xgb_metrics, ORANGE))

    n = len(models)
    w = 0.22 if n == 3 else 0.32
    offsets = np.linspace(-(n-1)*w/2, (n-1)*w/2, n)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12, 5))
    for offset, (name, res, color) in zip(offsets, models):
        vals = [res['accuracy'], res['precision'], res['recall'], res['f1']]
        bars = ax.bar(x + offset, [v * 100 for v in vals], w,
                      label=name, color=color + 'CC',
                      edgecolor=color, linewidth=1.5, zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.4,
                    f'{h:.1f}%', ha='center', va='bottom',
                    color=TEXT, fontsize=7, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 118)
    ax.set_title('Perbandingan Metrik Evaluasi: DT vs NB' + (' vs XGBoost' if xgb_metrics else ''),
                 pad=14)
    ax.set_ylabel('Persentase (%)')
    ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, framealpha=1)
    _apply_theme(fig, ax)
    plt.tight_layout(pad=2.5)
    return fig


# ── F1 Per Kelas Line ─────────────────────────────────────────────────────
def plot_f1_per_class_line(dt_f1, nb_f1, xgb_f1=None):
    classes = ['Rendah', 'Sedang', 'Tinggi']
    fig, ax = plt.subplots(figsize=(9, 5))

    model_series = [('Decision Tree', dt_f1, INDIGO, 'o-'),
                    ('Naive Bayes',   nb_f1, TEAL,   's--')]
    if xgb_f1 is not None:
        model_series.append(('XGBoost', xgb_f1, ORANGE, '^:'))

    for name, f1, color, style in model_series:
        ax.plot(classes, [v * 100 for v in f1], style, color=color,
                linewidth=2.5, markersize=9,
                markerfacecolor=color, markeredgecolor='white', markeredgewidth=1.5,
                label=name, zorder=4)
        for i, v in enumerate(f1):
            ax.annotate(f'{v*100:.1f}%', (classes[i], v*100),
                        textcoords='offset points', xytext=(0, 12),
                        ha='center', color=color, fontsize=8, fontweight='bold')

    ax.set_title('F1-Score Per Kelas: Perbandingan Model', pad=14)
    ax.set_ylabel('F1-Score (%)')
    ax.set_ylim(0, 120)
    ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, framealpha=1)
    _apply_theme(fig, ax)
    plt.tight_layout(pad=2.5)
    return fig


# ── F1 Per Kelas Bar ──────────────────────────────────────────────────────
def plot_f1_per_class_bar(dt_f1, nb_f1, xgb_f1=None):
    classes = ['Rendah', 'Sedang', 'Tinggi']
    model_series = [('Decision Tree', dt_f1, INDIGO),
                    ('Naive Bayes',   nb_f1, TEAL)]
    if xgb_f1 is not None:
        model_series.append(('XGBoost', xgb_f1, ORANGE))

    n = len(model_series)
    w = 0.22 if n == 3 else 0.32
    offsets = np.linspace(-(n-1)*w/2, (n-1)*w/2, n)
    x = np.arange(len(classes))

    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, (name, f1, color) in zip(offsets, model_series):
        bars = ax.bar(x + offset, [v * 100 for v in f1], w,
                      label=name, color=color + 'CC',
                      edgecolor=color, linewidth=1.5, zorder=3)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f'{h:.1f}%', ha='center', va='bottom',
                    color=TEXT, fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 118)
    ax.set_title('F1-Score Per Kelas: Perbandingan Model', pad=14)
    ax.set_ylabel('F1-Score (%)')
    ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, framealpha=1)
    _apply_theme(fig, ax)
    plt.tight_layout(pad=2.5)
    return fig


# ── Feature Importance (DT) ───────────────────────────────────────────────
def plot_feature_importance(dt_model, feature_names, top_n=15):
    importances = dt_model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    top_feats   = [feature_names[i] for i in indices]
    top_vals    = importances[indices]

    def fmt(name):
        name = name.replace('Area_', '[Area] ').replace('Item_', '[Item] ')
        return name.replace('_', ' ').title()

    top_feats_fmt = [fmt(f) for f in top_feats]
    norm   = plt.Normalize(top_vals.min(), top_vals.max())
    colors = plt.cm.cool(norm(top_vals))

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(range(top_n), top_vals[::-1],
                   color=colors[::-1], edgecolor=BORDER, linewidth=0.8, zorder=3)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_feats_fmt[::-1], fontsize=9, color=TEXT)
    ax.set_xlabel('Importance Score')
    ax.set_title('Top 15 Feature Importance (Decision Tree)', pad=14)
    for bar, val in zip(bars, top_vals[::-1]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', color=MUTED, fontsize=8)
    _apply_theme(fig, ax)
    ax.invert_yaxis()
    plt.tight_layout(pad=2.5)
    return fig


# ── Cross Validation ──────────────────────────────────────────────────────
def plot_cross_validation(cv_scores, mean_score):
    folds = [f'Fold {i+1}' for i in range(len(cv_scores))]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(folds, cv_scores * 100, mean_score * 100,
                    alpha=0.12, color=INDIGO, zorder=2)
    ax.plot(folds, cv_scores * 100, 'o-', color=INDIGO, linewidth=2.5, markersize=10,
            zorder=4, markerfacecolor=INDIGO, markeredgecolor='white', markeredgewidth=1.5)
    ax.axhline(mean_score * 100, color=ROSE, linestyle='--', linewidth=1.8,
               label=f'Mean CV: {mean_score*100:.2f}%', zorder=3)
    for fold, score in zip(folds, cv_scores):
        ax.annotate(f'{score*100:.2f}%', (fold, score * 100),
                    textcoords='offset points', xytext=(0, 13),
                    ha='center', color=TEXT, fontsize=9, fontweight='bold')
    ax.set_title('Hasil 5-Fold Cross Validation — Decision Tree', pad=14)
    ax.set_ylabel('Akurasi (%)')
    ax.set_ylim(min(cv_scores * 100) - 1.5, max(cv_scores * 100) + 2.5)
    ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, framealpha=1)
    _apply_theme(fig, ax)
    plt.tight_layout(pad=2.5)
    return fig


# ── Naive Bayes Priors ────────────────────────────────────────────────────
def plot_nb_priors(nb_model, class_labels):
    priors = nb_model.class_prior_
    colors = [CLASS_COLORS.get(l, INDIGO) for l in class_labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(class_labels, priors,
                  color=[c + 'BB' for c in colors],
                  edgecolor=colors, linewidth=2, width=0.5, zorder=3)
    for bar, val in zip(bars, priors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.3f}', ha='center', va='bottom',
                color=TEXT, fontsize=10, fontweight='bold')
    ax.set_title('Naive Bayes — Class Priors', pad=14)
    ax.set_xlabel('Kelas')
    ax.set_ylabel('Probability')
    ax.set_ylim(0, max(priors) * 1.25)
    _apply_theme(fig, ax)
    plt.tight_layout(pad=2.5)
    return fig
