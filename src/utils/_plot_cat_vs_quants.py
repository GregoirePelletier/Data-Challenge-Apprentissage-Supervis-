import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def plot_cat_vs_quants(df, cat, quants=None, pdf_path=None,
                               order=None, palette="tab10", inner="quartile",
                               show_kde=False):
    """
    Pour chaque variable quantitative q :
      - Violin plot de q par modalité de `cat` (sans points).
      - Optionnel: panneau KDE par catégorie si show_kde=True.
    """
    if quants is None:
        quants = df.select_dtypes(include="number").columns.tolist()
    quants = [q for q in quants if q != cat]

    dfv = df.copy()
    dfv[cat] = dfv[cat].astype(str)
    if order is None:
        order = list(dfv[cat].dropna().unique())

    pdf = PdfPages(pdf_path) if pdf_path else None

    for q in quants:
        print("*** "+q+" ***")
        d = dfv[[cat, q]].dropna()
        if d.empty:
            continue

        if show_kde:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            ax0, ax1 = axes
        else:
            fig, ax0 = plt.subplots(figsize=(8, 4))
            ax1 = None

        # Violin seul (pas de points)
        sns.violinplot(
            data=d, x=cat, y=q,
            order=order, palette=palette,
            inner=inner,   # 'quartile', 'box', 'point' ou None
            cut=0, scale="width",
            ax=ax0
        )
        ax0.set_title(f"{q} par {cat} (violins)")
        ax0.set_xlabel(cat); ax0.set_ylabel(q)
        ax0.set_xticklabels(ax0.get_xticklabels(), rotation=30, ha="right")

        # KDE en option
        if show_kde:
            try:
                sns.kdeplot(data=d, x=q, hue=cat, common_norm=False, fill=True, alpha=.3,
                            palette=palette, ax=ax1)
                ax1.set_title(f"Densités de {q} par {cat}")
                ax1.set_xlabel(q); ax1.set_ylabel("Densité")
            except Exception as e:
                ax1.text(0.5, 0.5, f"KDE indisponible : {e}", ha="center", va="center",
                         transform=ax1.transAxes)

        fig.tight_layout()
        if pdf:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    if pdf:
        pdf.close()
        print(f"PDF écrit : {pdf_path}")
