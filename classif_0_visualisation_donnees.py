
# Classif 0 : Visualisation des données

#______________________________________________________________________________
#______________________________________________________________________________
#### VARIABLES CATÉGORIELLES ###

pdf_path = chemin_sortie+"\\classif_0_histplot_target.pdf"
with PdfPages(pdf_path) as pdf:
    # (facultatif) métadonnées
    info = pdf.infodict()
    info["Title"] = "Histplots vs cible"
    info["Author"] = "Ton script"
    info["Subject"] = f"Proportions de {target} par catégories"
    info["CreationDate"] = datetime.now()

    for x in liste_var_categ:
        d = df[[x, target]].dropna().copy()
        if d.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(
            data=d,
            x=x,
            hue=target,
            multiple="fill",
            stat="proportion",
            discrete=True,
            shrink=0.9,
            ax=ax
        )
        ax.set_ylabel("Proportion")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
        ax.set_title(f"{x} vs {target} — proportions (empilées)")
        fig.tight_layout()

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

print("PDF écrit :", pdf_path)

# zoom deposit_type
df['deposit_type'].unique()
eph = df.loc[ df['deposit_type']=='Non Refund',['deposit_type','reservation_status','customer_type','market_segment']]
# --> Non Refund sont presque toutes annulées

#______________________________________________________________________________
#______________________________________________________________________________
### VARIABLES QUANTITATIVES ###

plot_cat_vs_quants(df, target, quants=liste_var_quanti, pdf_path=chemin_sortie+"\\classif_0_cat_vs_quants.pdf",
                               order=None, palette="tab10", inner="quartile",
                               show_kde=True)



