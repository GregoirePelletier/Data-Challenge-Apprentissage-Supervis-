# Classif 3 : Groupe (Analyse Factorielle Multiple)

import prince
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd, numpy as np
import os

#______________________________________________________________________________
#______________________________________________________________________________
# Analyse Factorielle Multiple

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# TEMPORAIRE : tolère le doublon d’OpenMP (pas idéal mais pratique)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from sklearn.cluster import KMeans

# Données finales (avec quanti + quali non encodées)
cols_quanti = [
    'lead_time_log','stays_in_weekend_nights','stays_in_week_nights','adults','children','babies',
    'previous_cancellations','previous_bookings_not_canceled','booking_changes','days_in_waiting_list',
    'adr_per_person','required_car_parking_spaces','total_of_special_requests',
    'arrival_date_day_of_month','arrival_date_year','month_sin','month_cos','week_sin','week_cos',
    'party_size','prev_cancel_ratio'
]
cols_quali = [
    'is_repeated_guest','hotel','meal','country','market_segment',
    'distribution_channel','room_changed','deposit_type','customer_type'
]

X = df[cols_quanti + cols_quali].dropna().copy()
for c in cols_quanti:
    X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
for c in cols_quali:
    X[c] = X[c].astype("category")

# 2) forcer TOUTES les quantitatives en float (pas int)
#X[cols_quanti] = X[cols_quanti].apply(pd.to_numeric, errors="coerce").astype(float)

# 3) forcer les qualitatives en category
X[cols_quali] = X[cols_quali].astype("category")

# 4) fit FAMD frais
famd = prince.FAMD(n_components=6, random_state=55).fit(X)
coord = famd.row_coordinates(X)

print("cat_cols_:", famd.cat_cols_)
print("num_cols_:", famd.num_cols_)
assert "adults" not in famd.cat_cols_
assert "babies" not in famd.cat_cols_
assert "booking_changes" not in famd.cat_cols_
assert "arrival_date_year" not in famd.cat_cols_

scaler_famd = StandardScaler().fit(coord)
Z = scaler_famd.transform(coord)
Z_df = pd.DataFrame(Z, index=coord.index)

km_route = KMeans(n_clusters=3, n_init=30, random_state=55).fit(Z)

joblib.dump(famd,        "famd.pkl")
joblib.dump(scaler_famd, "scaler_famd.pkl")
joblib.dump(km_route,    "km_route.pkl")
joblib.dump(X_train.columns.tolist(), "ohe_columns.pkl")  # pour réaligner le test

df.loc[Z_df.index, "cluster_famd"] = km_route.labels_.astype(int)

print(pd.Series(km_route).value_counts().sort_index())
print(pd.crosstab(df["cluster_famd"], df["country"].apply(lambda c: "Portugal" if c=="PRT" else "Etranger")))
print(pd.crosstab(df["cluster_famd"], df["market_segment"]))

#______________________________________________________________________________
#______________________________________________________________________________
# Analyse de la classification

import matplotlib.pyplot as plt

# 1️⃣ Tableaux de proportions par cluster
tab_country = (
    pd.crosstab(df["cluster_famd"], df["country"].eq("PRT").map({True: "Portugal", False: "Etranger"}))
    .apply(lambda r: r / r.sum(), axis=1)
    .round(2)
)
tab_segment = (
    pd.crosstab(df["cluster_famd"], df["market_segment"].eq("Online TA").map({True: "Online", False: "Offline"}))
    .apply(lambda r: r / r.sum(), axis=1)
    .round(2)
)

print("\nRépartition Portugal / Etranger (%) :")
print(tab_country * 100)
print("\nRépartition Online / Offline (%) :")
print(tab_segment * 100)

# 2️⃣ Visualisation FAMD (axes 1–2)
coord = famd.row_coordinates(X)
plt.figure(figsize=(7, 6))
plt.scatter(coord.iloc[:, 0], coord.iloc[:, 1], c=df.loc[X.index, "cluster_famd"],
            cmap="Set2", s=5, alpha=0.6)
plt.xlabel("Axe 1")
plt.ylabel("Axe 2")
plt.title("FAMD — projection individus (colorés par cluster)")
plt.grid(alpha=0.2)
plt.show()

# Analyse des groupes

quanti = ["adr_per_person","lead_time_log","stays_in_weekend_nights","stays_in_week_nights",
          "party_size","previous_cancellations","previous_bookings_not_canceled",
          "prev_cancel_ratio","booking_changes","days_in_waiting_list",
          "total_of_special_requests","children","adults","babies"]

summary_mean = df.groupby("cluster_famd")[quanti].mean().round(2)
summary_med  = df.groupby("cluster_famd")[quanti].median().round(2)

tab_roomchg = df.groupby("cluster_famd")["room_changed"].mean().round(3)

tab_country = (
    pd.crosstab(df["cluster_famd"], df["country"].eq("PRT").map({True:"Portugal", False:"Etranger"}))
    .apply(lambda r: r / r.sum(), axis=1).round(3)
)

tab_segment = (
    pd.crosstab(df["cluster_famd"], df["market_segment"].eq("Online TA").map({True:"Online", False:"Offline"}))
    .apply(lambda r: r / r.sum(), axis=1).round(3)
)

# répartition par type de chambre (top 5)
ct_room = (pd.crosstab(df["cluster_famd"], df["reserved_room_type"], normalize="index").round(3))

# correction: diff en Series avec index explicite
diff = pd.Series(
    (summary_mean.loc[0.0] - summary_mean.loc[1.0]).abs(),
    index=summary_mean.columns
).sort_values(ascending=False)

# --- PDF ---
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

pdf_path = chemin_sortie / "profil_clusters_FAMD.pdf"
styles = getSampleStyleSheet()
doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
story = []

story.append(Paragraph("<b>Profil des clusters FAMD</b>", styles["Title"]))
story.append(Spacer(1, 8))

# Résumé % Portugal/Étranger + Online/Offline
story.append(Paragraph("<b>Résumé par cluster (%): Portugal/Étranger</b>", styles["Heading2"]))
t = Table([["Cluster","Portugal","Étranger"]] +
          [[str(i), f"{100*tab_country.loc[i,'Portugal']:.1f}%", f"{100*tab_country.loc[i,'Etranger']:.1f}%"]
           for i in tab_country.index])
t.setStyle([("GRID",(0,0),(-1,-1),0.5,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.lightgrey)])
story.append(t); story.append(Spacer(1,6))

story.append(Paragraph("<b>Résumé par cluster (%): Online/Offline</b>", styles["Heading2"]))
t = Table([["Cluster","Online","Offline"]] +
          [[str(i), f"{100*tab_segment.loc[i,'Online']:.1f}%", f"{100*tab_segment.loc[i,'Offline']:.1f}%"]
           for i in tab_segment.index])
t.setStyle([("GRID",(0,0),(-1,-1),0.5,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.lightgrey)])
story.append(t); story.append(Spacer(1,10))

# Moyennes & Médianes
for title, data in [("Moyennes (quanti clés)", summary_mean), ("Médianes (quanti clés)", summary_med)]:
    story.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))
    cols = ["Var"] + [str(i) for i in data.index]
    rows = [cols] + [[c] + [f"{v:.2f}" for v in data.loc[:, c]] for c in data.columns]
    t = Table(rows)
    t.setStyle([("GRID",(0,0),(-1,-1),0.5,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.lightgrey)])
    story.append(t); story.append(Spacer(1,8))

# room_changed
story.append(Paragraph("<b>Taux de room_changed</b>", styles["Heading2"]))
t = Table([["Cluster","room_changed"]] + [[str(i), f"{tab_roomchg.loc[i]:.3f}"] for i in tab_roomchg.index])
t.setStyle([("GRID",(0,0),(-1,-1),0.5,colors.grey)])
story.append(t); story.append(Spacer(1,8))

for i in [c for c in [0.0, 1.0] if c in ct_room.index]:
    story.append(Paragraph(f"<b>Top 5 reserved_room_type — cluster {i}</b>", styles["Heading3"]))
    top5 = ct_room.loc[i].sort_values(ascending=False).head(5)
    t = Table([[k, f"{100*v:.1f}%"] for k, v in top5.items()])
    t.setStyle([("GRID", (0,0), (-1,-1), 0.5, colors.grey)])
    story.append(t)
    story.append(Spacer(1,6))

# Top écarts
story.append(Paragraph("<b>Top écarts absolus (moyennes) entre clusters 0 et 1</b>", styles["Heading2"]))
t = Table([["Variable","Écart"]] + [[var, f"{val:.2f}"] for var,val in diff.head(10).items()])
t.setStyle([("GRID",(0,0),(-1,-1),0.5,colors.grey),("BACKGROUND",(0,0),(-1,0),colors.lightgrey)])
story.append(t)

doc.build(story)
print(f"✅ PDF créé : {pdf_path}")







'''
#______________________________________________________________________________
#______________________________________________________________________________
# t-SNE


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ------------------------------
# 1️⃣ Préparation des données
# ------------------------------

# Suppose ton DataFrame principal s'appelle df
# et ta variable cible est 'reservation_status'
# (avec valeurs 0=Check-out, 1=Cancel, 2=No-Show)

# Sélection des classes 1 et 2
mask_12 = df_c["reservation_status"].isin([1, 2])
X_cluster_12 = df.loc[mask_12, liste_var_quanti].copy()
y_cluster_12 = df.loc[mask_12, "reservation_status"]

# Prédictions correspondantes (sur ces mêmes individus)
y_pred_all = random_forest_model.predict(df_c[X_train.columns])
y_pred_cluster_12 = y_pred_all[mask_12]

# Standardisation (t-SNE est sensible aux échelles)
scaler = StandardScaler()
X_scaled_12 = scaler.fit_transform(X_cluster_12)

# ------------------------------
# 2️⃣ Calcul du t-SNE
# ------------------------------
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    max_iter=1500,
    random_state=42
)
emb = tsne.fit_transform(X_scaled_12)

# ------------------------------
# 3️⃣ Visualisation
# ------------------------------
plt.figure(figsize=(9,7))

# 🔵 Vrais Cancel (classe 1)
mask1 = (y_cluster_12 == 1)
plt.scatter(
    emb[mask1, 0], emb[mask1, 1],
    color="lightsteelblue", s=10, alpha=0.3,
    label="Cancel (1)"
)

# 🔴 Vrais No-Show (classe 2)
mask2 = (y_cluster_12 == 2)
plt.scatter(
    emb[mask2, 0], emb[mask2, 1],
    color="red", s=35, alpha=0.9,
    edgecolors="black", linewidths=0.4,
    label="Vrai No-Show (2)"
)

# 🟡 Prédits No-Show (tous ceux que le modèle classe 2)
mask_pred2 = (y_pred_cluster_12 == 2)
plt.scatter(
    emb[mask_pred2, 0], emb[mask_pred2, 1],
    color="gold", marker="*", s=90, alpha=1,
    edgecolors="black", linewidths=0.8,
    label="Prédit No-Show (classe 2)"
)

plt.title("t-SNE — Classes réelles et prédictions No-Show", fontsize=13)
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
'''



'''
#______________________________________________________________________________
#______________________________________________________________________________

# UMAP
import umap
um = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="euclidean", random_state=42)
emb = um.fit_transform(X_cluster_12)
plt.figure(figsize=(6,5))
plt.scatter(emb[:,0], emb[:,1], c=y_cluster_12.map({1:"royalblue",2:"tomato"}), s=8, alpha=0.5)
plt.title("UMAP"); plt.tight_layout(); plt.show()
'''
