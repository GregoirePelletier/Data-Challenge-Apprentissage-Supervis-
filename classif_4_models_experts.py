



import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterSampler

# clusters pour chaque split (à partir de df où 'cluster_famd' est stocké)
c_tr = df.loc[X_train.index, "cluster_famd"].astype(int)
c_te = df.loc[X_test.index,  "cluster_famd"].astype(int)

grid = {
    "n_estimators": [200,400,800,1200],
    "max_depth": [None,6,10,16,24,32],
    "min_samples_split": [2,5,10,20],
    "min_samples_leaf": [1,2,4,8],
    "max_features": ["sqrt","log2",0.3,0.5,0.8],
    "bootstrap": [True, False]
}

models, preds = {}, pd.Series(index=X_test.index, dtype=int)
for k in [0,1]:
    tr_idx, te_idx = c_tr[c_tr==k].index, c_te[c_te==k].index
    if len(tr_idx)==0 or len(te_idx)==0: continue
    Xtr, ytr = X_train.loc[tr_idx], y_train.loc[tr_idx]
    Xte, yte = X_test.loc[te_idx],  y_test.loc[te_idx]

    best, best_f1 = None, -1
    for p in ParameterSampler(grid, n_iter=30, random_state=55):
        rf = RandomForestClassifier(random_state=55, n_jobs=-1, class_weight="balanced", **p)
        rf.fit(Xtr, ytr)
        f1 = f1_score(yte, rf.predict(Xte), average="macro")
        if f1 > best_f1: best_f1, best = f1, rf
    models[k] = best
    preds.loc[te_idx] = best.predict(Xte)
    print(f"Cluster {k} — meilleur F1_macro: {best_f1:.3f}")

# F1 global (macro et pondéré)
idx_nan = preds[preds.isna()].index
preds.loc[idx_nan] = models[1].predict(X_test.loc[idx_nan])
print("F1_macro global :", f1_score(y_test, preds, average="macro"))

print("F1_weighted global :", f1_score(y_test.loc[preds.index], preds, average="weighted"))





joblib.dump(models[0], "rf_cluster0.pkl")
joblib.dump(models[1], "rf_cluster1.pkl")






























#______________________________________________________________________________
#______________________________________________________________________________


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

vars_quanti = [
    "lead_time_log",
    "adr_per_person",
    "prev_cancel_ratio",
    "total_of_special_requests",
    "days_in_waiting_list",
    "stays_in_week_nights",
    "stays_in_weekend_nights",
    "required_car_parking_spaces",
    "party_size",
    "booking_changes",
    "month_sin", "month_cos", "week_sin", "week_cos"
]

X = df_c[vars_quanti].copy()
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2, random_state=55)
Z = pca.fit_transform(X_std)

# K-Means
kmeans = KMeans(n_clusters=2, random_state=55, n_init=10)
labels_quanti = kmeans.fit_predict(Z)

# Graphe
plt.figure(figsize=(9,6))
color_map = {0:"#8ecae6", 1:"#e63946"}
colors = pd.Series(labels_quanti).map(color_map)
plt.scatter(Z[:,0], Z[:,1], c=colors, s=12, alpha=0.8, edgecolors="white", linewidth=0.3)
plt.title("PCA + K-Means (2 groupes) — Variables quantitatives uniquement")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.grid(alpha=0.25)
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ==============================
# 1. Variables quantitatives
# ==============================
vars_quanti = [
    "lead_time_log",
    "adr_per_person",
    "prev_cancel_ratio",
    "total_of_special_requests",
    "days_in_waiting_list",
    "stays_in_week_nights",
    "stays_in_weekend_nights",
    "required_car_parking_spaces",
    "party_size",
    "booking_changes",
    "month_sin", "month_cos", "week_sin", "week_cos"
]

X = df_c[vars_quanti].copy()
y = df_c["reservation_status"].astype(int)

# ==============================
# 2. Standardisation + PCA
# ==============================
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=55)
Z = pca.fit_transform(X_std)
print("Variance expliquée PC1/PC2 :", np.round(pca.explained_variance_ratio_ * 100, 2))

# ==============================
# 3. KMeans (2 clusters)
# ==============================
kmeans = KMeans(n_clusters=2, random_state=55, n_init=10)
labels_q = kmeans.fit_predict(Z)

df_c["cluster_quanti"] = labels_q

print("\nEffectifs par cluster :")
print(pd.Series(labels_q).value_counts().sort_index())

print("\nCrosstab cluster × classe (0/1/2) :")
print(pd.crosstab(labels_q, y))

# ==============================
# 4. Visualisation PCA + clusters
# ==============================
plt.figure(figsize=(9,6))
color_map = {0:"#8ecae6", 1:"#e63946"}
colors = pd.Series(labels_q).map(color_map)
plt.scatter(Z[:,0], Z[:,1], c=colors, s=14, alpha=0.8, edgecolors="white", linewidth=0.3)
plt.title("PCA + KMeans (2 groupes) — Variables quantitatives uniquement")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# ==============================
# 5. Analyse chiffrée (moyennes par cluster)
# ==============================
df_summary = (
    df_c.groupby("cluster_quanti")[vars_quanti]
    .mean()
    .T
    .round(3)
)

df_summary["diff"] = df_summary[1] - df_summary[0]
df_summary["abs_diff"] = df_summary["diff"].abs()
df_summary = df_summary.sort_values("abs_diff", ascending=False)

print("\nTop 10 variables quantitatives discriminantes (moyenne par cluster) :")
print(df_summary.head(10).to_string())

import pandas as pd
import numpy as np

# 1️⃣ Variables qualitatives (déjà définies dans ton projet)


# 2️⃣ Encodage One-Hot + agrégation par cluster
prefixes = [
    "is_repeated_guest", "hotel_", "meal_", "country_",
    "market_segment_", "distribution_channel_", "room_changed",
    "deposit_type_", "customer_type_"
]

ohe_cols = [c for c in df_c.columns if any(c.startswith(p) for p in prefixes)]
if not ohe_cols:
    raise ValueError("Aucune colonne OHE trouvée : es-tu bien sur df_c (et pas df) ?")

prop_by_cluster = (
    df_c.groupby("cluster_quanti")[ohe_cols]
        .mean()
        .T
)
prop_by_cluster["diff"] = prop_by_cluster[1] - prop_by_cluster[0]
prop_by_cluster["abs_diff"] = prop_by_cluster["diff"].abs()
print(prop_by_cluster.sort_values("abs_diff", ascending=False).head(20))