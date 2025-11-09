# ============================================
#  Booking Cancel — Inference & Submission
#  FE -> OHE -> FAMD+Scaler+KMeans (2->0) -> RF experts -> CSV
# ============================================
# ============================================
#  Booking Cancel — Inference & Submission
#  FE -> OHE -> FAMD+Scaler+KMeans (2->0) -> RF experts -> CSV
# ============================================

import numpy as np
import pandas as pd
from pathlib import Path
import joblib

# -----------------------------
# Paramètres / chemins
# -----------------------------
nom_modele = "submission_experts"

TEST_CSV = Path(r"C:\Users\saout\Documents\Data-Challenge-Apprentissage-Supervis-\data\test_data.csv")
OUT_CSV  = Path(rf"C:\Users\saout\Documents\Data-Challenge-Apprentissage-Supervis-\sorties\{nom_modele}.csv")

FAMD_PATH     = "famd.pkl"
SCALER_PATH   = "scaler_famd.pkl"
KM_PATH       = "km_route.pkl"
OHE_COLS_PATH = "ohe_columns.pkl"
RF0_PATH      = "rf_cluster0.pkl"
RF1_PATH      = "rf_cluster1.pkl"

# -----------------------------
# Chargements
# -----------------------------
famd        = joblib.load(FAMD_PATH)
scaler_famd = joblib.load(SCALER_PATH)
km_route    = joblib.load(KM_PATH)
ohe_columns = joblib.load(OHE_COLS_PATH)
rf0         = joblib.load(RF0_PATH)
rf1         = joblib.load(RF1_PATH)

# -----------------------------
# Lecture test + FE
# -----------------------------
df_test = pd.read_csv(TEST_CSV)
if "Unnamed: 0" in df_test.columns:
    df_test.drop(columns=["Unnamed: 0"], inplace=True)
if "row_id" not in df_test.columns:
    df_test["row_id"] = np.arange(len(df_test))

# FE (cohérent train)
df_test["lead_time_tronq"] = df_test["lead_time"].clip(upper=df_test["lead_time"].quantile(0.99))
df_test["lead_time_log"]   = np.log1p(df_test["lead_time"])
df_test.loc[~df_test["market_segment"].isin(["Online TA","Offline TA/TO","Groups","Direct","Corporate"]), "market_segment"] = "Other"
top_countries = ["PRT","GBR","FRA","ESP","DEU","ITA","IRL","BEL","BRA","USA","NLD","CHE","CN","AUT"]
df_test.loc[~df_test["country"].isin(top_countries), "country"] = "Other"
df_test["party_size"] = df_test["adults"] + df_test["children"].fillna(0) + df_test["babies"].fillna(0)
df_test["adr_per_person"] = (df_test["adr"] / df_test["party_size"].replace(0, np.nan)).fillna(0)
total_prev = df_test["previous_cancellations"] + df_test["previous_bookings_not_canceled"]
df_test["prev_cancel_ratio"] = np.where(total_prev > 0, df_test["previous_cancellations"] / total_prev, 0)
month_map = {"January":1,"February":2,"March":3,"April":4,"May":5,"June":6,"July":7,"August":8,"September":9,"October":10,"November":11,"December":12}
df_test["month_num"] = df_test["arrival_date_month"].map(month_map).astype(float)
df_test["month_sin"] = np.sin(2 * np.pi * df_test["month_num"] / 12)
df_test["month_cos"] = np.cos(2 * np.pi * df_test["month_num"] / 12)
w = pd.to_numeric(df_test["arrival_date_week_number"], errors="coerce").fillna(1).clip(1, 53).astype(int)
df_test["week_sin"] = np.sin(2 * np.pi * w / 52)
df_test["week_cos"] = np.cos(2 * np.pi * w / 52)
df_test["room_changed"] = (df_test["reserved_room_type"] != df_test["assigned_room_type"]).astype(int)

# -----------------------------
# OHE alignée (features pour RF)
# -----------------------------
liste_var_categ = [
    "is_repeated_guest","hotel","meal","country",
    "market_segment","distribution_channel",
    "reserved_room_type","assigned_room_type",
    "room_changed","deposit_type","customer_type"
]
liste_var_quanti = [
    "lead_time_log","stays_in_weekend_nights","stays_in_week_nights",
    "adults","children","babies","previous_cancellations",
    "previous_bookings_not_canceled","booking_changes","days_in_waiting_list",
    "adr_per_person","required_car_parking_spaces","total_of_special_requests",
    "arrival_date_day_of_month","arrival_date_year",
    "month_sin","month_cos","week_sin","week_cos",
    "party_size","prev_cancel_ratio"
]

df_test_FE = df_test[liste_var_categ + liste_var_quanti + ["row_id"]].copy()
cat_cols = [c for c in liste_var_categ if c in df_test_FE.columns]
df_test_c = pd.get_dummies(df_test_FE, columns=cat_cols, drop_first=False, dtype=int)
df_test_c = df_test_c.set_index("row_id").reindex(columns=ohe_columns, fill_value=0)

# -----------------------------
# Routage FAMD -> Scaler -> KMeans (2 -> 0)  [pré-OHE]
# (forçage du typage "d’hier")
# -----------------------------
# Catégorielles d’hier (même si ce sont des entiers)
cat_expected = [
    'adults','arrival_date_day_of_month','arrival_date_year','babies','booking_changes',
    'country','customer_type','days_in_waiting_list','deposit_type','distribution_channel',
    'hotel','is_repeated_guest','market_segment','meal','previous_bookings_not_canceled',
    'previous_cancellations','required_car_parking_spaces','room_changed',
    'stays_in_week_nights','stays_in_weekend_nights','total_of_special_requests'
]
# Numériques d’hier
num_expected = [
    'lead_time_log','children','adr_per_person','month_sin','month_cos',
    'week_sin','week_cos','party_size','prev_cancel_ratio'
]

# Préparation input FAMD (mimant l’ancienne définition)
df_famd_input = df_test[cat_expected + num_expected].copy()

# cast num -> float (et NA -> 0.0)
for c in num_expected:
    df_famd_input[c] = pd.to_numeric(df_famd_input[c], errors="coerce").fillna(0.0).astype("float64")

# clamp quali sur le vocabulaire appris par la FAMD "d’hier"
# (on suppose que famd.pkl correspond à ce typage et que l'ordre des catégories est le même)
for i, c in enumerate(famd.cat_cols_):
    allowed = pd.Index(famd.cat_scaler_.categories_[i])
    fb = "Other" if "Other" in allowed else allowed[0]
    col = df_famd_input[c].astype(object)
    col = col.where(col.notna(), fb)
    col = col.where(col.isin(allowed), fb)
    df_famd_input[c] = pd.Categorical(col, categories=allowed)

# aligne l'index sur row_id
df_famd_input.index = df_test["row_id"].values

# coordonnées + standardisation + routage
coord_test = famd.row_coordinates(df_famd_input)
Z_test = pd.DataFrame(scaler_famd.transform(coord_test), index=df_famd_input.index)
clusters = km_route.predict(Z_test).astype(int)

# règle métier (si tu la conserves) :
clusters = np.where(clusters == 2, 0, clusters)

# Series indexée (servira plus bas)
clusters_s = pd.Series(clusters, index=Z_test.index)


# -----------------------------
# Prédictions experts (ALIGNÉES PAR INDEX)
# -----------------------------
# s'assure que les index coïncident; sinon, on réindexe df_test_c
if not df_test_c.index.equals(Z_test.index):
    df_test_c = df_test_c.reindex(Z_test.index, fill_value=0)

clusters_s = pd.Series(clusters, index=Z_test.index)
pred_s = pd.Series(0, index=df_test_c.index, dtype=int)

idx_c0 = clusters_s.index[clusters_s == 0]
idx_c1 = clusters_s.index[clusters_s == 1]

if len(idx_c0):
    pred_s.loc[idx_c0] = rf0.predict(df_test_c.loc[idx_c0])
if len(idx_c1):
    pred_s.loc[idx_c1] = rf1.predict(df_test_c.loc[idx_c1])

# -----------------------------
# Export CSV + checks
# -----------------------------
submit = pd.DataFrame({"row_id": pred_s.index, "prediction": pred_s.values})
submit.to_csv(OUT_CSV, index=False)

print(f"✅ Fichier de soumission écrit : {OUT_CSV}")
print("Clusters (après 2→0) :", clusters_s.value_counts(normalize=True).round(3).to_dict())
print("Prédictions           :", submit["prediction"].value_counts(normalize=True).round(3).to_dict())

# Checks utiles
assert df_test_c.shape[1] == len(ohe_columns), "Nb colonnes OHE ≠ entraînement"
assert (df_test_c.columns == pd.Index(ohe_columns)).all(), "Ordre de colonnes OHE différent"
assert len(df_test_c) == len(Z_test) == len(clusters), "Alignement lignes OHE/FAMD/KMeans"




*
# -----------------------------
# Comparaison répartition clusters concours vs entraînement
# -----------------------------
try:
    # clusters sauvegardés pendant l'entraînement
    train_clusters = df["cluster_famd"].astype(int)
    # règle 2→0 si appliquée dans le modèle
    train_clusters_routed = np.where(train_clusters == 2, 0, train_clusters)

    pct_train = pd.Series(train_clusters_routed).value_counts(normalize=True).round(3).sort_index()
    pct_test  = pd.Series(clusters).value_counts(normalize=True).round(3).sort_index()

    print("\n=== Comparaison distribution clusters ===")
    print(pd.DataFrame({
        "train_%": pct_train,
        "concours_%": pct_test
    }).fillna(0))

except Exception as e:
    print("⚠️ Impossible de comparer les clusters (pas de df local chargé) :", e)
 





print(submit['prediction'].value_counts())

print("rf0.classes_:", getattr(rf0, "classes_", None))
print("rf1.classes_:", getattr(rf1, "classes_", None))

import numpy as np

idx = np.arange(len(df_test_c))
idx_c0 = idx[clusters == 0]
idx_c1 = idx[clusters == 1]

if len(idx_c0):
    y0 = rf0.predict(df_test_c.iloc[idx_c0, :])
    print("rf0 -> uniques & counts:", np.unique(y0, return_counts=True))
if len(idx_c1):
    y1 = rf1.predict(df_test_c.iloc[idx_c1, :])
    print("rf1 -> uniques & counts:", np.unique(y1, return_counts=True))

print("nb colonnes OHE vides (sum=0):", int((df_test_c.sum(axis=0) == 0).sum()))
print("(nb features non-nulles par ligne) ->", (df_test_c != 0).sum(axis=1).describe())

(df_test_c != 0).sum(axis=1).median()

num_cols_check = ["lead_time_log","adr_per_person","party_size","prev_cancel_ratio",
                  "stays_in_week_nights","stays_in_weekend_nights","adults","children","babies"]
print({c: (c in df_test_c.columns) for c in num_cols_check})
print(df_test_c[num_cols_check].describe().T[["mean","std","min","max"]])
