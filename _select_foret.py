
# === Random Forest: grid "maison" + rapport PDF (Accuracy & F1 macro) ===
import itertools
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

X_train_0 = X_train_tl
X_test_0 = X_test
y_train_0 = y_train_tl
y_test_0 = y_test



# -------------------------
# 1) Grille d'hyperparamètres
# -------------------------
param_grid = {
    "n_estimators": [ 200 ],
    "max_depth":        [64],
    "max_features":     ["sqrt"],
    "min_samples_split" :[2],
    "min_samples_leaf":[ 1]
    # "class_weight":   [None, "balanced"]  # décommente si utile
}

RANDOM_STATE = 42
OUT_DIR = "sorties"
os.makedirs(OUT_DIR, exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
pdf_path = os.path.join(OUT_DIR, f"rf_grid_report_F1_{stamp}.pdf")
csv_path = os.path.join(OUT_DIR, f"rf_grid_results_F1_{stamp}.csv")

# -------------------------------------------------
# 2) Grid search exhaustif + collecte des métriques
# -------------------------------------------------
rows = []
keys, values = zip(*param_grid.items())
for combo in itertools.product(*values):
    params = dict(zip(keys, combo))
    clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **params)
    clf.fit(X_train_0, y_train_0)

    y_pred_tr = clf.predict(X_train_0)
    y_pred_te = clf.predict(X_test_0)

    acc_tr = accuracy_score(y_train_0, y_pred_tr)
    acc_te = accuracy_score(y_test_0,  y_pred_te)

    f1_tr  = f1_score(y_train_0, y_pred_tr, average="macro")
    f1_te  = f1_score(y_test_0,  y_pred_te, average="macro")

    row = {
        **params,
        "acc_train": acc_tr,
        "acc_test":  acc_te,
        "f1_train":  f1_tr,
        "f1_test":   f1_te,
        "n_features_": getattr(clf, "n_features_in_", np.nan)
    }
    rows.append(row)

results = pd.DataFrame(rows)

# Classement: on privilégie F1_test puis accuracy_test_0
results = results.sort_values(by=["f1_test", "acc_test"], ascending=False).reset_index(drop=True)
best = results.iloc[0].to_dict()

# Sauvegarde des résultats
results.to_csv(csv_path, index=False)
print(f"[OK] Résultats complets -> {csv_path}")

# ------------------------------------------
# 3) Rapport PDF (multi-pages) avec PdfPages
# ------------------------------------------
with PdfPages(pdf_path) as pdf:
    # Page 1 — résumé & meilleurs params
    fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
    fig.suptitle("Random Forest — Grid Search (Accuracy & F1 macro)", fontsize=16)

    txt = []
    txt.append(f"Généré: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    txt.append(f"Nb configs testées: {len(results)}")
    txt.append(f"Train size: {len(y_train_0)}  |  Test size: {len(y_test)}")
    txt.append("")
    txt.append("=== Meilleure configuration (tri: F1_test puis ACC_test) ===")
    for k in param_grid.keys():
        txt.append(f"  - {k}: {best[k]}")
    txt.append("")
    txt.append("=== Scores (best) ===")
    txt.append(f"  - acc_train: {best['acc_train']:.4f}")
    txt.append(f"  - acc_test : {best['acc_test']:.4f}")
    txt.append(f"  - F1_train : {best['f1_train']:.4f} (macro)")
    txt.append(f"  - F1_test  : {best['f1_test']:.4f} (macro)")

    fig.text(0.08, 0.92, "\n".join(txt), va="top", family="monospace")
    pdf.savefig(fig); plt.close(fig)

    # Page 2 — Top 20 en table
    topk = results.head(20).copy()
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 paysage
    plt.axis('off')
    plt.title("Top 20 configurations (triées par F1_test, ACC_test)", y=1.03)
    disp = topk.copy()
    for c in ["acc_train","acc_test","f1_train","f1_test"]:
        disp[c] = disp[c].map(lambda v: f"{v:.4f}")
    table = plt.table(cellText=disp.values,
                      colLabels=disp.columns,
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    pdf.savefig(fig); plt.close(fig)

    # Page 3 — F1_test moyen vs n_estimators (si présent)
    if "n_estimators" in results.columns:
        fig = plt.figure(figsize=(8, 5))
        mean_by_ne = (results.groupby("n_estimators")["f1_test"]
                             .mean()
                             .reset_index())
        plt.plot(mean_by_ne["n_estimators"], mean_by_ne["f1_test"], marker='o')
        plt.xlabel("n_estimators")
        plt.ylabel("F1 macro (test)")
        plt.title("F1_test moyen par n_estimators")
        plt.grid(True, linestyle="--", linewidth=0.5)
        pdf.savefig(fig); plt.close(fig)

    # Page 4 — table croisée F1_test moyen (max_depth × n_estimators) si applicable
    if set(["n_estimators","max_depth"]).issubset(results.columns):
        pivot = (results.groupby(["max_depth","n_estimators"])["f1_test"]
                        .mean().unstack("n_estimators"))
        fig = plt.figure(figsize=(8, 5))
        plt.axis('off')
        plt.title("F1_test moyen par (max_depth × n_estimators)")
        disp = pivot.round(4).reset_index()
        table = plt.table(cellText=disp.values,
                          colLabels=disp.columns,
                          loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
        pdf.savefig(fig); plt.close(fig)

print(f"[OK] Rapport PDF -> {pdf_path}")

# Bonus console: top 5
print("\nTop 5 configurations:")
print(results.head(5))
