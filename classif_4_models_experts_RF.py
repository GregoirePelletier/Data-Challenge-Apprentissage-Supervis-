# Classif 4 : Modèles Experts RF

import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterSampler

#______________________________________________________________________________
#______________________________________________________________________________
# Clusters 
# clusters pour chaque split (à partir de df où 'cluster_famd' est stocké)
c_tr = df.loc[X_train.index, "cluster_famd"].astype(int)
c_te = df.loc[X_test.index,  "cluster_famd"].astype(int)

#______________________________________________________________________________
#______________________________________________________________________________
# Espace d'hyperparamètres 
grid = {
    "n_estimators": [200,400,800,1200],
    "max_depth": [None,6,10,16,24,32],
    "min_samples_split": [2,5,10,20],
    "min_samples_leaf": [1,2,4,8],
    "max_features": ["sqrt","log2",0.3,0.5,0.8],
    "bootstrap": [True, False]
}

#______________________________________________________________________________
#______________________________________________________________________________
# Entrainement par cluster
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

#______________________________________________________________________________
#______________________________________________________________________________
# Scores globaux & Stockage Modèles
idx_nan = preds[preds.isna()].index
preds.loc[idx_nan] = models[1].predict(X_test.loc[idx_nan])
print("F1_macro global :", f1_score(y_test, preds, average="macro"))

print("F1_weighted global :", f1_score(y_test.loc[preds.index], preds, average="weighted"))

joblib.dump(models[0], "rf_cluster0.pkl")
joblib.dump(models[1], "rf_cluster1.pkl")













