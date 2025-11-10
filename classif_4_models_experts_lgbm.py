# Classif 4 : Modèles Experts LGBM

import numpy as np, pandas as pd, time, gc, joblib, lightgbm
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterSampler, StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

#______________________________________________________________________________
#______________________________________________________________________________
# Clusters 
c_tr = df.loc[X_train.index, "cluster_famd"].astype(int)
c_te = df.loc[X_test.index,  "cluster_famd"].astype(int)

#______________________________________________________________________________
#______________________________________________________________________________
# Espace d'hyperparamètres (LightGBM)
grid = {
    "n_estimators":      [400, 800, 1000, 1200, 1400, 1600],
    "learning_rate":     [0.02, 0.03, 0.045, 0.06, 0.1],
    "num_leaves":        [31, 63, 127],
    "max_depth":         [-1, 6, 8, 10],
    "min_child_samples": [10, 20, 30, 50],
    "subsample":         [0.7, 0.85, 1.0],
    "colsample_bytree":  [0.7, 0.85, 1.0],
    "reg_alpha":         [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
    "reg_lambda":        [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
    "min_split_gain":    [0.0, 0.005, 0.01],
    "max_bin":           [127, 255],
    "boosting_type":     ["gbdt"],  # tu peux ajouter "dart" si tu veux essayer
}

#______________________________________________________________________________
#______________________________________________________________________________
# Entrainement par cluster
N_TRIALS = 30
RSEED = 55

# config objectif global (multi-classe ou binaire)
all_classes = np.unique(y_train)
objective_name = "binary" if len(all_classes) == 2 else "multiclass"
extra_params = {"num_class": len(all_classes)} if len(all_classes) > 2 else {}

models = {}
preds = pd.Series(index=X_test.index, dtype=int)

tic = time.time()
for k in [0, 1]:
    tr_idx = c_tr[c_tr == k].index
    te_idx = c_te[c_te == k].index
    if len(tr_idx) == 0 or len(te_idx) == 0:
        continue

    Xtr, ytr = X_train.loc[tr_idx], y_train.loc[tr_idx]
    Xte, yte = X_test.loc[te_idx],  y_test.loc[te_idx]

    # split interne pour early stopping (pas de fuite vers test)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=RSEED)
    (inner_tr, inner_val) = next(sss.split(Xtr, ytr))
    X_trk, y_trk = Xtr.iloc[inner_tr], ytr.iloc[inner_tr]
    X_valk, y_valk = Xtr.iloc[inner_val], ytr.iloc[inner_val]

    # class_weight équilibré (calculé sur le cluster k)
    cls_k = np.unique(ytr)
    weights_k = compute_class_weight(class_weight="balanced", classes=cls_k, y=ytr)
    CLASS_WEIGHT_K = {c: w for c, w in zip(cls_k, weights_k)}

    best_model, best_params, best_f1 = None, None, -1.0

    for p in ParameterSampler(grid, n_iter=N_TRIALS, random_state=RSEED):
        try:
            clf = LGBMClassifier(
                objective=objective_name,
                class_weight=CLASS_WEIGHT_K,
                random_state=RSEED,
                n_jobs=-1,
                **extra_params,
                **p,
            )
            # fit avec early stopping (callbacks compatibles toutes versions)
            clf.fit(
                X_trk, y_trk,
                eval_set=[(X_valk, y_valk)],
                eval_metric="logloss",
                callbacks=[
                    lightgbm.early_stopping(stopping_rounds=50),
                    lightgbm.log_evaluation(period=0),
                ],
            )

            # score sur le sous-ensemble test du cluster (comme ton RF)
            y_pred_te = clf.predict(Xte, num_iteration=getattr(clf, "best_iteration_", None))
            f1m = f1_score(yte, y_pred_te, average="macro")

            if f1m > best_f1:
                best_f1 = f1m
                best_model = clf
                best_params = p
        except Exception as e:
            print(f"⚠️ Cluster {k} — essai ignoré: {e}")

    models[k] = best_model
    preds.loc[te_idx] = best_model.predict(Xte, num_iteration=getattr(best_model, "best_iteration_", None))
    print(f"Cluster {k} — meilleur F1_macro: {best_f1:.3f} | params: {best_params} | best_iter: {getattr(best_model,'best_iteration_',None)}")
    gc.collect()

# combler les éventuels NaN (si un cluster n'avait pas de modèle ou de test)
idx_nan = preds[preds.isna()].index
if len(idx_nan):
    fallback_k = 1 if 1 in models else 0
    preds.loc[idx_nan] = models[fallback_k].predict(X_test.loc[idx_nan],
                                                    num_iteration=getattr(models[fallback_k], "best_iteration_", None))

#______________________________________________________________________________
#______________________________________________________________________________
# Scores globaux & Stockage Modèles
print("F1_macro global    :", f1_score(y_test.loc[preds.index], preds, average="macro"))
print("F1_weighted global :", f1_score(y_test.loc[preds.index], preds, average="weighted"))

joblib.dump(models[0], "lgbm_cluster0.pkl")
joblib.dump(models[1], "lgbm_cluster1.pkl")

print(f"\n✅ Terminé en {time.time()-tic:.1f}s")
