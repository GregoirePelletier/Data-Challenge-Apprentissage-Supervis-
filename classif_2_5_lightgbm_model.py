# Classif 2.5 : LGBM

# pip install optuna
import numpy as np, pandas as pd, time, joblib, optuna, lightgbm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
import numpy as np
from sklearn.metrics import f1_score

# ---- split interne pour ES (pas de fuite test) ----
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
(tr_idx, va_idx) = next(sss.split(X_train, y_train))
X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
X_val, y_val = X_train.iloc[va_idx], y_train.iloc[va_idx]

# ---- class_weight équilibré ----
classes = np.unique(y_train)
num_classes = len(classes)
is_multiclass = num_classes > 2
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
CLASS_WEIGHT = {c: w for c, w in zip(classes, weights)}
objective_name = "multiclass" if is_multiclass else "binary"
extra_params = {"num_class": num_classes} if is_multiclass else {}

# ---- métrique custom F1_weighted (pour ES) ----
def lgbm_f1_weighted(y_pred, dataset):
    y_true = dataset.get_label().astype(int)
    if is_multiclass:
        y_pred = y_pred.reshape(-1, num_classes)  # (n_samples, n_classes)
        y_hat = np.argmax(y_pred, axis=1)
    else:
        y_hat = (y_pred > 0.5).astype(int)
    f1 = f1_score(y_true, y_hat, average="weighted")
    return ("f1_weighted", f1, True)  # higher is better

#______________________________________________________________________________
#______________________________________________________________________________
# 🔧 Fonction custom F1_weighted compatible avec LGBMClassifier (sklearn)
num_classes = len(np.unique(y_train))
is_multiclass = num_classes > 2

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def lgbm_f1_weighted(labels, preds):
    """
    Version compatible sklearn API (labels, preds)
    """
    y_true = labels.astype(int)

    if is_multiclass:
        if preds.ndim == 1:
            preds = preds.reshape(num_classes, -1).T
        else:
            preds = preds.reshape(-1, num_classes)
        y_hat = np.argmax(preds, axis=1)
    else:
        y_hat = (preds > 0).astype(int)
        # ou (_sigmoid(preds) > 0.5).astype(int) si tu veux un vrai proba seuil 0.5

    f1 = f1_score(y_true, y_hat, average="weighted")
    return ("f1_weighted", f1, True)

#______________________________________________________________________________
#______________________________________________________________________________
# ---- objectif Optuna ----
def objective(trial: optuna.trial.Trial) -> float:
    subsample = trial.suggest_categorical("subsample", [0.7, 0.85, 1.0])
    params = {
        "boosting_type":     trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
        "n_estimators":      trial.suggest_int("n_estimators", 800, 2000, step=200),
        "learning_rate":     trial.suggest_float("learning_rate", 0.015, 0.06, log=True),
        "num_leaves":        trial.suggest_categorical("num_leaves", [63, 127, 255]),
        "max_depth":         trial.suggest_categorical("max_depth", [-1, 10, 14]),
        "min_child_samples": trial.suggest_categorical("min_child_samples", [20, 30, 40, 60]),
        "subsample":         subsample,
        "colsample_bytree":  trial.suggest_categorical("colsample_bytree", [0.7, 0.85, 1.0]),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-6, 0.2, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-6, 0.2, log=True),
        "min_split_gain":    trial.suggest_float("min_split_gain", 0.0, 0.05),
        "bagging_freq":      1 if subsample < 1.0 else 0,  # active le bagging si subsample < 1
        "max_bin":           trial.suggest_categorical("max_bin", [127, 255]),
    }

    m = LGBMClassifier(
        objective=objective_name,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **extra_params,
        **params,
    )
    # ES sur F1_weighted
    m.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric=lgbm_f1_weighted,
        callbacks=[
            lightgbm.early_stopping(stopping_rounds=100),
            lightgbm.log_evaluation(period=0),
        ],
    )

    # score validation (critère d'Optuna)
    yv = m.predict(X_val, num_iteration=getattr(m, "best_iteration_", None))
    f1v = f1_score(y_val, yv, average="weighted")

    # logging utile
    trial.set_user_attr("best_iteration", getattr(m, "best_iteration_", None))
    print(f"[Trial {trial.number+1}] F1_val_w={f1v:.5f} | best_iter={trial.user_attrs.get('best_iteration')} | {params}")

    return f1v

#______________________________________________________________________________
#______________________________________________________________________________
# ---- run Optuna ----
N_TRIALS = 40
tic = time.time()
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
toc = time.time()

print(f"\n✅ Optuna v2 en {toc - tic:.1f}s | Best F1_val_weighted = {study.best_value:.5f}")
print("Best params:", study.best_params, "| best_iteration:", study.best_trial.user_attrs.get("best_iteration"))

# ---- refit final sur tout X_train / test sur X_test ----
best_params = study.best_params.copy()
best_iter = study.best_trial.user_attrs.get("best_iteration")
if best_iter is not None and best_iter > 0:
    best_params["n_estimators"] = int(best_iter)

final = LGBMClassifier(
    objective=objective_name,
    class_weight=CLASS_WEIGHT,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    **extra_params,
    **best_params,
)
final.fit(X_train, y_train)
yp = final.predict(X_test)
print(f"🎯 F1_weighted TEST = {f1_score(y_test, yp, average='weighted'):.5f}")

joblib.dump(final, "lgbm_optuna_v2_best.pkl")

#______________________________________________________________________________
#______________________________________________________________________________
# Mélange de modèles
from sklearn.metrics import f1_score
import numpy as np


Prf   = random_forest_model.predict_proba(X_test) # Pgms 2_2
Plgbm = final.predict_proba(X_test)  

alphas = [0.4 ,0.5, 0.6 ,0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69 ,0.7, 0.8]
for a in alphas:
    P = a*Plgbm + (1-a)*Prf
    y_hat = np.argmax(P, axis=1) if len(np.unique(y_train))>2 else (P[:,1] > 0.5).astype(int)
    print(f"Blend a={a:.1f} -> F1_weighted={f1_score(y_test, y_hat, average='weighted'):.5f}")











