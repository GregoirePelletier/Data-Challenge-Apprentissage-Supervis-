# Classif 2.4 : AdaBoost

import numpy as np, pandas as pd, time, joblib, warnings
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterSampler

warnings.filterwarnings("ignore", category=FutureWarning)  # optionnel pour nettoyer les logs

rng = np.random.RandomState(RANDOM_STATE)

#______________________________________________________________________________
#______________________________________________________________________________
# Espace d'hyperparamètres 
grid = {
    "n_estimators":   [100, 200, 400, 800, 1200],
    "learning_rate":  [0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
    "max_depth":      [1, 2, 3, 4, 5],
}

N_TRIALS = 80
scores, models = [], []
tic = time.time()

#______________________________________________________________________________
#______________________________________________________________________________
# Entraitement
for i, p in enumerate(ParameterSampler(grid, n_iter=N_TRIALS, random_state=RANDOM_STATE), 1):
    try:
        base = DecisionTreeClassifier(max_depth=p["max_depth"], random_state=RANDOM_STATE)
        ada = AdaBoostClassifier(
            estimator=base,
            n_estimators=p["n_estimators"],
            learning_rate=p["learning_rate"],
            random_state=RANDOM_STATE,
        )
        ada.fit(X_train, y_train)
        y_pred = ada.predict(X_test)  
        f1w = f1_score(y_test, y_pred, average="weighted")
        scores.append((f1w, p))
        models.append((f1w, ada))
        print(f"Trial {i}: F1_weighted={f1w:.5f} | {p}")
    except Exception as e:
        print(f"⚠️ Skipped trial {i}: {e}")

scores.sort(key=lambda x: x[0], reverse=True)
best_f1, best_params = scores[0]
best_model = max(models, key=lambda x: x[0])[1]

print(f"\n✅ {len(scores)} essais valides en {time.time()-tic:.1f}s | Best F1_weighted = {best_f1:.5f}")
print("Best params:", best_params)
joblib.dump(best_model, "adaboost_best.pkl")
