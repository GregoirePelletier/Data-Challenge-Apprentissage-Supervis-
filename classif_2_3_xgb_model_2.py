from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import xgboost as xgb
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from _export_model_report_pdf import export_model_report_pdf
import itertools, random

# split train/val
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE,
)

param_grid = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.3],
    "reg_lambda": [1, 2],
    "reg_alpha": [0, 0.1],
}

keys = list(param_grid.keys())
all_combos = list(itertools.product(*[param_grid[k] for k in keys]))

# Tirage aléatoire de 100 combinaisons uniques
random.seed(RANDOM_STATE)
sampled_combos = random.sample(all_combos, 100)

best_score, best_params, best_model = -1, None, None

i=0
for combo in sampled_combos:
    print(i)
    params = dict(zip(keys, combo))
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        random_state=RANDOM_STATE,
        n_estimators=1000,
        eval_metric="mlogloss",
        **params
    )
    model.fit(
        X_fit, y_fit,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    f1 = f1_score(y_val, model.predict(X_val), average="macro")

    if f1 > best_score:
        best_score, best_model, best_params = f1, model, params

    i=i+1

print("Best F1-macro:", round(best_score, 4))
print("Best params:", best_params)




#______________________________________________________________________________
#______________________________________________________________________________
# MODELE SELECTIONNE

# Best params: {'max_depth': 7, 'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 1.0}
#Best params: {'max_depth': 8, 'learning_rate': 0.1, 'subsample': 1.0, 'colsample_bytree': 1.0, 'min_child_weight': 1, 'gamma': 0.1, 'reg_lambda': 2, 'reg_alpha': 0.1}

model_name = "classif_2_2_random_forest_mod_2"

xgb_model = xgb.XGBClassifier(
    learning_rate = 0.105,
    max_depth = 7,
    subsample = 0.8,
    colsample_bytree = 1.0,
    #min_child_weight= 1.0, 
    #gamma= 0.1,
    #reg_lambda= 2,
    #reg_alpha= 0.1,
    random_state=RANDOM_STATE,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss"     # 👉 early stopping basé sur log-loss
)

xgb_model.fit(
    X_fit, y_fit,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=True
)


print(f"Metrics train:\n\tf1: {f1_score(y_train, xgb_model.predict(X_train), average='macro'):.4f}\n"
      f"Metrics test:\n\tf1 score: {f1_score(y_test, xgb_model.predict(X_test), average='macro'):.4f}")

# === Prédictions ===

predictions_train = xgb_model.predict(X_train) ## The predicted values for the train dataset
predictions_test = xgb_model.predict(X_test) ## The predicted values for the test dataset

# === Matrice de confusion brute ===

cm = confusion_matrix(y_test, predictions_test)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=xgb_model.classes_)
disp.plot(cmap="Greens")
plt.show()

# === Export Resultat ===

f1, acc, name, _ = export_model_report_pdf(xgb_model, X_test, y_test, pdf_path= chemin_sortie+"\\classif_2_3_xgb_model.pdf", 
                                       title = "xgb_model")

print("f1:", f1, "| Modèle:", name)

