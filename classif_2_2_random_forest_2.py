

import numpy as np, pandas as pd, joblib, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

rng = np.random.RandomState(RANDOM_STATE)

def sample_params(n, rng):
    for _ in range(n):
        # max_features: tirer soit une STRING ('sqrt'/'log2'), soit un FLOAT (0,1]
        if rng.rand() < 0.5:
            max_features = rng.choice(['sqrt', 'log2'])
        else:
            max_features = float(rng.choice([0.3, 0.5, 0.7, 0.9]))  # flotteurs purs

        bootstrap = bool(rng.choice([True, False]))
        # max_samples uniquement si bootstrap=True (sinon None)
        max_samples = float(rng.choice([0.7, 0.85, 0.95])) if bootstrap and rng.rand() < 0.7 else None

        yield {
            "n_estimators": int(rng.choice([200, 400, 600, 800, 1000, 1200, 1600])),
            "max_depth": rng.choice([None, 8, 12, 16, 24, 32, 48, 64]),
            "max_features": max_features,
            "min_samples_split": int(rng.choice([2, 5, 10, 20, 40, 80])),
            "min_samples_leaf": int(rng.choice([1, 2, 3, 4, 6, 8])),
            "bootstrap": bootstrap,
            "max_leaf_nodes": rng.choice([None, 256, 512, 1024]),
            "min_impurity_decrease": float(rng.choice([0.0, 1e-7, 1e-6, 1e-5])),
            "max_samples": max_samples,
        }

N_TRIALS = 120
scores, models = [], []
tic = time.time()

for p in sample_params(N_TRIALS, rng):
    try:
        rf = RandomForestClassifier(
            random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced", **p
        )
        rf.fit(X_train, y_train)
        f1 = f1_score(y_test, rf.predict(X_test), average="macro")
        scores.append((f1, p))
        models.append((f1, rf))
    except Exception as e:
        print(f"⚠️ Skipped: {e}")

toc = time.time()

scores.sort(key=lambda x: x[0], reverse=True)
best_f1, best_params = scores[0]
best_model = sorted(models, key=lambda x: x[0], reverse=True)[0][1]

print(f"\n✅ {len(scores)} valid trials in {toc-tic:.1f}s | Best F1_macro = {best_f1:.5f}")
print("Best params:", best_params)

top10 = pd.DataFrame([{"F1_macro": s, **p} for (s, p) in scores[:10]])
print("\nTop 10:\n", top10)

joblib.dump(best_model, "rf_single_best.pkl")



# (Optionnel) Importances rapides
imp = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(15)
print("\nTop-15 importances:")
print(imp.round(4))


#______________________________________________________________________________
#______________________________________________________________________________
# MODELE SELECTIONNE

model_name = "classif_2_2_random_forest_best"


random_forest_model = RandomForestClassifier(n_estimators= 1000, max_depth= 12, 
                                             max_features= 'sqrt', min_samples_split= 80, min_samples_leaf=8,
 bootstrap= False, max_leaf_nodes= 256, min_impurity_decrease= 1e-05, max_samples= None).fit(X_train,y_train)


print(f"Metrics train:\n\tf1: {f1_score(y_train, random_forest_model.predict(X_train), average='macro'):.4f}\n"
      f"Metrics test:\n\tf1 score: {f1_score(y_test, random_forest_model.predict(X_test), average='macro'):.4f}")

print("F1_weighted global :", f1_score(y_test, random_forest_model.predict(X_test), average="weighted"))

# === Prédictions ===

predictions_train = random_forest_model.predict(X_train) ## The predicted values for the train dataset
predictions_test = random_forest_model.predict(X_test) ## The predicted values for the test dataset

# === Matrice de confusion brute ===

cm = confusion_matrix(y_test, predictions_test)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=random_forest_model.classes_)
disp.plot(cmap="Greens")
plt.show()

# === Export Resultat ===

f1, acc, name, _ = export_model_report_pdf(random_forest_model, X_test, y_test, pdf_path= chemin_sortie+"\\"+model_name+".pdf", 
                                       title = "random_forest_model")

print("f1:", f1, "| Modèle:", model_name)
print("acc:", acc, "| Modèle:", model_name)
