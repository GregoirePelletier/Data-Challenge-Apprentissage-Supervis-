# Classif 2.2 : Random Forrest Gridsearch 

import numpy as np, pandas as pd, joblib, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

rng = np.random.RandomState(RANDOM_STATE)

# ===== Fonction pour tirer aléatoirement des hyperparamètres =====
def sample_params(n, rng):
    for _ in range(n):
        if rng.rand() < 0.5:
            max_features = rng.choice(['sqrt', 'log2'])
        else:
            max_features = float(rng.choice([0.3, 0.5, 0.7, 0.9]))

        bootstrap = bool(rng.choice([True, False]))
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

# ====== Boucle de recherche manuelle ======
N_TRIALS = 120
scores, models = [], []
tic = time.time()

i=0
for p in sample_params(N_TRIALS, rng):
    print(i)
    try:
        rf = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
            **p
        )
        rf.fit(X_train, y_train)
        f1 = f1_score(y_test, rf.predict(X_test), average="weighted")  # <<< F1 pondéré
        scores.append((f1, p))
        models.append((f1, rf))
        print(f"Trial done: F1_weighted={f1:.5f}")
    except Exception as e:
        print(f"⚠️ Skipped: {e}")
    i=i+1

toc = time.time()

# ===== Résultats =====
scores.sort(key=lambda x: x[0], reverse=True)
best_f1, best_params = scores[0]
best_model = sorted(models, key=lambda x: x[0], reverse=True)[0][1]

print(f"\n✅ {len(scores)} essais valides en {toc-tic:.1f}s | Best F1_weighted = {best_f1:.5f}")
print("Best params:", best_params)

# ===== Top 10 =====
top10 = pd.DataFrame([{"F1_weighted": s, **p} for (s, p) in scores[:10]])
print("\nTop 10:\n", top10)

# ===== Sauvegarde du meilleur modèle =====
joblib.dump(best_model, "rf_single_best.pkl")

#______________________________________________________________________________
#______________________________________________________________________________
# MODELE SELECTIONNE

model_name = "classif_2_2_random_forest_grille"

random_forest_model = RandomForestClassifier(
        class_weight='balanced',
        max_depth=48,
        max_features=0.5,
        min_impurity_decrease=1e-07,
        min_samples_leaf=2,
        min_samples_split=5,
        n_estimators=600,
        n_jobs=-1,
        random_state=55
    ).fit(X_train,y_train)

print(f"Metrics train:\n\tf1: {f1_score(y_train, random_forest_model.predict(X_train), average='weighted'):.4f}\n"
      f"Metrics test:\n\tf1 score: {f1_score(y_test, random_forest_model.predict(X_test), average='weighted'):.4f}")

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

f1, acc, name, _ = export_model_report_pdf(random_forest_model, X_test, y_test,
                                    pdf_path= str(chemin_sortie / "classif_2_2_random_forest_grille.pdf"),
                                       title = "random_forest_model")

print("f1:", f1, "| Modèle:", model_name)
print("acc:", acc, "| Modèle:", model_name)
