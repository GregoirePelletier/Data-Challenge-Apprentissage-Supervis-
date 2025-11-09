import numpy as np
import pandas as pd
from scipy.sparse import hstack
from collections import Counter

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

from imblearn.combine import SMOTETomek


# ================================
# 1. Séparation types de variables
# ================================

var = [x for x in df.columns if x != target] ## Removing our target variable

X_train, X_test, y_train, y_test = train_test_split(df[var], df[target],\
                train_size = 0.8, random_state = RANDOM_STATE)

num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

print("Numériques :", num_cols)
print("Catégorielles :", cat_cols)

# ================================
# 2. Encodage One-Hot des catégorielles
# ================================
# On fit uniquement sur X_train pour éviter la fuite
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=0.05)
ohe.fit(X_train[cat_cols])

Xtr_num = X_train[num_cols].to_numpy()
Xte_num = X_test[num_cols].to_numpy()

Xtr_cat = ohe.transform(X_train[cat_cols])
Xte_cat = ohe.transform(X_test[cat_cols])

# Assemblage en sparse matrix (num + cat)
Xtr_all = hstack([Xtr_num, Xtr_cat])
Xte_all = hstack([Xte_num, Xte_cat])

print("Avant SMOTE+Tomek :", Counter(y_train))

# ================================
# 3. Ré-échantillonnage SMOTE + Tomek
# ================================
smt = SMOTETomek(random_state=42, n_jobs=-1)
X_train_res, y_train_res = smt.fit_resample(Xtr_all, y_train)

print("Après SMOTE+Tomek :", Counter(y_train_res))

# ================================
# 4. Normalisation standard (facultative mais utile pour SVM)
# ================================
scaler = StandardScaler(with_mean=False)  # False car données sparse
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(Xte_all)

# ================================
# 5. Entraînement SVM linéaire
# ================================
clf = LinearSVC(
    C=1.0,
    class_weight="balanced",  # utile si déséquilibre
    max_iter=5000,
    random_state=42
)
clf.fit(X_train_res_scaled, y_train_res)

model = RandomForestClassifier(
   n_estimators = 200,
    max_depth = 64, 
    max_features = "sqrt",
    min_samples_split = 2,
    min_samples_leaf = 1,
    random_state=42

)
model.fit(X_train_res_scaled, y_train_res)

# ================================
# 6. Prédictions et scores
# ================================
y_pred_train = clf.predict(X_train_res_scaled)
y_pred_test = clf.predict(X_test_scaled)

y_pred_train = model.predict(X_train_res_scaled)
y_pred_test = model.predict(X_test_scaled)

print("\n=== Résultats ===")
print(f"Accuracy (train): {accuracy_score(y_train_res, y_pred_train):.4f}")
print(f"F1 macro (train): {f1_score(y_train_res, y_pred_train, average='macro'):.4f}")
print(f"Accuracy (test) : {accuracy_score(y_test, y_pred_test):.4f}")
print(f"F1 macro (test) : {f1_score(y_test, y_pred_test, average='macro'):.4f}")
