# Guide d'Utilisation - Challenge Spotify Popularity Prediction

## Démarrage Rapide

### Installation

```bash
# Installer les dépendances de base
pip install -r requirements.txt


# Installer les modèles de boosting (recommandé)
pip install lightgbm xgboost catboost
```
---

## Modèles Disponibles

#### 1. Random Forest
```bash
python train_final.py --model random_forest
```
- **Performance:** R² = 0.536
- **Temps:** ~5 minutes
- **Fichier généré:** `submission_random_forest.csv`

#### 2. Ridge Polynomial (R² = 0.264)
```bash
python train_final.py --model polynomial_ridge
```
- **Performance:** R² = 0.264 (stable, pas de sur-apprentissage)
- **Temps:** ~30 secondes
- **Fichier généré:** `submission_polynomial_ridge.csv`

### Modèles Avancés (nécessitent installation)

#### 3. LightGBM (R² = 0.520)
```bash
# Installation (léger, ~1.5 MB)
python -m pip install lightgbm

# Entraînement
python train_final.py --model lightgbm
```
- **Avantages:** Très rapide, gestion efficace des catégorielles
- **Temps:** ~2 minutes

#### 4. XGBoost MEILLEUR BASE (R² = 0.538)
```bash
# Installation (lourd, ~57 MB)
python -m pip install xgboost

# Entraînement
python train_final.py --model xgboost
```
- **Avantages:** Régularisation L1/L2, excellentes performances
- **Temps:** ~3 minutes

### Modèle d'Ensemble (Potentiel Max)
#### 5. Stacking
```bash
python train_final.py --model stacking
```
- **Avantages:** Combine les prédictions de XGBoost, RF et LGBM. Souvent le meilleur score en compétition.

- **Temps:** Très long (~15-25 minutes) car il entraîne 3 modèles.

#### Autres Modèles
catboost: Modèle de base (non optimisé) de CatBoost.
---

## Workflow Recommandé

### Option 1: Utiliser le Meilleur Modèle (Recommandé)

```bash
# Entraîner XGBoost (meilleur modèle de base CV)
python train_final.py --model xgboost
```

**Résultat:** Fichier `submission_xgboost.csv` prêt pour Kaggle

### Option 2: Évaluer Tous les Modèles Disponibles

```bash
# Évaluer avec validation croisée 3-fold
python evaluate_final.py
```

**Résultat:**
- Classement des modèles par R²
- Recommandation automatique
- Fichier `results/evaluation_final.csv`

**Note:** Cette commande évalue uniquement les modèles installés. Si LightGBM/XGBoost ne sont pas installés, ils seront ignorés.

### Option 3: Stacking

```bash
# Entraîner le modèle d'ensemble
python train_final.py --model stacking
```

**Résultat:** `submission_stacking.csv` à tester sur Kaggle

---

## 📈 Performances Attendues

Modèle,R² (test CV),RMSE (estimé),Temps (approx.)
XGBoost,0.538,~15.2,~4 min
Random Forest,0.536,~15.3,~6 min
LightGBM,0.520,~15.5,~2 min
Stacking,(> 0.538),?,~20 min
Ridge Polynomial,0.269,19.1,~30 sec

**Recommandation:** Commencer avec **XGBoost** (meilleur résultat modèle seul, nécessite installation) ou **Random Forest** (bon résultat, pas d'installation supplémentaire), ou **Stacking** meilleur modèle

---

## Fichiers Générés

### Après `train_final.py`

```
submission_<nom_modele>.csv
```

**Format:**
```csv
row_id,popularity
85500,30.25
85501,38.38
...
```


### Après `evaluate_final.py`

```
results/evaluation_final.csv
```

**Contenu:**
- Classement des modèles par R²
- Métriques détaillées (R², RMSE, écart train-test)
- Recommandation du meilleur modèle

---

## Conseils

### Pour Maximiser le Score Kaggle

1. Utiliser XGBoost (R² = 0.538)
2. Soumettre `submission_xgboost_search.csv`
3. Attention au sur-apprentissage (surveiller le score public vs privé)

### Pour la Présentation

1. Montrer l'évaluation comparative (`evaluate_final.py`)
2. Expliquer le choix de Random Forest (optimisation, performance)
3. Documenter les hyperparamètres (voir `docs/rapport.md`)

### Pour Aller Plus Loin (Optionnel)

1. Tester LightGBM et XGBoost
2. Comparer avec Random Forest
3. Analyser les erreurs de prédiction
4. Tester ensemble methods (stacking/blending)

---

## Résumé

### Commandes Essentielles

```bash
# Installation
pip install -r requirements.txt

# Meilleur modèle (recommandé)
python train_final.py --model random_forest

# Évaluation comparative
python evaluate_final.py

# Modèle stable et rapide
python train_final.py --model polynomial_ridge
```

### Fichiers Importants

- `train_final.py` - Script d'entraînement
- `evaluate_final.py` - Script d'évaluation
- `docs/rapport.md` - Rapport complet
- `README.md` - Documentation principale

### Résultats

- **Meilleur modèle:** XGBoost (R² = 0.538)
- **Fichier de soumission:** `submission_xgboost_search.csv`
- **Temps total:** ~3 minutes