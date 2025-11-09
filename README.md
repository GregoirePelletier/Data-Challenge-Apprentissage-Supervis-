# Spotify Popularity Prediction Project

Prédiction de la popularité d'un titre musical Spotify (0-100) en utilisant l'apprentissage supervisé.

**Métrique d'évaluation:** Coefficient de détermination (R²)
**Modèle le plus performant:** XGBoost optimisé (R² = 0.538)

---

## Structure du Projet

```
spotify-predire-la-popularite-dun-titre/
├── src/                    # Modules Python
│   ├── data_preparation.py # Préparation des données
│   └── pipelines.py        # Pipelines de prétraitement
├── docs/                   # Documentation
│   └── rapport.md          # Rapport complet
├── results/                # Résultats et soumissions
│   └── evaluation_final.csv (une fois le script evaluate_final.py utilisé)
├── figures/                # Graphiques EDA
│   ├── eda_popularity.png
│   ├── eda_correlation.png
│   ├── eda_genres.png
│   └── eda_distributions.png
├── train_final.py          # Script d'entraînement
├── evaluate_final.py       # Évaluation avec validation croisée
├── requirements.txt        # Dépendances Python
├── train_data.csv          # Données (85,500 observations)
└── test_data.csv           # Données (28,500 observations)
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Dépendances optionnelles** (pour LightGBM et XGBoost):
```bash

python -m pip install lightgbm xgboost catboost
```

**Note:** Les modèles `polynomial_ridge` et `random_forest` fonctionnent sans ces dépendances.

---

## Utilisation

### 1. Évaluer les modèles

Comparer les performances avec validation croisée 3-fold :

```bash
python evaluate_final.py
```

**Sortie:**
- Résultats détaillés pour chaque modèle
- Classement par R²
- Fichier `results/evaluation_final.csv`

### 2. Entraîner un modèle

Entraîner un modèle spécifique et générer une soumission :

```bash
python train_final.py --model <nom_du_modele>
```

**Modèles disponibles:**
rpolynomial_ridge: Ridge avec features polynomiales (R² ~ 0.269)

random_forest: Random Forest optimisé (R² = 0.536)

lightgbm: LightGBM optimisé (R² = 0.520)

xgboost: XGBoost optimisé (R² = 0.538) - MEILLEUR MODÈLE DE BASE

catboost: CatBoost (gestion native des catégorielles)

stacking: Modèle d'ensemble (XGB+RF+LGBM) - MEILLEUR POTENTIEL

Lancer une recherche d'hyperparamètres :

random_forest_search, lightgbm_search, xgboost_search, catboost_search

**Exemples:**
```bash
# Entraîner le meilleur modèle de base
python train_final.py --model xgboost

# Entraîner le modèle d'ensemble (potentiel de score le plus élevé)
python train_final.py --model stacking
```

**Sortie:** Fichier `submission_<nom_du_modele>.csv` prêt pour Kaggle

---

## Résultats

### Performances (Validation Croisée)

Modèle,R² (test CV),Notes
XGBoost,0.538,Meilleur modèle de base
Random Forest,0.536,Performance quasi-identique
LightGBM,0.520,Performant et rapide
CatBoost,(À évaluer),Potentiel élevé
Ridge Polynomial,0.269,Baseline linéaire stable
Stacking,(> 0.538),Potentiel le plus élevé (combine les 3 meilleurs)

**Modèle le plus performant:** XGBoost optimisé
- **Configuration:** n_estimators=500, max_depth=12, learning_rate=0.1, subsample=0.7, colsample_bytree=0.7, reg_lambda=2.0, reg_alpha=0, min_child_weight=3, gamma=0
- **Amélioration:** +254% vs baseline (0.152 → 0.538)
- **Note:** Sur-apprentissage modéré (R² train = 0.70-0.75)

---

## Méthodologie

### 1. Analyse Exploratoire (EDA)
- 85,500 observations, aucune valeur manquante
- Corrélations très faibles avec popularité (max |r| = 0.094)
- 114 genres musicaux bien équilibrés
- Features asymétriques (duration_ms, speechiness, etc.)

### 2. Prétraitement
Encodage cyclique pour key (tonalité).

Transformation Log pour features asymétriques (duration_ms, speechiness...).

Features d'Interaction (energy * loudness, speech_density...).

Discrétisation (Binning): duration_ms et tempo groupés en 10 quantiles.

Encodage Catégoriel:

TargetEncoder pour track_genre (haute cardinalité).

OneHotEncoder pour les features à faible cardinalité (mode, time_signature...).

### 3. Modèles Sélectionnés

StackingRegressor (Ensemble): Modèle final combinant XGBoost, RandomForest et LightGBM via un méta-modèle RidgeCV pour agréger leurs prédictions.
**Ridge Polynomial:**
- Features polynomiales + régularisation L2
- Pas de sur-apprentissage, stable et rapide

**Random Forest (optimisé):**
- Hyperparamètres optimisés par RandomizedSearchCV
- 500 arbres, profondeur illimitée
- Meilleure performance mais sur-apprentissage modéré

**LightGBM (optimisé):**
- Implémentation optimisée du Gradient Boosting
- Très rapide, gestion efficace des catégorielles
- Hyperparamètres optimisés par RandomizedSearchCV
- Paramètres: n_estimators=500, num_leaves=127, max_depth=12, learning_rate=0.2, subsample=0.6, colsample_bytree=0.8, reg_lambda=1.5, reg_alpha=0.01

**XGBoost (optimisé):**
- Gradient Boosting avec régularisation L1/L2
- Excellentes performances sur données tabulaires
- Hyperparamètres optimisés par RandomizedSearchCV
- Paramètres: n_estimators=500, max_depth=12, learning_rate=0.1, subsample=0.7, colsample_bytree=0.7, reg_lambda=2.0, reg_alpha=0, min_child_weight=3, gamma=0

**Catboost:**

### 4. Évaluation
- **Validation croisée 3-fold** (allégée pour rapidité)
- **Métriques:** R², RMSE, écart train-test
- **Détection du sur-apprentissage**
Pourquoi 3-fold au lieu de 5-fold ?
- **Rapidité:** 40% plus rapide
- **Suffisant:** Dataset large (85,500 observations)
- **Compromis:** Précision vs temps de calcul
---

## Documentation

- **`docs/rapport.md`** : Rapport complet avec méthodologie et résultats

---
