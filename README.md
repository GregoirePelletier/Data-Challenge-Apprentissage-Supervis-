# Spotify Popularity Prediction Project

Prédiction de la popularité d'un titre musical Spotify (0-100) en utilisant l'apprentissage supervisé.

**Métrique d'évaluation:** Coefficient de détermination (R²)
**Meilleur modèle:** Random Forest optimisé (R² = 0.472)

---

## Structure du Projet

```
spotify-predire-la-popularite-dun-titre/
├── src/                    # Modules Python
│   ├── data_preparation.py # Préparation des données
│   └── pipelines.py        # Pipelines de prétraitement
├── docs/                   # Documentation
│   ├── CONTEXT.md          # Contexte du challenge
│   └── rapport.md          # Rapport complet
├── results/                # Résultats et soumissions
│   └── submission_polynomial_ridge.csv
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
# LightGBM (léger, ~1.5 MB)
python -m pip install lightgbm

# XGBoost (lourd, ~57 MB)
python -m pip install xgboost
```

**Note:** Les modèles `polynomial_ridge` et `random_forest` fonctionnent sans ces dépendances.

---

## Utilisation

### 1. Évaluer les modèles (recommandé)

Comparer les performances avec validation croisée 3-fold :

```bash
python evaluate_final.py
```

**Sortie:**
- Résultats détaillés pour chaque modèle
- Classement par R²
- Recommandation du meilleur modèle
- Fichier `results/evaluation_final.csv`

### 2. Entraîner un modèle

Entraîner un modèle spécifique et générer une soumission :

```bash
python train_final.py --model <nom_du_modele>
```

**Modèles disponibles:**
- `polynomial_ridge` : Ridge avec features polynomiales (R² = 0.264)
- `random_forest` : Random Forest optimisé (R² = 0.472) - **MEILLEUR**
- `lightgbm` : LightGBM (rapide et efficace)
- `xgboost` : XGBoost (régularisation L1/L2)

**Exemples:**
```bash
# Meilleur modèle
python train_final.py --model random_forest

# Modèle le plus rapide
python train_final.py --model lightgbm
```

**Sortie:** Fichier `submission_<nom_du_modele>.csv` prêt pour Kaggle

---

## Résultats

### Performances (Validation Croisée)

| Modèle | R² (test) | RMSE | Notes |
|--------|-----------|------|-------|
| **Random Forest** | **0.472** | 16.20 | Meilleur, attention sur-apprentissage |
| Ridge Polynomial | 0.264 | 19.14 | Stable, pas de sur-apprentissage |
| LightGBM | À évaluer | - | Rapide, efficace |
| XGBoost | À évaluer | - | Régularisation forte |

**Meilleur modèle:** Random Forest optimisé
- **Configuration:** 500 arbres, max_depth=None, max_features=0.7
- **Amélioration:** +210% vs baseline (0.152 → 0.472)
- **Attention:** Sur-apprentissage modéré (R² train = 0.756)

---

## Méthodologie

### 1. Analyse Exploratoire (EDA)
- 85,500 observations, aucune valeur manquante
- Corrélations très faibles avec popularité (max |r| = 0.094)
- 114 genres musicaux bien équilibrés
- Features asymétriques (duration_ms, speechiness, etc.)

### 2. Prétraitement
- **Encodage cyclique** pour `key`: sin(2π×key/12), cos(2π×key/12)
- **StandardScaler** pour features numériques
- **OneHotEncoder** pour `track_genre` (114 genres)
- **Features polynomiales** (degree=2) pour Ridge uniquement

### 3. Modèles Sélectionnés

**Ridge Polynomial:**
- Features polynomiales + régularisation L2
- Pas de sur-apprentissage, stable et rapide

**Random Forest (optimisé):**
- Hyperparamètres optimisés par RandomizedSearchCV
- 500 arbres, profondeur illimitée
- Meilleure performance mais sur-apprentissage modéré

**LightGBM:**
- Implémentation optimisée du Gradient Boosting
- Très rapide, gestion efficace des catégorielles
- Paramètres: 300 estimators, lr=0.05, max_depth=7

**XGBoost:**
- Gradient Boosting avec régularisation L1/L2
- Excellentes performances sur données tabulaires
- Paramètres: 300 estimators, lr=0.05, max_depth=6

### 4. Évaluation
- **Validation croisée 3-fold** (allégée pour rapidité)
- **Métriques:** R², RMSE, écart train-test
- **Détection du sur-apprentissage**

---

## Choix Techniques

### Pourquoi 3-fold au lieu de 5-fold ?
- **Rapidité:** 40% plus rapide
- **Suffisant:** Dataset large (85,500 observations)
- **Compromis:** Précision vs temps de calcul

### Pourquoi Random Forest avec 500 arbres ?
- **Optimisation:** Trouvé par RandomizedSearchCV (50 itérations)
- **Performance:** R² = 0.472 (meilleur résultat)
- **Compromis:** Temps d'entraînement acceptable (~5 min)

### Pourquoi LightGBM et XGBoost ?
- **Efficacité:** Plus rapides que sklearn GradientBoosting
- **Performance:** Souvent meilleurs sur données tabulaires
- **Régularisation:** Meilleure gestion du sur-apprentissage

---

## Documentation

- **`docs/CONTEXT.md`** : Contexte et objectifs du challenge
- **`docs/rapport.md`** : Rapport complet avec méthodologie et résultats

---

## Auteur

Grégoire - M2 Apprentissage Supervisé (2025)
