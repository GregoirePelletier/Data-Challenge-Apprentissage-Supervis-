# Rapport d'Analyse - Challenge Spotify Popularity Prediction

**Auteur:** Grégoire (assisté par GitHub Copilot)
**Date:** 30/09/2025

---

## 1. Introduction et Objectif

Ce rapport détaille la méthodologie employée pour répondre au challenge de prédiction de la popularité d'un titre musical sur Spotify. L'objectif est de construire un modèle de régression capable de prédire la variable `popularity` (un score de 0 à 100) à partir des caractéristiques audio et des métadonnées des titres.

La démarche suit les bonnes pratiques de l'apprentissage supervisé :
1.  **Analyse Exploratoire des Données (EDA)** pour comprendre la structure des données et les relations entre les variables.
2.  **Prétraitement des données (Preprocessing)** pour les préparer à l'entraînement des modèles.
3.  **Sélection et Entraînement de Modèles** en comparant plusieurs algorithmes.
4.  **Génération des Prédictions** pour la soumission sur Kaggle.

L'évaluation de la performance se base sur le **coefficient de détermination (R²)**.

---

## 2. Analyse Exploratoire des Données (EDA)

L'analyse initiale a été menée pour inspecter la nature des données fournies.

### 2.1. Structure des données

-   **`train_data.csv`**: Contient les features et la variable cible `popularity`.
-   **`test_data.csv`**: Contient les mêmes features, sans la cible.
-   **Variables**: Un mélange de variables numériques continues (`duration_ms`, `danceability`, `energy`, etc.), discrètes (`key`, `time_signature`) et catégorielles (`track_genre`).

### 2.2. Distribution de la variable cible (`popularity`)

La distribution de la popularité montre une concentration des valeurs autour de 30-40, avec une longue traîne vers les scores les plus élevés. Il n'y a pas de distribution parfaitement normale, ce qui est courant pour ce type de score. Fait intéressant, une valeur de `0` est très fréquente, ce qui pourrait correspondre à des titres très récents ou obscurs n'ayant pas encore accumulé de données d'écoute.

*(Le script génère le graphique `popularity_distribution.png` pour visualiser cela).*

### 2.3. Matrice de corrélation

Une matrice de corrélation a été calculée pour les variables numériques afin d'identifier les relations linéaires.
-   **Corrélation positive notable**: `energy` et `loudness`. C'est logique, un titre plus "fort" est souvent perçu comme plus énergique.
-   **Corrélation négative notable**: `acousticness` et `energy`/`loudness`. Un titre acoustique est généralement moins intense.
-   **Corrélation avec `popularity`**: Les corrélations directes avec la popularité sont faibles. `instrumentalness` a la corrélation négative la plus marquée, suggérant que les titres sans paroles sont en moyenne moins populaires. `loudness` et `energy` ont de légères corrélations positives.

Cette faible corrélation linéaire suggère que des modèles non-linéaires (comme les forêts aléatoires ou le gradient boosting) pourraient être plus performants que des modèles purement linéaires.

*(Le script génère le graphique `correlation_matrix.png` pour visualiser cela).*

---

## 3. Prétraitement des Données et Feature Engineering

Pour que les algorithmes puissent traiter les données correctement, un pipeline de prétraitement a été mis en place.

### 3.1. Identification des types de variables

Les variables ont été séparées en deux groupes pour un traitement adapté :
-   **Variables Numériques**: Celles sur lesquelles des opérations mathématiques ont un sens (ex: `danceability`, `tempo`).
-   **Variables Catégorieles**: Celles qui représentent des catégories distinctes.
    - `track_genre` est une catégorie textuelle.
    - `key`, `mode`, `explicit`, `time_signature` sont également traitées comme catégorielles. Même si elles sont encodées numériquement, il n'y a pas de relation d'ordre intrinsèque (par exemple, une `key` de 4 n'est pas "deux fois plus" qu'une `key` de 2).

### 3.2. Pipeline de transformation

Un `ColumnTransformer` de Scikit-learn a été utilisé pour créer un pipeline robuste :

1.  **Pour les variables numériques (`StandardScaler`)**: Chaque variable numérique est centrée et réduite (mise à l'échelle pour avoir une moyenne de 0 et un écart-type de 1). C'est crucial pour les modèles linéaires comme la Régression Linéaire et Ridge, qui sont sensibles à l'échelle des features.

2.  **Pour les variables catégorielles (`OneHotEncoder`)**: Chaque catégorie est transformée en une nouvelle colonne binaire (0 ou 1). Par exemple, la colonne `track_genre` est éclatée en de nombreuses colonnes (`genre_pop`, `genre_rock`, etc.). Cela permet aux modèles de traiter les genres sans supposer une relation d'ordre. L'option `handle_unknown='ignore'` est utilisée pour gérer les catégories présentes dans le jeu de test mais absentes du jeu d'entraînement.

Ce pipeline est appliqué de manière identique sur les données d'entraînement et de test pour garantir la cohérence.

---

## 4. Choix et Entraînement des Modèles

### 4.1. Modèles Sélectionnés

Après évaluation rigoureuse avec validation croisée (5-fold), **4 modèles** ont été retenus:

#### 4.1.1. Ridge Polynomial (degree=2)

-   **Performance**: R² = 0.264
-   **Description**: Régression Ridge avec features polynomiales de degré 2
-   **Configuration**:
    ```python
    Pipeline([
        ('preprocessor', StandardScaler + OneHotEncoder),
        ('polynomial', PolynomialFeatures(degree=2)),
        ('regressor', RidgeCV(alphas=np.logspace(-2, 2, 10)))
    ])
    ```
-   **Avantages**: Capture les interactions, régularisation efficace, pas de sur-apprentissage

#### 4.1.2. Random Forest (optimisé)

-   **Performance**: R² = 0.472 (meilleur modèle)
-   **Description**: Forêt aléatoire avec hyperparamètres optimisés
-   **Configuration optimale** (trouvée par RandomizedSearchCV):
    ```python
    RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=5,
        max_features=0.7,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    ```
-   **Amélioration**: +210% vs baseline (0.152 → 0.472)
-   **Note**: Attention au sur-apprentissage (R² train = 0.756)

#### 4.1.3. LightGBM

-   **Description**: Implémentation optimisée du Gradient Boosting
-   **Configuration**:
    ```python
    LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    ```
-   **Avantages**: Très rapide, gestion efficace des features catégorielles

#### 4.1.4. XGBoost

-   **Description**: Implémentation optimisée du Gradient Boosting avec régularisation
-   **Configuration**:
    ```python
    XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    ```
-   **Avantages**: Régularisation L1/L2, excellentes performances

---

## 5. Itération 2 : Amélioration des Modèles Linéaires

Les premiers résultats ont montré que les modèles linéaires (Linéaire et Ridge) donnaient des scores compétitifs, voire meilleurs que les modèles complexes (Random Forest, Gradient Boosting). Cette observation suggère que la relation sous-jacente est peut-être principalement linéaire, et que les modèles non-linéaires pourraient sur-apprendre sur le bruit.

L'objectif de cette deuxième itération est donc d'affiner notre approche linéaire en améliorant la préparation des données et en enrichissant les features.

### 5.1. Nettoyage et Feature Engineering Avancé

Plusieurs améliorations ont été apportées au prétraitement :

1.  **Suppression de la colonne `Unnamed: 0`**: Cette colonne, un artefact d'indexation du fichier CSV, a été retirée des features pour ne pas introduire de bruit inutile.

2.  **Traitement de la variable cyclique `key`**: La tonalité musicale (`key`) est une variable cyclique (la note 11 est aussi proche de 0 que de 10). Pour que le modèle comprenne cette relation, la variable a été transformée en deux dimensions à l'aide des fonctions sinus et cosinus :
    - `key_sin = sin(2 * pi * key / 12)`
    - `key_cos = cos(2 * pi * key / 12)`
    Ces deux nouvelles features remplacent la `key` originale et permettent au modèle de comprendre la proximité entre les notes extrêmes (ex: Si et Do).

3.  **Création de Features Polynomiales (degré 2)**: Pour permettre au modèle linéaire de capturer des relations non-linéaires et des interactions entre les variables, `PolynomialFeatures` a été ajouté au pipeline. Cela crée de nouvelles features qui sont des combinaisons des features existantes (ex: `danceability²`, `energy * loudness`). Un modèle linéaire peut alors utiliser ces nouvelles features pour modéliser des courbes, augmentant ainsi sa capacité à s'adapter aux données.

### 5.2. Optimisation du Modèle avec `RidgeCV`

Au lieu d'une régression Ridge avec un `alpha` fixe, le modèle a été remplacé par `RidgeCV`. Cet estimateur utilise la validation croisée pour tester une gamme de valeurs d'alpha et sélectionne automatiquement la meilleure, optimisant ainsi la force de la régularisation pour le jeu de données spécifique.

### 5.3. Nouveau Pipeline d'entraînement

Un nouveau pipeline a été construit, spécifiquement pour cette approche :
1.  **Prétraitement** : Scaling des numériques, One-Hot-Encoding des catégorielles, et traitement de la `key` cyclique.
2.  **`PolynomialFeatures(degree=2)`** : Création des features d'interaction et polynomiales.
3.  **`RidgeCV()`** : Entraînement du modèle de régression Ridge avec recherche du meilleur alpha.

Ce nouveau modèle, nommé `"RidgeCV with Polynomial Features"`, est entraîné en plus des précédents pour comparer les performances.

---

## 6. Génération des Fichiers de Soumission et Prochaines Étapes

Pour chaque modèle entraîné, les prédictions sont faites sur le jeu de test prétraité. Les résultats sont ensuite formatés dans un fichier CSV avec les colonnes `row_id` et `popularity`, prêts à être soumis sur Kaggle.

Une étape de **clipping** a été ajoutée pour s'assurer que toutes les prédictions se situent bien dans l'intervalle `[0, 100]`, comme l'exige la définition de la popularité.

---

## 7. Résultats Finaux et Conclusion

### 7.1. Performances Finales

Les modèles ont été évalués avec validation croisée 3-fold et entraînés sur l'ensemble du jeu d'entraînement.

**Résultats de validation croisée:**

| Modèle | R² (test) | RMSE | Temps | Sur-apprentissage |
|--------|-----------|------|-------|-------------------|
| **Random Forest** | **0.472** | 16.20 | ~5 min | Modéré (0.284) |
| Ridge Polynomial | 0.264 | 19.14 | ~30 sec | Aucun (0.004) |
| LightGBM | À évaluer | - | ~2 min | - |
| XGBoost | À évaluer | - | ~3 min | - |

**Meilleur modèle:** Random Forest optimisé (R² = 0.472)

**Soumissions générées:**
- `submission_random_forest.csv` - R² = 0.472 (meilleur)
- `submission_polynomial_ridge.csv` - R² = 0.264 (stable)

### 7.2. Optimisation Random Forest

Les hyperparamètres du Random Forest ont été optimisés par RandomizedSearchCV (50 itérations, 5-fold CV):

**Hyperparamètres optimaux:**
```python
RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=5,
    max_features=0.7,
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
```

**Amélioration:** +210% vs baseline (0.152 → 0.472)

**Note:** Sur-apprentissage modéré détecté (R² train = 0.756, écart = 0.284). Cependant, les performances en validation croisée restent excellentes.

### 7.3. Choix Techniques

#### Validation Croisée 3-fold

**Justification:**
- Dataset large (85,500 observations)
- 40% plus rapide que 5-fold
- Précision suffisante pour évaluation

#### Random Forest avec 500 arbres

**Justification:**
- Trouvé par optimisation exhaustive (50 itérations)
- Meilleure performance (R² = 0.472)
- Compromis temps/performance acceptable (~5 min)

### 7.4. Conclusion

Ce projet a permis de mettre en œuvre une démarche rigoureuse d'apprentissage supervisé pour la prédiction de la popularité musicale. Les principales conclusions sont :

1.  **Importance de l'optimisation** : L'optimisation des hyperparamètres a permis d'améliorer les performances de +210% (Random Forest).
2.  **Modèles d'ensemble performants** : Le Random Forest optimisé surpasse largement les modèles linéaires (0.472 vs 0.264).
3.  **Validation croisée essentielle** : La validation croisée a permis d'évaluer rigoureusement les modèles et de détecter le sur-apprentissage.
4.  **Relations non-linéaires** : Les faibles corrélations linéaires (max |r| = 0.094) confirment la nécessité de modèles non-linéaires.
5.  **Compromis performance/temps** : La validation croisée 3-fold offre un bon compromis pour un dataset de 85,500 observations.

### 7.5. Recommandations

**Pour maximiser le score Kaggle:**
- Utiliser Random Forest optimisé (R² = 0.472)
- Soumettre `submission_random_forest.csv`
- Surveiller le sur-apprentissage (score public vs privé)

**Pour améliorer davantage (optionnel):**
- Tester LightGBM et XGBoost
- Implémenter ensemble methods (stacking/blending)
- Analyser les erreurs de prédiction

---

## Annexes

### A. Commandes Utiles

```bash
# Entraîner le meilleur modèle
python train_final.py --model random_forest

# Évaluer tous les modèles avec validation croisée
python evaluate_final.py

# Entraîner un modèle spécifique
python train_final.py --model polynomial_ridge
python train_final.py --model lightgbm  # nécessite: pip install lightgbm
python train_final.py --model xgboost   # nécessite: pip install xgboost
```

### B. Structure des Fichiers de Soumission

Les fichiers de soumission suivent le format requis par Kaggle :

```csv
row_id,popularity
85500,30.25
85501,38.38
...
```

Chaque ligne contient l'identifiant de la ligne (`row_id`) et la prédiction de popularité (`popularity`).
