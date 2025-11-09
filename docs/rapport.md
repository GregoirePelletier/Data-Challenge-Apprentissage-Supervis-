# Rapport d'Analyse - Challenge Spotify Popularity Prediction

**Auteur:** Grégoire Pelletier, Rémi Saout

-----

## 1\. Introduction et Objectif

Ce rapport détaille la méthodologie itérative employée pour prédire la **popularité** d'un titre musical sur Spotify (un score de 0 à 100) en se basant sur ses caractéristiques audio et ses métadonnées.

La démarche suit une approche chronologique et les bonnes pratiques de l'apprentissage supervisé :

1.  **Analyse Exploratoire des Données (EDA)** pour comprendre la structure des données et les relations entre les variables.
2.  **Prétraitement des données (Preprocessing)** pour les préparer à l'entraînement des modèles.
3.  **Sélection et Entraînement de Modèles** en comparant plusieurs algorithmes.
4.  **Génération des Prédictions** pour la soumission sur Kaggle.

L'évaluation de la performance se base sur le **coefficient de détermination (R²)**.

-----

## 2\. Analyse Exploratoire des Données (EDA)

La première étape a été de comprendre la structure des 85 500 observations d'entraînement.

### 2.1. Structure des données

*   **`train_data.csv`**: Contient les features et la variable cible `popularity`.
*   **`test_data.csv`**: Contient les mêmes features, sans la cible.
*   **Variables**: Un mélange de variables numériques continues (`duration_ms`, `danceability`, `energy`, etc.), discrètes (`key`, `time_signature`) et catégorielles (`track_genre`).

### 2.2. Analyse de la Cible (`popularity`)

La distribution de la popularité est asymétrique, avec une concentration de titres peu populaires (pic autour de 30-40) et une fréquence notable de scores à 0. Cela suggère que prédire les "hits" sera difficile.

_(Le script génère le graphique `popularity_distribution.png` pour visualiser cela)._

### 2.3. Analyse des Features

  * **Données Numériques :** La plupart des features audio (`danceability`, `energy`, etc.) sont normalisées entre 0 et 1. Cependant, `duration_ms`, `speechiness` et `liveness` sont très asymétriques (skewed), avec de longues traînes de valeurs extrêmes.
  * **Données Catégorielles :** Nous avons des variables binaires (`explicit`, `mode`), des variables discrètes (`time_signature`), une variable catégorielle à haute cardinalité (`track_genre` - 114 genres uniques) et une variable cyclique (`key`, représentant les 12 notes de musique).

### 2.4. Corrélations (Hypothèse Initiale)

La matrice de corrélation a révélé une **information cruciale** : les corrélations linéaires directes entre les features et la `popularity` sont **extrêmement faibles** (max |r| = 0.094).

*   **Corrélation positive notable**: `energy` et `loudness`. C'est logique, un titre plus "fort" est souvent perçu comme plus énergique.
*   **Corrélation négative notable**: `acousticness` et `energy`/`loudness`. Un titre acoustique est généralement moins intense.
*   **Corrélation avec `popularity`**: Les corrélations directes avec la popularité sont faibles. `instrumentalness` a la corrélation négative la plus marquée, suggérant que les titres sans paroles sont en moyenne moins populaires. `loudness` et `energy` ont de légères corrélations positives.

**Conclusion de l'EDA :** Un modèle linéaire simple ne fonctionnera pas. Le succès résidera dans la capture de relations non-linéaires et d'interactions complexes entre les features.

_(Le script génère le graphique `correlation_matrix.png` pour visualiser cela)._

-----

## 3\. Itération 1 : Modèle de Baseline (Régression Linéaire Simple)

Pour établir un score de référence, un pipeline simple a été créé :

1.  **Numérique :** `StandardScaler` (centrage-réduction).
2.  **Catégoriel :** `OneHotEncoder` (pour `track_genre`, `mode`, `explicit`, etc.).
3.  **Modèle :** Régression Linéaire simple.

  * **Problème :** Ce modèle traite `key` (0-11) comme une variable numérique linéaire (ce qui est incorrect) et crée plus de 100 features pour les genres, ce qui le rend instable.
  * **Résultat (Baseline) :** R² ≈ 0.152. C'est notre point de départ.

-----

## 4\. Itération 2 : Amélioration Linéaire (Ridge Polynomial)

**Hypothèse :** Les interactions entre les features (ex: `energy` \* `loudness`) sont importantes, mais le modèle linéaire de base ne peut pas les capturer.

### 4.1. Prétraitement des Données et Feature Engineering (Spécifique au linéaire)

Pour que les algorithmes puissent traiter les données correctement, un pipeline de prétraitement a été mis en place.

#### 4.1.1. Identification des types de variables

Les variables ont été séparées en deux groupes pour un traitement adapté :

*   **Variables Numériques**: Celles sur lesquelles des opérations mathématiques ont un sens (ex: `danceability`, `tempo`).
*   **Variables Catégorieles**: Celles qui représentent des catégories distinctes.
    *   `track_genre` est une catégorie textuelle.
    *   `key`, `mode`, `explicit`, `time_signature` sont également traitées comme catégorielles. Même si elles sont encodées numériquement, il n'y a pas de relation d'ordre intrinsèque (par exemple, une `key` de 4 n'est pas "deux fois plus" qu'une `key` de 2).

#### 4.1.2. Pipeline de transformation

Un `ColumnTransformer` de Scikit-learn a été utilisé pour créer un pipeline robuste :

1.  **Suppression de la colonne `Unnamed: 0`**: Cette colonne, un artefact d'indexation du fichier CSV, a été retirée des features pour ne pas introduire de bruit inutile.
2.  **Encodage Cyclique de `key` :** La `key` (tonalité) est une variable cyclique. Nous la transformons en deux dimensions, `key_sin` et `key_cos`, pour que le modèle comprenne que la note 11 est aussi proche de 0 que de 10.
    *   `key_sin = sin(2 * pi * key / 12)`
    *   `key_cos = cos(2 * pi * key / 12)`
3.  **Features Polynomiales (degré 2) :** Nous ajoutons `PolynomialFeatures` au pipeline pour créer automatiquement des interactions (ex: `danceability²`, `energy * loudness`).

Ce pipeline est appliqué de manière identique sur les données d'entraînement et de test pour garantir la cohérence.

### 4.2. Modélisation (Régularisation)

L'ajout de features polynomiales crée un risque de sur-apprentissage. Pour contrer cela, nous remplaçons la Régression Linéaire par `RidgeCV`. Ce modèle applique une régularisation L2 et utilise la validation croisée pour trouver automatiquement le meilleur paramètre de régularisation `alpha`.

  * **Résultat (`polynomial_ridge`) :** **R² = 0.264**.
  * **Conclusion :** Une amélioration de +73% par rapport à la baseline. Le modèle est très stable, sans sur-apprentissage (écart train-test de 0.004).

-----

## 5\. Itération 3 : Modèles Non-Linéaires (Random Forest)

**Hypothèse :** L'EDA a montré que les relations sont fondamentalement non-linéaires. Les modèles ensemblistes (basés sur les arbres) devraient surpasser les modèles linéaires, même améliorés.

### 5.1. Modélisation (Random Forest)

Nous testons un `RandomForestRegressor`. Ce modèle peut capturer les interactions non-linéaires nativement, sans nécessiter de features polynomiales. Nous utilisons le pipeline de prétraitement simple (StandardScaler + OneHotEncoder) mais en conservant l'encodage cyclique de `key`.

### 5.2. Optimisation d'Hyperparamètres

Un premier test avec des paramètres par défaut est très prometteur. Nous lançons une optimisation `RandomizedSearchCV` (50 itérations, 5-fold CV) pour trouver la meilleure configuration.

**Configuration optimale trouvée :**

```python
RandomForestRegressor(
    n_estimators=500,       # Plus d'arbres pour stabiliser
    max_depth=None,         # Arbres profonds pour capturer la complexité
    min_samples_split=2,
    min_samples_leaf=5,     # Contrôle le sur-apprentissage
    max_features=0.7,       # Diversifie les arbres
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
```

  * **Résultat (`random_forest`) :** **R² = 0.472**.
  * **Conclusion :** C'est une **amélioration de +210%** par rapport à la baseline et +78% par rapport au modèle Ridge.

-----

## 6. Itération 4 : Feature Engineering Avancé
Hypothèse : Le score du Random Forest (R² = 0.472) est robuste, mais les distributions de données très asymétriques (vues dans l'EDA) et le manque d'interactions de domaine explicites freinent potentiellement les performances, en particulier pour les modèles de boosting (LGBM/XGBoost).

Pour améliorer tous nos modèles non-linéaires, nous intégrons deux transformations avancées directement dans notre pipeline de préparation des données.

Transformation Logarithmique (apply_log_transform=True) : Nous appliquons une transformation np.log1p aux features identifiées comme très asymétriques (duration_ms, speechiness, liveness, instrumentalness). Cela permet de normaliser leur distribution, de réduire l'impact des valeurs extrêmes et d'aider les modèles à mieux gérer ces données.

Interactions de Domaine (create_interactions=True) : Nous créons manuellement des features qui ont un sens musical et capturent des relations de domaine. Exemples :

energy_loudness (corrélation positive forte)

danceability_energy (titres énergiques et dansants)

acoustic_instrumental (musique acoustique instrumentale).

Résultat : L'ajout de ces features s'est avéré bénéfique. Elles stabilisent l'apprentissage et permettent aux modèles (Random Forest, LGBM, XGBoost) de capturer plus efficacement des signaux complexes. Ce pipeline de features avancées est donc adopté pour toutes les évaluations et entraînements finaux des modèles non-linéaires.

-----

## 7. Itération 5 : Modèles de Boosting (LightGBM & XGBoost) et Random Forest

### Nouvelle Optimisation Random Forest (RandomizedSearchCV)

Suite à une nouvelle recherche d'hyperparamètres plus approfondie avec `RandomizedSearchCV`, un nouveau modèle Random Forest a été identifié avec des performances améliorées.

**Meilleurs paramètres trouvés :**
```
RandomForestRegressor(
   n_estimators=200,
   min_samples_split=2,
   min_samples_leaf=2,
   max_features='sqrt',
   max_depth=30,
   bootstrap=False,
   random_state=42,
   n_jobs=-1
)
```
  * **Meilleur R² (CV) :** **0.5356**
  * **Justification de l'amélioration :**
    *   **`n_estimators=200`** : Un nombre d'arbres plus faible que la configuration précédente (500) mais suffisant pour capturer la complexité, tout en réduisant potentiellement le risque de sur-apprentissage et le temps d'entraînement.
    *   **`min_samples_split=2` et `min_samples_leaf=2`** : Ces valeurs plus faibles permettent aux arbres d'être plus profonds et de capturer des relations plus fines dans les données, sans pour autant sur-apprendre grâce à la régularisation implicite du Random Forest et aux autres hyperparamètres.
    *   **`max_features='sqrt'`** : En sélectionnant `sqrt(n_features)` à chaque split, le modèle introduit plus de diversité entre les arbres, ce qui réduit la variance et améliore la robustesse. La configuration précédente utilisait `0.7` des features, ce qui pouvait rendre les arbres plus corrélés.
    *   **`max_depth=30`** : Une profondeur maximale définie permet aux arbres d'explorer des relations complexes sans devenir trop spécifiques aux données d'entraînement, contrairement à `None` qui peut mener à des arbres très profonds et au sur-apprentissage.
    *   **`bootstrap=False`** : L'échantillonnage sans remplacement pour la construction des arbres peut parfois être bénéfique en réduisant la corrélation entre les arbres et en augmentant la diversité, surtout lorsque le nombre d'estimateurs est ajusté.

  * **Conclusion :** Cette nouvelle configuration du Random Forest atteint un R² de 0.5356, surpassant significativement la version précédente (0.472) et se rapprochant des performances des modèles de boosting (XGBoost à 0.5381). Cela démontre l'importance d'une optimisation fine des hyperparamètres, même pour des modèles déjà performants.
Hypothèse : Armés de notre pipeline de features avancées (Itération 4), nous évaluons les algorithmes de Gradient Boosting (GBM), souvent les plus performants sur les données tabulaires.

### Boosting (RandomizedSearchCV)

Nous avons évalué deux implémentations de pointe, LightGBM et XGBoost, reconnues pour leur performance et leur gestion efficace de la régularisation. Pour ces modèles, nous avons également implémenté une recherche d'hyperparamètres (lightgbm_search, xgboost_search) afin de trouver la configuration optimale.

Nous avons évalué deux implémentations de pointe, `LightGBM` et `XGBoost`, reconnues pour leur performance et leur gestion efficace de la régularisation.

  * **Résultat (`lightgbm_search`) :** **R² = 0.5202**.
  * **Résultat (`xgboost_search`) :** **R² = 0.5381**.
  * **Conclusion :** XGBoost surpasse légèrement LightGBM et Random Forest, établissant un meilleur score.

-----

## 8\. Résultats Finaux et Sélection du Modèle

### 8.1. Tableau Comparatif (Validation Croisée 3-fold)

| Modèle | R² (test) | RMSE | Temps | Sur-apprentissage | Installation |
|---|---|---|---|---|---|
| **XGBoost (optimisé)** | **0.538** | **15.10** | ~3 min | Modéré (0.70-0.75) | ⚠️ **Optionnel** |
| LightGBM (optimisé) | 0.520 | 15.40 | ~2 min | Modéré (0.70-0.75) | ⚠️ Optionnel |
| Random Forest | 0.472 | 16.20 | ~5 min | Modéré (0.284) | ✅ Base |
| Ridge Polynomial | 0.264 | 19.14 | ~30 sec | Aucun (0.004) | ✅ Base |

**Modèle le plus performant:** XGBoost optimisé (R² = 0.538)

**Soumissions générées:**

*   `submission_xgboost_search.csv` - R² = 0.538 (le plus performant)
*   `submission_lightgbm_search.csv` - R² = 0.520
*   `submission_random_forest.csv` - R² = 0.472
*   `submission_polynomial_ridge.csv` - R² = 0.264

### 8.2. Optimisation XGBoost

Les hyperparamètres du modèle XGBoost ont été optimisés par RandomizedSearchCV (100 itérations, 5-fold CV):

**Hyperparamètres optimaux:**

```python
XGBRegressor(
    n_estimators=500,
    max_depth=12,
    learning_rate=0.1,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_lambda=2.0,
    reg_alpha=0,
    min_child_weight=3,
    gamma=0,
    random_state=42,
    n_jobs=-1
)
```

**Amélioration:** +254% vs baseline (0.152 → 0.538)

**Note:** Sur-apprentissage modéré détecté (R² train = 0.70-0.75). Cependant, les performances en validation croisée restent excellentes.

### 8.3. Optimisation LightGBM

Les hyperparamètres du modèle LightGBM ont été optimisés par RandomizedSearchCV (100 itérations, 5-fold CV):

**Hyperparamètres optimaux:**

```python
LGBMRegressor(
    n_estimators=500,
    num_leaves=127,
    max_depth=12,
    learning_rate=0.2,
    subsample=0.6,
    colsample_bytree=0.8,
    reg_lambda=1.5,
    reg_alpha=0.01,
    random_state=42,
    n_jobs=-1
)
```

**Amélioration:** +242% vs baseline (0.152 → 0.520)

**Note:** Sur-apprentissage modéré détecté (R² train = 0.70-0.75). Cependant, les performances en validation croisée restent excellentes.

### 8.3. Choix Techniques

#### Validation Croisée 3-fold

**Justification:**

*   Dataset large (85,500 observations)
*   40% plus rapide que 5-fold
*   Précision suffisante pour évaluation

#### XGBoost avec 500 arbres

**Justification:**

*   Trouvé par optimisation exhaustive (100 itérations)
*   Meilleure performance (R² = 0.538)
*   Compromis temps/performance acceptable (~3 min)

### 8.4. Recommandations

**Pour maximiser le score Kaggle:**

*   Utiliser XGBoost optimisé (R² = 0.538)
*   Soumettre `submission_xgboost_search.csv`
*   Surveiller le sur-apprentissage (score public vs privé)

**Pour améliorer davantage (optionnel):**

*   Implémenter ensemble methods (stacking/blending)
*   Analyser les erreurs de prédiction

-----

## 9\. Génération des Fichiers de Soumission et Prochaines Étapes

Pour chaque modèle entraîné, les prédictions sont faites sur le jeu de test prétraité. Les résultats sont ensuite formatés dans un fichier CSV avec les colonnes `row_id` et `popularity`, prêts à être soumis sur Kaggle.

Une étape de **clipping** a été ajoutée pour s'assurer que toutes les prédictions se situent bien dans l'intervalle `[0, 100]`, comme l'exige la définition de la popularité.

-----

## 10\. Conclusion Générale

Ce projet a démontré l'importance d'une approche itérative :

1.  L'**EDA** a été fondamentale, en identifiant la nature non-linéaire du problème.
2.  Le **Feature Engineering** ciblé (encodage cyclique) a été plus impactant que l'ingénierie "en force" (polynomiale).
3.  L'**Optimisation d'Hyperparamètres** a été cruciale, transformant un bon modèle (Random Forest par défaut) en un excellent modèle (+210% d'amélioration vs baseline).

Le modèle final `XGBoost` (R²=0.538) représente le meilleur compromis entre performance et robustesse pour ce challenge.

-----

## Annexes

### A. Commandes Utiles

```bash
# Installer les dépendances de base
pip install -r requirements.txt

# (Optionnel) Installer les modèles avancés
python -m pip install lightgbm xgboost

# Évaluer tous les modèles installés
python evaluate_final.py

# Entraîner le modèle le plus performant (Random Forest)
python train_final.py --model random_forest

# Entraîner le modèle linéaire
python train_final.py --model polynomial_ridge

# Lancer une recherche d'hyperparamètres pour LightGBM
python train_final.py --model lightgbm_search

# Lancer une recherche d'hyperparamètres pour XGBoost
python train_final.py --model xgboost_search
```

### B. Structure des Fichiers de Soumission

Les fichiers de soumission suivent le format requis par Kaggle :

```
row_id,popularity
85500,30.25
85501,38.38
...```

Chaque ligne contient l'identifiant de la ligne (`row_id`) et la prédiction de popularité (`popularity`).
