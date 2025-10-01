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

Plusieurs modèles ont été sélectionnés pour couvrir différentes approches (linéaires et non-linéaires) et comparer leurs performances. Pour chaque modèle, un fichier de soumission distinct est généré.

### 4.1. Régression Linéaire (`Linear Regression`)

-   **Description**: C'est le modèle de base. Il cherche à trouver la meilleure relation linéaire entre les features et la cible.
-   **Choix**: Sert de baseline. Si des modèles plus complexes ne font pas significativement mieux, cela peut indiquer que la relation est principalement linéaire ou que les features ne sont pas assez informatives.
-   **Fichier de soumission**: `submission_linear_regression.csv`

### 4.2. Régression Ridge (`Ridge Regression`)

-   **Description**: Une variante de la régression linéaire qui inclut une régularisation L2. Cela pénalise les coefficients de régression trop grands, ce qui aide à prévenir le sur-apprentissage (overfitting), surtout quand de nombreuses features sont corrélées (ce qui est le cas après le One-Hot Encoding).
-   **Choix**: C'est une amélioration simple mais souvent efficace de la régression linéaire, la rendant plus robuste.
-   **Fichier de soumission**: `submission_ridge_regression.csv`

### 4.3. Forêt Aléatoire (`Random Forest Regressor`)

-   **Description**: Un modèle d'ensemble basé sur les arbres de décision. Il construit de nombreux arbres sur des sous-ensembles de données et de features, puis moyenne leurs prédictions.
-   **Choix**: Les forêts aléatoires sont très robustes, gèrent bien les interactions complexes et non-linéaires entre les features, et sont moins sujettes à l'overfitting qu'un unique arbre de décision. C'est un modèle puissant et polyvalent. Des hyperparamètres de base (`max_depth=15`, `min_samples_leaf=5`) ont été choisis pour limiter la complexité et éviter un sur-apprentissage trop important.
-   **Fichier de soumission**: `submission_random_forest.csv`

### 4.4. Gradient Boosting (`Gradient Boosting Regressor`)

-   **Description**: Un autre modèle d'ensemble qui construit des arbres de manière séquentielle. Chaque nouvel arbre tente de corriger les erreurs de l'arbre précédent.
-   **Choix**: Le Gradient Boosting est souvent l'un des modèles les plus performants sur les données tabulaires. Il peut capturer des dépendances très fines. Les hyperparamètres (`max_depth=5`, `learning_rate=0.1`) sont des valeurs de départ communes qui offrent un bon compromis entre performance et temps de calcul.
-   **Fichier de soumission**: `submission_gradient_boosting.csv`

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

### Prochaines étapes possibles :

1.  **Optimisation des hyperparamètres**: Utiliser des techniques comme `GridSearchCV` ou `RandomizedSearchCV` pour trouver les meilleurs hyperparamètres pour les modèles les plus prometteurs (Random Forest et Gradient Boosting).
2.  **Feature Engineering plus avancé**:
    -   Créer des features d'interaction (ex: `danceability * energy`).
    -   Analyser plus en profondeur l'impact de `track_genre` (regrouper les genres rares, par exemple).
3.  **Validation croisée**: Évaluer la performance des modèles de manière plus robuste en utilisant la validation croisée sur le jeu d'entraînement, au lieu de se fier uniquement au score public de Kaggle.
4.  **Essayer d'autres modèles**: Des modèles comme XGBoost, LightGBM ou CatBoost, qui sont des implémentations optimisées du Gradient Boosting, donnent souvent d'excellents résultats.
