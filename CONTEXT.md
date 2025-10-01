# Contexte du Projet : Prédiction de Popularité de Titres Spotify

## 1. Objectif Principal du Challenge

L'objectif est de développer un modèle d'apprentissage supervisé capable de prédire la **popularité** d'un titre musical sur Spotify (une note de 0 à 100) en se basant sur ses caractéristiques audio et ses métadonnées.

---

## 2. Contexte Académique (M2 Apprentissage Supervisé)

Ce challenge s'inscrit dans le cadre d'un module de M2. L'évaluation portera sur la capacité à mettre en œuvre une démarche rigoureuse d'apprentissage supervisé.

### Livrables Attendus
- Une présentation finale.
- Un rapport écrit expliquant les choix méthodologiques (prétraitement, sélection de variables, choix des modèles, optimisation d'hyperparamètres, etc.).

### Objectifs Pédagogiques
- Maîtriser la mise en place d'une démarche complète : apprentissage, validation, comparaison de méthodes.
- Comprendre les fondements théoriques : fonction de coût, risque, compromis biais/variance.
- Appliquer des techniques de régression linéaires et non-linéaires (Forêts Aléatoires, Gradient Boosting, etc.).

---

## 3. Spécifications du Challenge (Type Kaggle)

### Métrique d'Évaluation
- Le score des prédictions est évalué à l'aide du **coefficient de détermination (R²)**.

### Format de Soumission
- Le fichier de soumission doit être un `.csv` avec deux colonnes : `row_id` et `popularity`.
- Il doit contenir une prédiction de popularité pour chaque `row_id` du jeu de test.
- **Exemple de format :**
  ```csv
  row_id,popularity
  85500,45.5
  85501,67.1
  85502,42
  ```

---

## 4. Description des Données

### Fichiers Fournis
- `train_data.csv` : Données d'entraînement avec la variable cible `popularity`.
- `test_data.csv` : Données de test, sans la variable `popularity`.
- `naive_submission.csv` : Un exemple de fichier de soumission au format attendu.

### Dictionnaire des Variables

| Variable         | Type          | Description                                                                                                |
|------------------|---------------|------------------------------------------------------------------------------------------------------------|
| **popularity**   | **Numérique** | **(Cible)** Score de popularité du titre (0-100). Plus le score est élevé, plus le titre est populaire.      |
| `row_id`         | Numérique     | Identifiant unique de la ligne. À utiliser pour la soumission, mais pas comme feature d'entraînement.        |
| `duration_ms`    | Numérique     | Durée du titre en millisecondes.                                                                           |
| `explicit`       | Binaire (0/1) | Indique si le titre contient des paroles explicites (1) ou non (0).                                        |
| `danceability`   | Numérique     | Mesure de la facilité à danser sur le titre (0.0-1.0).                                                      |
| `energy`         | Numérique     | Mesure de l'intensité et de l'activité du titre (0.0-1.0).                                                 |
| `key`            | Catégorielle  | Tonalité du titre (0=Do, 1=Do#, ..., 11=Si). **Variable cyclique.**                                         |
| `loudness`       | Numérique     | Niveau sonore moyen en décibels (dB), généralement entre -60 et 0.                                         |
| `mode`           | Binaire (0/1) | Indique si le titre est en mode majeur (1) ou mineur (0).                                                  |
| `speechiness`    | Numérique     | Présence de paroles. Proche de 1.0 indique un titre très parlé (podcast, rap).                             |
| `acousticness`   | Numérique     | Probabilité que le titre soit acoustique (0.0-1.0).                                                        |
| `instrumentalness`| Numérique    | Absence de voix. Proche de 1.0 indique un titre instrumental.                                              |
| `liveness`       | Numérique     | Probabilité que le titre ait été enregistré en public (0.0-1.0).                                           |
| `valence`        | Numérique     | Mesure de la positivité musicale. Proche de 1.0 indique un titre joyeux.                                   |
| `tempo`          | Numérique     | Tempo du titre en battements par minute (BPM).                                                             |
| `time_signature` | Numérique     | Signature rythmique du titre (ex: 4 pour 4/4).                                                             |
| `track_genre`    | Catégorielle  | Genre musical principal du titre.                                                                          |