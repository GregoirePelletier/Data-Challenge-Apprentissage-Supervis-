# Spotify Popularity Prediction Project

Ce projet a pour but de prédire la popularité d'un titre musical Spotify en utilisant des techniques d'apprentissage supervisé.

---

## Structure du Projet

Le projet a été restructuré pour suivre les bonnes pratiques de développement, en séparant les différentes logiques en modules distincts.

-   `CONTEXT.md`: Fichier de contexte décrivant les objectifs et les données du challenge.
-   `rapport.md`: Rapport d'analyse détaillant la démarche, les expérimentations et les résultats.
-   `requirements.txt`: Liste des dépendances Python nécessaires pour exécuter le projet.
-   `run_all_models.ps1`: Script PowerShell pour lancer l'entraînement de tous les modèles définis.
-   `train.py`: Script principal pour entraîner un modèle spécifique.
-   `src/`: Répertoire contenant le code source modulaire.
    -   `data_preparation.py`: Fonctions pour charger, nettoyer et préparer les données.
    -   `pipelines.py`: Fonctions pour construire les pipelines de prétraitement des données.

---

## Installation

Pour utiliser ce projet, suivez les étapes ci-dessous.

### 1. Créer un Environnement Virtuel

Il est fortement recommandé d'utiliser un environnement virtuel pour isoler les dépendances du projet.

```powershell
# Crée un environnement virtuel nommé 'venv'
python -m venv venv

# Active l'environnement. Cette commande doit être lancée à chaque nouvelle session de terminal.
.\venv\Scripts\Activate.ps1
```

### 2. Installer les Dépendances

Une fois l'environnement activé, installez toutes les librairies requises à l'aide du fichier `requirements.txt`.

```powershell
pip install -r requirements.txt
```

---

## Utilisation

### Entraîner un Modèle Spécifique

Vous pouvez entraîner un seul modèle en utilisant le script `train.py` avec l'argument `--model_name`.

Les noms de modèles disponibles sont :
- `linear` (Régression Linéaire)
- `ridge` (Régression Ridge)
- `random_forest` (Forêt Aléatoire)
- `gradient_boosting` (Gradient Boosting)
- `polynomial_ridge` (Régression Ridge avec features polynomiales)

**Exemple :**
```powershell
python train.py --model_name polynomial_ridge
```

### Entraîner Tous les Modèles

Pour entraîner tous les modèles les uns après les autres et générer tous les fichiers de soumission, exécutez le script PowerShell `run_all_models.ps1`.

```powershell
.\run_all_models.ps1
```

Chaque exécution générera un fichier `submission_<nom_du_modèle>.csv` à la racine du projet, prêt à être soumis sur Kaggle.
