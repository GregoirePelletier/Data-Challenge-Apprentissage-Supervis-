# Guide d'Utilisation - Challenge Spotify Popularity Prediction

## 🚀 Démarrage Rapide

### Installation

```bash
# Installer les dépendances de base
pip install -r requirements.txt
```

**C'est tout !** Les 2 meilleurs modèles fonctionnent sans dépendances supplémentaires.

---

## 📊 Modèles Disponibles

### Modèles de Base (sans installation supplémentaire)

#### 1. Random Forest ⭐ MEILLEUR (R² = 0.472)
```bash
python train_final.py --model random_forest
```
- **Performance:** R² = 0.472 (meilleur résultat)
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

#### 3. LightGBM (optionnel)
```bash
# Installation (léger, ~1.5 MB)
python -m pip install lightgbm

# Entraînement
python train_final.py --model lightgbm
```
- **Avantages:** Très rapide, gestion efficace des catégorielles
- **Temps:** ~2 minutes

#### 4. XGBoost (optionnel)
```bash
# Installation (lourd, ~57 MB)
python -m pip install xgboost

# Entraînement
python train_final.py --model xgboost
```
- **Avantages:** Régularisation L1/L2, excellentes performances
- **Temps:** ~3 minutes

---

## 🎯 Workflow Recommandé

### Option 1: Utiliser le Meilleur Modèle (Recommandé)

```bash
# Entraîner Random Forest (meilleur modèle)
python train_final.py --model random_forest
```

**Résultat:** Fichier `submission_random_forest.csv` prêt pour Kaggle

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

### Option 3: Comparer Plusieurs Soumissions

```bash
# Générer plusieurs soumissions
python train_final.py --model polynomial_ridge
python train_final.py --model random_forest

# Si LightGBM/XGBoost installés
python train_final.py --model lightgbm
python train_final.py --model xgboost
```

**Résultat:** Plusieurs fichiers `submission_*.csv` à tester sur Kaggle

---

## 📈 Performances Attendues

| Modèle | R² (test) | RMSE | Temps | Installation |
|--------|-----------|------|-------|--------------|
| **Random Forest** | **0.472** | 16.20 | ~5 min | ✅ Base |
| Ridge Polynomial | 0.264 | 19.14 | ~30 sec | ✅ Base |
| LightGBM | ~0.45-0.48 | ~16-17 | ~2 min | ⚠️ Optionnel |
| XGBoost | ~0.45-0.48 | ~16-17 | ~3 min | ⚠️ Optionnel |

**Recommandation:** Commencer avec **Random Forest** (meilleur résultat, pas d'installation supplémentaire)

---

## 🔧 Résolution de Problèmes

### Erreur: "ModuleNotFoundError: No module named 'lightgbm'"

**Solution:**
```bash
python -m pip install lightgbm
```

Ou utilisez les modèles de base:
```bash
python train_final.py --model random_forest
```

### Erreur: "ModuleNotFoundError: No module named 'xgboost'"

**Solution:**
```bash
python -m pip install xgboost
```

Ou utilisez les modèles de base:
```bash
python train_final.py --model random_forest
```

### Erreur: "Fatal error in launcher"

**Cause:** Problème avec pip

**Solution:** Utiliser `python -m pip` au lieu de `pip`:
```bash
python -m pip install lightgbm
```

### Installation de XGBoost annulée (fichier trop lourd)

**Solution 1:** Réessayer l'installation
```bash
python -m pip install xgboost
```

**Solution 2:** Utiliser Random Forest (meilleur résultat de toute façon)
```bash
python train_final.py --model random_forest
```

---

## 📊 Comparaison des Modèles

### Random Forest vs Ridge Polynomial

| Critère | Random Forest | Ridge Polynomial |
|---------|---------------|------------------|
| **R²** | 0.472 ⭐ | 0.264 |
| **RMSE** | 16.20 | 19.14 |
| **Temps** | ~5 min | ~30 sec |
| **Sur-apprentissage** | Modéré (0.284) | Aucun (0.004) |
| **Stabilité** | Bonne | Excellente |
| **Recommandation** | **Meilleur pour Kaggle** | Bon pour baseline |

**Verdict:** Random Forest est le meilleur choix pour maximiser le score Kaggle.

### Pourquoi Random Forest est Meilleur ?

1. **Performance:** +78% de R² vs Ridge Polynomial (0.472 vs 0.264)
2. **Optimisé:** Hyperparamètres trouvés par RandomizedSearchCV (50 itérations)
3. **Robuste:** 500 arbres, profondeur illimitée
4. **Temps acceptable:** ~5 minutes d'entraînement

---

## 🎯 Cas d'Usage

### Cas 1: Je veux le meilleur score Kaggle

```bash
python train_final.py --model random_forest
```

**Résultat:** R² = 0.472 (meilleur modèle)

### Cas 2: Je veux un modèle rapide et stable

```bash
python train_final.py --model polynomial_ridge
```

**Résultat:** R² = 0.264 (stable, 30 secondes)

### Cas 3: Je veux comparer plusieurs modèles

```bash
# Évaluer d'abord
python evaluate_final.py

# Entraîner les meilleurs
python train_final.py --model random_forest
python train_final.py --model polynomial_ridge
```

### Cas 4: Je veux tester LightGBM/XGBoost

```bash
# Installer
python -m pip install lightgbm xgboost

# Évaluer
python evaluate_final.py

# Entraîner le meilleur
python train_final.py --model <meilleur_modele>
```

---

## 📁 Fichiers Générés

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

**Prêt pour soumission Kaggle !**

### Après `evaluate_final.py`

```
results/evaluation_final.csv
```

**Contenu:**
- Classement des modèles par R²
- Métriques détaillées (R², RMSE, écart train-test)
- Recommandation du meilleur modèle

---

## 🎓 Conseils

### Pour Maximiser le Score Kaggle

1. ✅ **Utiliser Random Forest** (R² = 0.472)
2. ✅ **Soumettre `submission_random_forest.csv`**
3. ⚠️ **Attention au sur-apprentissage** (surveiller le score public vs privé)

### Pour la Présentation

1. ✅ **Montrer l'évaluation comparative** (`evaluate_final.py`)
2. ✅ **Expliquer le choix de Random Forest** (optimisation, performance)
3. ✅ **Documenter les hyperparamètres** (voir `docs/rapport.md`)

### Pour Aller Plus Loin (Optionnel)

1. Tester LightGBM et XGBoost
2. Comparer avec Random Forest
3. Analyser les erreurs de prédiction
4. Tester ensemble methods (stacking/blending)

---

## 📝 Résumé

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

- **Meilleur modèle:** Random Forest (R² = 0.472)
- **Fichier de soumission:** `submission_random_forest.csv`
- **Temps total:** ~5 minutes

---

## 🎉 Conclusion

**Pour un data challenge académique, Random Forest est le meilleur choix:**

✅ **Performance:** R² = 0.472 (meilleur résultat)  
✅ **Simplicité:** Pas d'installation supplémentaire  
✅ **Rapidité:** ~5 minutes d'entraînement  
✅ **Prêt:** Fichier de soumission généré automatiquement  

**Commande unique:**
```bash
python train_final.py --model random_forest
```

**Bonne chance pour le challenge ! 🚀**

