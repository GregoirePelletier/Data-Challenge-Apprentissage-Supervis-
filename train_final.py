"""
Script d'entraînement final pour le challenge Spotify Popularity Prediction.
Utilise le Target Encoding pour les genres.

Modèles disponibles:
- polynomial_ridge: Ridge avec features polynomiales (R² = 0.264)
- random_forest: Random Forest optimisé (R² = 0.472)
- random_forest_search: Random Forest avec RandomizedSearchCV
- lightgbm: LightGBM (rapide et efficace)
- xgboost: XGBoost (régularisation L1/L2)
- lightgbm_search: LightGBM avec RandomizedSearchCV
- xgboost_search: XGBoost avec RandomizedSearchCV
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, make_scorer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import warnings
import os # Ajout pour vérifier le dossier results
warnings.filterwarnings('ignore')

# Import des modules personnalisés
from src.data_preparation import (
    load_data,
    initial_feature_engineering,
    get_features_and_target,
    get_test_features,
    get_feature_names
)
from src.pipelines import create_simple_preprocessor, create_polynomial_preprocessor

# Définition du scorer R² pour RandomizedSearchCV
r2_scorer = make_scorer(r2_score)

param_grid_lightgbm = {
    'n_estimators': [100, 300, 500, 700],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [5, 7, 9, 12, 15],
    'num_leaves': [15, 31, 63, 127],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.01, 0.1, 0.5],
    'reg_lambda': [0.5, 1, 1.5]
}

# Nouvelle grille pour XGBoost, basée sur les résultats de LightGBM
param_grid_xgboost = {
    # Paramètres communs centrés sur les meilleurs scores de LGBM
    'n_estimators': [400, 500, 600, 700],
    'learning_rate': [0.1, 0.15, 0.2, 0.25],
    'max_depth': [10, 12, 14, 16], # 'max_depth' 12 était optimal pour LGBM
    'subsample': [0.5, 0.6, 0.7],          # Optimal LGBM = 0.6
    'colsample_bytree': [0.7, 0.8, 0.9],  # Optimal LGBM = 0.8
    'reg_alpha': [0, 0.01, 0.05, 0.1],    # Optimal LGBM = 0.01
    'reg_lambda': [1.0, 1.5, 2.0, 3.0],   # Optimal LGBM = 1.5
    
    # Paramètres spécifiques à XGBoost (sans équivalent LGBM)
    'gamma': [0, 0.1, 0.25, 0.5],          # 'min_split_gain' dans LGBM
    'min_child_weight': [1, 3, 5, 7]       # 'min_child_samples' dans LGBM
}

param_grid_random_forest = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': [0.5, 0.7, 1.0, 'sqrt'],
    'bootstrap': [True, False]
}
# --- FIN DES CORRECTIONS DE GRILLES ---


def get_model_and_preprocessor(model_name: str, numeric_features: list, ohe_features: list, target_encode_features: list):
    """Retourne le modèle et le préprocesseur appropriés selon le nom du modèle."""
    
    if model_name == "polynomial_ridge":
        print("Modèle: Ridge Polynomial (degree=2)")
        print("R² attendu: 0.264")
        model = RidgeCV(alphas=np.logspace(-2, 2, 10))
        
        all_categorical_features = ohe_features + target_encode_features
        preprocessor = create_polynomial_preprocessor(numeric_features, all_categorical_features)
        
    elif model_name == "random_forest":
        print("Modèle: Random Forest (optimisé)")
        print("R² attendu: ~0.472 (va changer avec Target Enc.)")
        model = RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=5,
            max_features=0.7,
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        preprocessor = create_simple_preprocessor(numeric_features, ohe_features, target_encode_features)
        
    elif model_name == "random_forest_search":
        print("Modèle: Random Forest (Recherche Hyperparamètres)")
        print("Configuration: RandomizedSearchCV (n_iter=100, cv=5, scoring=R²)")
        
        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        model = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid_random_forest, # Grille SANS préfixe
            n_iter=100,
            scoring=r2_scorer,
            cv=5,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        preprocessor = create_simple_preprocessor(numeric_features, ohe_features, target_encode_features)
    
    elif model_name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            raise ImportError(
                "LightGBM n'est pas installé. Installez-le avec: pip install lightgbm"
            )
        
        print("Modèle: LightGBM")
        print("Configuration: n_estimators=300, learning_rate=0.05, max_depth=7")
        model = LGBMRegressor(
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
        preprocessor = create_simple_preprocessor(numeric_features, ohe_features, target_encode_features)

    elif model_name == "lightgbm_search":
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            raise ImportError(
                "LightGBM n'est pas installé. Installez-le avec: pip install lightgbm"
            )
        
        print("Modèle: LightGBM (Recherche Hyperparamètres)")
        print("Configuration: RandomizedSearchCV (n_iter=200, cv=5, scoring=R²)")
        
        base_model = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
        model = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid_lightgbm, # Grille SANS préfixe
            n_iter=200, 
            scoring=r2_scorer,
            cv=5, 
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        preprocessor = create_simple_preprocessor(numeric_features, ohe_features, target_encode_features)

    elif model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError(
                "XGBoost n'est pas installé. Installez-le avec: pip install xgboost"
            )
        
        print("Modèle: XGBoost")
        print("Configuration: n_estimators=300, learning_rate=0.05, max_depth=6")
        model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        preprocessor = create_simple_preprocessor(numeric_features, ohe_features, target_encode_features)

    elif model_name == "xgboost_search":
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError(
                "XGBoost n'est pas installé. Installez-le avec: pip install xgboost"
            )
        
        print("Modèle: XGBoost (Recherche Hyperparamètres)")
        print("Configuration: RandomizedSearchCV (n_iter=100, cv=5, scoring=R²)")
        
        base_model = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
        model = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid_xgboost, # Grille SANS préfixe
            n_iter=100, 
            scoring=r2_scorer,
            cv=5, 
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        preprocessor = create_simple_preprocessor(numeric_features, ohe_features, target_encode_features)
        
    else:
        raise ValueError(
            f"Modèle '{model_name}' non reconnu.\n"
            "Modèles valides: polynomial_ridge, random_forest, random_forest_search, "
            "lightgbm, xgboost, lightgbm_search, xgboost_search"
        )
    
    return model, preprocessor


def main(model_name):
    """
    Fonction principale pour entraîner un modèle et générer la soumission.
    """
    print("="*80)
    print(f"ENTRAÎNEMENT - {model_name.upper()}")
    print("="*80)
    print()

    # 1. Chargement et préparation des données
    print("1. Chargement des données...")
    
    train_df, test_df = load_data()
    
    # Utilisation des features AVANCÉES
    train_df = initial_feature_engineering(train_df, apply_log_transform=True, create_interactions=True)
    test_df = initial_feature_engineering(test_df, apply_log_transform=True, create_interactions=True)
    
    X_train, y_train = get_features_and_target(train_df)
    X_test, test_row_ids = get_test_features(test_df)
    
    # Récupérer les 3 listes de features
    numeric_features, ohe_features, target_encode_features = get_feature_names(
        include_log_features=True, 
        include_interactions=True
    )
    
    print(f"   Observations train: {len(X_train):,}")
    print(f"   Observations test: {len(X_test):,}")
    print(f"   Features: {X_train.shape[1]}")
    print()

    # 2. Obtenir le modèle et le préprocesseur
    print("2. Configuration du modèle...")
    regressor, preprocessor = get_model_and_preprocessor(
        model_name, numeric_features, ohe_features, target_encode_features
    )
    print()

    # 3. Créer le pipeline complet
    print("3. Création du pipeline...")
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor) # Nom de l'étape est 'regressor'
    ])
    print("   ✓ Pipeline créé")
    print()

    # 4. Entraîner le modèle
    print("4. Entraînement du modèle...")
    print("   (Cela peut prendre quelques minutes...)")
    
    # .fit(X, y) gère tout, y compris le TargetEncoder
    full_pipeline.fit(X_train, y_train)

    print("   ✓ Entraînement terminé")
    print()

    # Afficher les hyperparamètres optimaux si RandomizedSearchCV a été utilisé
    if 'search' in model_name:
        # Accéder à l'étape 'regressor' (qui est le RandomizedSearchCV)
        best_r2 = full_pipeline.named_steps['regressor'].best_score_
        best_params = full_pipeline.named_steps['regressor'].best_params_
        
        print("   ✓ Résultats de la recherche (RandomizedSearchCV):")
        print(f"   Meilleur R² (CV): {best_r2:.4f}")
        print("   Meilleurs paramètres:")
        for k, v in best_params.items():
            # Les clés sont maintenant 'n_estimators', etc. SANS préfixe
            print(f"     - {k}: {v}")
        print()
        
        if not os.path.exists('results'):
            os.makedirs('results')
            print("   ✓ Dossier 'results' créé.")
            
        model_filename = f"results/best_model_{model_name}.joblib"
        joblib.dump(full_pipeline.named_steps['regressor'].best_estimator_, model_filename)
        print(f"   ✓ Meilleur modèle enregistré sous '{model_filename}'")
        print()

    # Afficher l'alpha optimal pour Ridge
    if model_name == "polynomial_ridge":
        best_alpha = full_pipeline.named_steps['regressor'].alpha_
        print(f"   Alpha optimal (RidgeCV): {best_alpha:.4f}")
        print()

    # 5. Faire les prédictions
    print("5. Génération des prédictions...")
    predictions = full_pipeline.predict(X_test)
    predictions = np.clip(predictions, 0, 100)
    print(f"   Prédictions générées: {len(predictions):,}")
    print(f"   Min: {predictions.min():.2f}, Max: {predictions.max():.2f}, Moyenne: {predictions.mean():.2f}")
    print()

    # 6. Créer le fichier de soumission
    print("6. Création du fichier de soumission...")
    submission_df = pd.DataFrame({
        'row_id': test_row_ids,
        'popularity': predictions
    })
    filename = f"submission_{model_name}.csv"
    submission_df.to_csv(filename, index=False)
    
    print(f"   ✓ Fichier '{filename}' créé avec succès")
    print()
    print("Aperçu des prédictions:")
    print(submission_df.head(10))
    print()
    
    print("="*80)
    print("ENTRAÎNEMENT TERMINÉ")
    print("="*80)
    print()
    print(f"Fichier de soumission: {filename}")
    print("Vous pouvez maintenant soumettre ce fichier sur Kaggle.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entraîner un modèle pour la prédiction de popularité Spotify.",
        epilog="Exemple: python train_final.py --model random_forest"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        choices=["polynomial_ridge", "random_forest", "random_forest_search", "lightgbm", "xgboost", "lightgbm_search", "xgboost_search"],
        help="Le modèle à entraîner. Les options '_search' lancent une recherche d'hyperparamètres."
    )
    args = parser.parse_args()
    main(args.model)