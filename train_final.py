"""
Script d'entraînement final pour le challenge Spotify Popularity Prediction.

Modèles disponibles:
- polynomial_ridge: Ridge avec features polynomiales (R² = 0.264)
- random_forest: Random Forest optimisé (R² = 0.472)
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

# Grilles de paramètres pour RandomizedSearchCV
# Complétées avec des plages de valeurs courantes pour la recherche
param_grid_xgboost = {
    'regressor__n_estimators': [100, 300, 500, 700],
    'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'regressor__max_depth': [3, 5, 7, 9],
    'regressor__subsample': [0.6, 0.8, 1.0],
    'regressor__colsample_bytree': [0.6, 0.8, 1.0],
    'regressor__gamma': [0, 0.1, 0.2],
    'regressor__reg_alpha': [0, 0.01, 0.1, 0.5],
    'regressor__reg_lambda': [0.5, 1, 1.5]
}

param_grid_lightgbm = {
    'regressor__n_estimators': [100, 300, 500, 700],
    'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'regressor__max_depth': [5, 7, 9, 12],
    'regressor__num_leaves': [15, 31, 63, 127],
    'regressor__subsample': [0.6, 0.8, 1.0],
    'regressor__colsample_bytree': [0.6, 0.8, 1.0],
    'regressor__reg_alpha': [0, 0.01, 0.1, 0.5],
    'regressor__reg_lambda': [0.5, 1, 1.5]
}

def get_model_and_preprocessor(model_name, numeric_features, categorical_features):
    """
    Retourne le modèle et le pipeline de prétraitement approprié.
    """
    
    if model_name == "polynomial_ridge":
        print("Modèle: Ridge Polynomial (degree=2)")
        print("R² attendu: 0.264")
        model = RidgeCV(alphas=np.logspace(-2, 2, 10))
        preprocessor = create_polynomial_preprocessor(numeric_features, categorical_features)
        
    elif model_name == "random_forest":
        print("Modèle: Random Forest (optimisé)")
        print("R² attendu: 0.472")
        print("Configuration: n_estimators=500, max_depth=None, max_features=0.7")
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
        preprocessor = create_simple_preprocessor(numeric_features, categorical_features)
        
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
        preprocessor = create_simple_preprocessor(numeric_features, categorical_features)

    elif model_name == "lightgbm_search":
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            raise ImportError(
                "LightGBM n'est pas installé. Installez-le avec: pip install lightgbm"
            )
        
        print("Modèle: LightGBM (Recherche Hyperparamètres)")
        print("Configuration: RandomizedSearchCV (n_iter=10, cv=3, scoring=R²)")
        
        base_model = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
        model = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid_lightgbm,
            n_iter=10,
            scoring=r2_scorer,
            cv=3,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        preprocessor = create_simple_preprocessor(numeric_features, categorical_features)
        
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
        preprocessor = create_simple_preprocessor(numeric_features, categorical_features)

    elif model_name == "xgboost_search":
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError(
                "XGBoost n'est pas installé. Installez-le avec: pip install xgboost"
            )
        
        print("Modèle: XGBoost (Recherche Hyperparamètres)")
        print("Configuration: RandomizedSearchCV (n_iter=10, cv=3, scoring=R²)")
        
        base_model = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
        model = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid_xgboost,
            n_iter=10,
            scoring=r2_scorer,
            cv=3,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        preprocessor = create_simple_preprocessor(numeric_features, categorical_features)
        
    else:
        raise ValueError(
            f"Modèle '{model_name}' non reconnu.\n"
            f"Modèles valides: polynomial_ridge, random_forest, lightgbm, xgboost, lightgbm_search, xgboost_search"
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
    
    train_df = initial_feature_engineering(train_df)
    test_df = initial_feature_engineering(test_df)
    
    X_train, y_train = get_features_and_target(train_df)
    X_test, test_row_ids = get_test_features(test_df)
    
    numeric_features, categorical_features = get_feature_names()
    
    print(f"   Observations train: {len(X_train):,}")
    print(f"   Observations test: {len(X_test):,}")
    print(f"   Features: {X_train.shape[1]}")
    print()

    # 2. Obtenir le modèle et le préprocesseur
    print("2. Configuration du modèle...")
    regressor, preprocessor = get_model_and_preprocessor(
        model_name, numeric_features, categorical_features
    )
    print()

    # 3. Créer le pipeline complet
    print("3. Création du pipeline...")
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    print("   ✓ Pipeline créé")
    print()

    # 4. Entraîner le modèle
    print("4. Entraînement du modèle...")
    print("   (Cela peut prendre quelques minutes...)")
    full_pipeline.fit(X_train, y_train)
    print("   ✓ Entraînement terminé")
    print()

    # Afficher les hyperparamètres optimaux si RandomizedSearchCV a été utilisé
    if 'search' in model_name:
        best_r2 = full_pipeline.best_score_
        best_params = full_pipeline.best_params_
        print("   ✓ Résultats de la recherche (RandomizedSearchCV):")
        print(f"   Meilleur R² (CV): {best_r2:.4f}")
        print("   Meilleurs paramètres:")
        for k, v in best_params.items():
            print(f"     - {k}: {v}")
        print()
        
        # Enregistrer le meilleur modèle trouvé par RandomizedSearchCV
        model_filename = f"results/best_model_{model_name}.joblib"
        joblib.dump(full_pipeline.best_estimator_, model_filename)
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
        choices=["polynomial_ridge", "random_forest", "lightgbm", "xgboost", "lightgbm_search", "xgboost_search"],
        help="Le modèle à entraîner. Les options '_search' lancent une recherche d'hyperparamètres."
    )
    args = parser.parse_args()
    main(args.model)