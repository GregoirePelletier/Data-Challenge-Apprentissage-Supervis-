"""
Script d'entraînement final pour le challenge Spotify Popularity Prediction.
Inclut Target Encoding, CatBoost, et Stacking.

Modèles disponibles:
- polynomial_ridge: Ridge avec features polynomiales
- random_forest: Random Forest optimisé (R²=0.536)
- lightgbm: LightGBM optimisé (R²=0.520)
- xgboost: XGBoost optimisé (R²=0.538)
- catboost: CatBoost (Paramètres de base)
- catboost_search: NOUVEAU. Lance une recherche pour CatBoost
- stacking: Combine XGB, RF et LGBM
- (les modèles '_search' pour RF/LGBM/XGB sont aussi disponibles)
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import warnings
import os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, make_scorer

# Import des modèles de boosting
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

warnings.filterwarnings('ignore')

from src.data_preparation import (
    load_data,
    initial_feature_engineering,
    get_features_and_target,
    get_test_features,
    get_feature_names
)
from src.pipelines import (
    create_simple_preprocessor, 
    create_polynomial_preprocessor,
    create_catboost_preprocessor
)

# Définition du scorer R² pour RandomizedSearchCV
r2_scorer = make_scorer(r2_score)

# Grilles de recherche (pour les options '_search')
param_grid_lightgbm = {
    'n_estimators': [300, 500, 700], 'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [7, 9, 12, 15], 'num_leaves': [31, 63, 127],
    'subsample': [0.6, 0.8], 'colsample_bytree': [0.6, 0.8]
}
param_grid_xgboost = {
    'n_estimators': [400, 500, 600, 700], 'learning_rate': [0.1, 0.15, 0.2],
    'max_depth': [10, 12, 14, 16], 'subsample': [0.5, 0.6, 0.7],
    'colsample_bytree': [0.7, 0.8, 0.9], 'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1.0, 1.5, 2.0]
}
param_grid_random_forest = {
    'n_estimators': [200, 300, 500], 'max_depth': [None, 20, 30],
    'min_samples_split': [2, 5], 'min_samples_leaf': [2, 4, 6],
    'max_features': [0.5, 0.7, 'sqrt']
}

param_grid_catboost = {
    'depth': [6, 8, 10, 12],
    'learning_rate': [0.03, 0.05, 0.1, 0.2],
    'iterations': [1000, 1500, 2000],
    'l2_leaf_reg': [1, 3, 5, 7],
    'border_count': [32, 64, 128]
}


def get_model_and_preprocessor(model_name: str, numeric_features: list, ohe_features: list, target_encode_features: list):
    """Retourne le modèle et le préprocesseur appropriés selon le nom du modèle."""
    
    # Concatène les listes de features pour certains préprocesseurs
    all_ohe_features = ohe_features + target_encode_features
    all_cat_features_for_catboost = ohe_features + target_encode_features

    if model_name == "polynomial_ridge":
        print("Modèle: Ridge Polynomial (degree=2)")
        model = RidgeCV(alphas=np.logspace(-2, 2, 10))
        preprocessor = create_polynomial_preprocessor(numeric_features, all_ohe_features)
        
    elif model_name == "random_forest":
        print("Modèle: Random Forest (Params optimisés R²=0.536)")
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=30,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=False,
            random_state=42, n_jobs=-1, verbose=0
        )
        preprocessor = create_simple_preprocessor(numeric_features, ohe_features, target_encode_features)
        
    elif model_name == "random_forest_search":
        print("Modèle: Random Forest (Recherche Hyperparamètres)")
        base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
        model = RandomizedSearchCV(
            estimator=base_model, param_distributions=param_grid_random_forest,
            n_iter=50, scoring=r2_scorer, cv=5, 
            random_state=42, n_jobs=-1, verbose=1
        )
        preprocessor = create_simple_preprocessor(numeric_features, ohe_features, target_encode_features)
    
    elif model_name == "lightgbm":
        if LGBMRegressor is None: raise ImportError("LightGBM n'est pas installé.")
        print("Modèle: LightGBM (Params optimisés R²=0.520)")
        model = LGBMRegressor(
            n_estimators=500,
            max_depth=12,
            learning_rate=0.2,
            num_leaves=127,
            subsample=0.6,
            colsample_bytree=0.8,
            reg_alpha=0.01,
            reg_lambda=1.5,
            random_state=42, n_jobs=-1, verbose=-1
        )
        preprocessor = create_simple_preprocessor(numeric_features, ohe_features, target_encode_features)

    elif model_name == "lightgbm_search":
        if LGBMRegressor is None: raise ImportError("LightGBM n'est pas installé.")
        print("Modèle: LightGBM (Recherche Hyperparamètres)")
        base_model = LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
        model = RandomizedSearchCV(
            estimator=base_model, param_distributions=param_grid_lightgbm,
            n_iter=50, scoring=r2_scorer, cv=5, 
            random_state=42, n_jobs=-1, verbose=1
        )
        preprocessor = create_simple_preprocessor(numeric_features, ohe_features, target_encode_features)

    elif model_name == "xgboost":
        if XGBRegressor is None: raise ImportError("XGBoost n'est pas installé.")
        print("Modèle: XGBoost (Params optimisés R²=0.538)")
        model = XGBRegressor(
            n_estimators=500,
            max_depth=12,
            learning_rate=0.1,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0,
            reg_lambda=2.0,
            min_child_weight=3,
            gamma=0,
            random_state=42, n_jobs=-1, verbosity=0
        )
        preprocessor = create_simple_preprocessor(numeric_features, ohe_features, target_encode_features)

    elif model_name == "xgboost_search":
        if XGBRegressor is None: raise ImportError("XGBoost n'est pas installé.")
        print("Modèle: XGBoost (Recherche Hyperparamètres)")
        base_model = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
        model = RandomizedSearchCV(
            estimator=base_model, param_distributions=param_grid_xgboost,
            n_iter=50, scoring=r2_scorer, cv=5, 
            random_state=42, n_jobs=-1, verbose=1
        )
        preprocessor = create_simple_preprocessor(numeric_features, ohe_features, target_encode_features)

    # Cas pour CatBoost
    elif model_name == "catboost":
        if CatBoostRegressor is None: raise ImportError("CatBoost n'est pas installé.")
        print("Modèle: CatBoost (Gestion native des catégorielles)")
        
        # Calculer les INDICES des features catégorielles
        cat_features_indices = list(range(
            len(numeric_features), 
            len(numeric_features) + len(all_cat_features_for_catboost)
        ))
        
        model = CatBoostRegressor(
            iterations=1500,
            learning_rate=0.05,
            depth=10,
            l2_leaf_reg=3,
            loss_function='RMSE',
            eval_metric='R2',
            random_seed=42,
            verbose=100,
            cat_features=cat_features_indices # Utiliser les indices
        )
        preprocessor = create_catboost_preprocessor(numeric_features, ohe_features, target_encode_features)
    
    # Cas pour CatBoost Search
    elif model_name == "catboost_search":
        if CatBoostRegressor is None: raise ImportError("CatBoost n'est pas installé.")
        print("Modèle: CatBoost (Recherche Hyperparamètres)")
        
        # Calculer les INDICES des features catégorielles
        cat_features_indices = list(range(
            len(numeric_features), 
            len(numeric_features) + len(all_cat_features_for_catboost)
        ))
        
        base_model = CatBoostRegressor(
            loss_function='RMSE',
            eval_metric='R2',
            random_seed=42,
            verbose=0,
            cat_features=cat_features_indices # Utiliser les indices
        )

        model = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid_catboost,
            n_iter=50, # 50 itérations
            scoring=r2_scorer,
            cv=5, 
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        preprocessor = create_catboost_preprocessor(numeric_features, ohe_features, target_encode_features)
    
    # Cas pour Stacking
    elif model_name == "stacking":
        if LGBMRegressor is None or XGBRegressor is None:
             raise ImportError("LGBM/XGB sont requis pour le stacking.")
        
        print("Modèle: Stacking Regressor (XGB + RF + LGBM)")
        
        # 1. Définir le préprocesseur simple
        preprocessor_simple = create_simple_preprocessor(numeric_features, ohe_features, target_encode_features)

        # 2. Définir les pipelines de base
        
        # Pipeline XGBoost (R²=0.538)
        pipe_xgb = Pipeline([
            ('preprocessor', preprocessor_simple),
            ('regressor', XGBRegressor(
                n_estimators=500, max_depth=12, learning_rate=0.1, subsample=0.7,
                colsample_bytree=0.7, reg_alpha=0, reg_lambda=2.0, min_child_weight=3,
                gamma=0, random_state=42, n_jobs=-1, verbosity=0
            ))
        ])

        # Pipeline Random Forest (R²=0.536)
        pipe_rf = Pipeline([
            ('preprocessor', preprocessor_simple),
            ('regressor', RandomForestRegressor(
                n_estimators=200, max_depth=30, min_samples_split=2,
                min_samples_leaf=2, max_features='sqrt', bootstrap=False,
                random_state=42, n_jobs=-1, verbose=0
            ))
        ])
        
        # Pipeline LGBM (R²=0.520)
        pipe_lgbm = Pipeline([
            ('preprocessor', preprocessor_simple), 
            ('regressor', LGBMRegressor(
                n_estimators=500, max_depth=12, learning_rate=0.2, num_leaves=127,
                subsample=0.6, colsample_bytree=0.8, reg_alpha=0.01, reg_lambda=1.5,
                random_state=42, n_jobs=-1, verbose=-1
            ))
        ])
        
        # 3. Définir les estimateurs de base
        estimators = [
            ('xgb', pipe_xgb),
            ('rf', pipe_rf),
            ('lgbm', pipe_lgbm)
        ]
        
        # 4. Définir le StackingRegressor
        # Utilise un Ridge simple comme méta-modèle (plus rapide et stable)
        model = StackingRegressor(
            estimators=estimators,
            final_estimator=RidgeCV(), # Plus stable
            cv=5, 
            n_jobs=-1,
            passthrough=False 
        )
        
        # Le StackingRegressor gère tout, le préprocesseur principal est inutile
        preprocessor = 'passthrough'
        
    else:
        raise ValueError(f"Modèle '{model_name}' non reconnu.")
    
    return model, preprocessor


def main(model_name):
    """
    Fonction principale pour entraîner un modèle et générer la soumission.
    """
    print("="*80)
    print(f"ENTRAÎNEMENT - {model_name.upper()}")
    print("="*80)

    # 1. Chargement et préparation des données
    print("1. Chargement des données...")
    train_df, test_df = load_data()
    
    # Utilisation des features AVANCÉES (Log, Interactions, Bins)
    train_df = initial_feature_engineering(train_df, apply_log_transform=True, create_interactions=True, create_bins=True)
    test_df = initial_feature_engineering(test_df, apply_log_transform=True, create_interactions=True, create_bins=True)
    
    X_train, y_train = get_features_and_target(train_df)
    X_test, test_row_ids = get_test_features(test_df)
    
    # Récupérer les 3 listes de features
    numeric_features, ohe_features, target_encode_features = get_feature_names(
        include_log_features=True, 
        include_interactions=True,
        include_bins=True
    )
    
    print(f"   Observations train: {len(X_train):,}")
    print(f"   Features (brutes): {X_train.shape[1]}")
    print()

    # 2. Obtenir le modèle et le préprocesseur
    print("2. Configuration du modèle...")
    regressor, preprocessor = get_model_and_preprocessor(
        model_name, numeric_features, ohe_features, target_encode_features
    )
    print()

    # 3. Créer le pipeline complet
    print("3. Création du pipeline...")
    
    # Le stacking a un préprocesseur 'passthrough' car il gère tout en interne
    if preprocessor == 'passthrough':
        full_pipeline = Pipeline(steps=[('regressor', regressor)])
    else:
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ])
    print("   ✓ Pipeline créé")
    print()

    # 4. Entraîner le modèle
    print("4. Entraînement du modèle...")
    if model_name == 'stacking':
        print("   (Le Stacking est très long, cela peut prendre 20-30 minutes...)")
    elif 'search' in model_name:
        print("   (La recherche d'hyperparamètres est très longue...)")
    else:
         print("   (Cela peut prendre quelques minutes...)")
    
    # Le .fit(X, y) gère tout (TargetEncoder, CatBoost cat_features, Stacking CV)
    full_pipeline.fit(X_train, y_train)

    print("   ✓ Entraînement terminé")
    print()

    # Afficher les hyperparamètres optimaux si RandomizedSearchCV
    if 'search' in model_name:
        best_r2 = full_pipeline.named_steps['regressor'].best_score_
        best_params = full_pipeline.named_steps['regressor'].best_params_
        
        print("   ✓ Résultats de la recherche (RandomizedSearchCV):")
        print(f"   Meilleur R² (CV): {best_r2:.4f}")
        print("   Meilleurs paramètres:")
        for k, v in best_params.items():
            print(f"     - {k}: {v}")
        print()
        
        if not os.path.exists('results'):
            os.makedirs('results')
        
        model_filename = f"results/best_model_{model_name}.joblib"
        joblib.dump(full_pipeline.named_steps['regressor'].best_estimator_, model_filename)
        print(f"   ✓ Meilleur modèle enregistré sous '{model_filename}'")
        print()

    if model_name == "polynomial_ridge":
        best_alpha = full_pipeline.named_steps['regressor'].alpha_
        print(f"   Alpha optimal (RidgeCV): {best_alpha:.4f}")
        print()

    # 5. Faire les prédictions
    print("5. Génération des prédictions...")
    predictions = full_pipeline.predict(X_test)
    predictions = np.clip(predictions, 0, 100) # Assure que les scores sont valides
    print(f"   Prédictions générées: {len(predictions):,}")
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
    print("Aperçu des prédictions:")
    print(submission_df.head(10))
    print()
    print("="*80)
    print("ENTRAÎNEMENT TERMINÉ")
    print("="*80)

if __name__ == "__main__":
    
    # Vérifie les imports optionnels
    models_available = ["polynomial_ridge", "random_forest", "random_forest_search"]
    if LGBMRegressor:
        models_available.extend(["lightgbm", "lightgbm_search"])
    if XGBRegressor:
        models_available.extend(["xgboost", "xgboost_search"])
    if CatBoostRegressor:
        models_available.extend(["catboost", "catboost_search"])
    if LGBMRegressor and XGBRegressor and RandomForestRegressor:
        models_available.append("stacking")

    parser = argparse.ArgumentParser(
        description="Entraîner un modèle pour la prédiction de popularité Spotify.",
        epilog="Exemple: python train_final.py --model stacking"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        choices=sorted(list(set(models_available))),
        help="Le modèle à entraîner."
    )
    args = parser.parse_args()
    main(args.model)