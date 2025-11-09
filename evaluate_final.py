"""
Script d'évaluation avec validation croisée pour comparer les modèles.
Utilise le Target Encoding et le nouveau CatBoost.
"""

import pandas as pd
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Import des modèles de boosting
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

from src.data_preparation import (
    load_data,
    initial_feature_engineering,
    get_features_and_target,
    get_feature_names
)
from src.pipelines import (
    create_simple_preprocessor, 
    create_polynomial_preprocessor,
    create_catboost_preprocessor # NOUVEAU
)

def evaluate_model(pipeline, X, y, model_name, cv_folds=3, fit_params=None):
    """
    Évalue un modèle avec validation croisée.
    Gère le TargetEncoder (qui a besoin de 'y') et les fit_params (pour CatBoost).
    """
    if fit_params is None:
        fit_params = {}
        
    print(f"\nÉvaluation: {model_name}")
    print("-" * 60)
    
    scoring = ['r2', 'neg_mean_squared_error']
    
    try:
        cv_results = cross_validate(
            pipeline, X, y,
            cv=cv_folds,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
            verbose=0,
            **fit_params # Passe les paramètres au .fit()
        )
    except Exception as e:
        print(f"ERREUR lors de l'évaluation de {model_name}: {e}")
        return None
    
    # Calculer les métriques
    test_r2 = cv_results['test_r2'].mean()
    test_r2_std = cv_results['test_r2'].std()
    test_rmse = np.sqrt(-cv_results['test_neg_mean_squared_error']).mean()
    train_r2 = cv_results['train_r2'].mean()
    overfit = train_r2 - test_r2
    
    print(f"R² (test):  {test_r2:.4f} ± {test_r2_std:.4f}")
    print(f"R² (train): {train_r2:.4f}")
    print(f"RMSE:       {test_rmse:.2f}")
    print(f"Écart:      {overfit:.4f}", end="")
    
    if overfit > 0.1: print(" Sur-apprentissage")
    else: print(" OK")
    
    return {
        'model': model_name,
        'test_r2_mean': test_r2,
        'test_r2_std': test_r2_std,
        'test_rmse_mean': test_rmse,
        'train_r2_mean': train_r2,
        'overfit': overfit
    }

def main():
    print("="*80)
    print("ÉVALUATION DES MODÈLES - VALIDATION CROISÉE 3-FOLD")
    print("="*80)
    
    # 1. Chargement des données
    print("1. Chargement des données...")
    train_df, _ = load_data()
    
    # Activation de toutes les features (Log, Interactions, Bins)
    train_df = initial_feature_engineering(train_df, apply_log_transform=True, create_interactions=True, create_bins=True)
    X_train, y_train = get_features_and_target(train_df)

    # Récupérer les listes de features
    numeric_features, ohe_features, target_encode_features = get_feature_names(
        include_log_features=True, 
        include_interactions=True,
        include_bins=True
    )
    
    print(f"   Observations: {len(X_train):,}")
    print()
    
    # 2. Définir les modèles
    print("2. Configuration des modèles...")
    
    models = []
    
    # Ridge Polynomial
    all_ohe_features = ohe_features + target_encode_features
    models.append({
        'name': 'Ridge Polynomial',
        'pipeline': Pipeline([
            ('preprocessor', create_polynomial_preprocessor(numeric_features, all_ohe_features)),
            ('regressor', RidgeCV(alphas=np.logspace(-2, 2, 10)))
        ]),
        'fit_params': {}
    })
    
    # Random Forest (Paramètres allégés pour évaluation rapide)
    models.append({
        'name': 'Random Forest',
        'pipeline': Pipeline([
            ('preprocessor', create_simple_preprocessor(numeric_features, ohe_features, target_encode_features)),
            ('regressor', RandomForestRegressor(
                n_estimators=200,  # Réduit
                min_samples_leaf=5, # Valeur sûre
                max_features='sqrt', # Rapide
                bootstrap=False, # Utilise vos params optimisés
                random_state=42, n_jobs=-1, verbose=0
            ))
        ]),
        'fit_params': {}
    })
    
    # LightGBM (Paramètres allégés pour évaluation rapide)
    if LGBMRegressor:
        models.append({
            'name': 'LightGBM',
            'pipeline': Pipeline([
                ('preprocessor', create_simple_preprocessor(numeric_features, ohe_features, target_encode_features)),
                ('regressor', LGBMRegressor(
                    n_estimators=300, # Réduit
                    learning_rate=0.2, 
                    max_depth=12,
                    num_leaves=127, 
                    random_state=42, n_jobs=-1, verbose=-1
                ))
            ]),
            'fit_params': {}
        })
    
    # XGBoost (Paramètres allégés pour évaluation rapide)
    if XGBRegressor:
        models.append({
            'name': 'XGBoost',
            'pipeline': Pipeline([
                ('preprocessor', create_simple_preprocessor(numeric_features, ohe_features, target_encode_features)),
                ('regressor', XGBRegressor(
                    n_estimators=300, # Réduit
                    learning_rate=0.1, 
                    max_depth=12,
                    subsample=0.7, 
                    colsample_bytree=0.7,
                    random_state=42, n_jobs=-1, verbosity=0
                ))
            ]),
            'fit_params': {}
        })

    # CatBoost
    if CatBoostRegressor:
        # Calculer les INDICES des features cat. APRES le ColumnTransformer
        all_cat_features_names = ohe_features + target_encode_features
        cat_features_indices = list(range(
            len(numeric_features), 
            len(numeric_features) + len(all_cat_features_names)
        ))
        
        models.append({
            'name': 'CatBoost',
            'pipeline': Pipeline([
                ('preprocessor', create_catboost_preprocessor(numeric_features, ohe_features, target_encode_features)),
                ('regressor', CatBoostRegressor(
                    iterations=500, # Réduit pour évaluation rapide
                    learning_rate=0.1, depth=10,
                    loss_function='RMSE', eval_metric='R2',
                    random_seed=42, verbose=0,
                    cat_features=cat_features_indices # CORRECTION: Utiliser les indices
                ))
            ]),
            'fit_params': {} # 'fit_params' est vide
        })
    
    print(f"   {len(models)} modèles configurés")
    print()
    
    # 3. Évaluer les modèles
    print("3. Évaluation avec validation croisée (3-fold)...")
    print("   (Peut prendre plusieurs minutes...)")
    
    results = []
    for model_config in models:
        result = evaluate_model(
            model_config['pipeline'], X_train, y_train,
            model_config['name'], cv_folds=3,
            fit_params=model_config['fit_params']
        )
        if result:
            results.append(result)
    
    # 4. Afficher le résumé
    print()
    print("="*80)
    print("RÉSUMÉ DES RÉSULTATS")
    print("="*80)
    
    if not results:
        print("Aucun modèle n'a été évalué. Vérifiez les imports (LGBM, XGB, CatBoost).")
        return

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_r2_mean', ascending=False)
    
    print("Classement par R² (test):")
    print()
    for idx, row in results_df.iterrows():
        rank = results_df.index.get_loc(idx) + 1
        print(f"{rank}. {row['model']:20s} R² = {row['test_r2_mean']:.4f} ± {row['test_r2_std']:.4f}  "
              f"RMSE = {row['test_rmse_mean']:.2f}  Écart = {row['overfit']:.4f}")
    
    print()
    
    if not os.path.exists('results'):
        os.makedirs('results')
    results_df.to_csv('results/evaluation_final.csv', index=False)
    print("✓ Résultats sauvegardés: results/evaluation_final.csv")
    print()
    
    # Recommandation
    best_model = results_df.iloc[0]
    print("="*80)
    print("RECOMMANDATION (BASÉE SUR L'ÉVALUATION)")
    print("="*80)
    print(f"Meilleur modèle de base: {best_model['model']}")
    print(f"R² (test): {best_model['test_r2_mean']:.4f}")
    print()
    print("Considérez d'entraîner le modèle 'stacking' pour la soumission finale,")
    print("car il combine la force de plusieurs de ces modèles de base.")
    print("\nExemples de commandes d'entraînement final :")
    print(f"  python train_final.py --model {best_model['model'].lower().replace(' ', '_')}")
    print(f"  python train_final.py --model stacking")

if __name__ == "__main__":
    main()