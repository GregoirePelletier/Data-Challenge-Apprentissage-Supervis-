"""
Script d'évaluation avec validation croisée pour comparer les modèles.

Évalue les 4 modèles sélectionnés avec validation croisée 3-fold (allégé).
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

from src.data_preparation import (
    load_data, 
    initial_feature_engineering, 
    get_features_and_target,
    get_feature_names
)
from src.pipelines import create_simple_preprocessor, create_polynomial_preprocessor

def evaluate_model(pipeline, X, y, model_name, cv_folds=3):
    """
    Évalue un modèle avec validation croisée.
    """
    print(f"\nÉvaluation: {model_name}")
    print("-" * 60)
    
    scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    
    cv_results = cross_validate(
        pipeline, X, y,
        cv=cv_folds,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
        verbose=0
    )
    
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
    
    if overfit > 0.1:
        print(" ⚠️  Sur-apprentissage")
    elif overfit < 0.01:
        print(" ✓ Excellent")
    else:
        print(" ✓ Bon")
    
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
    print()
    
    # 1. Chargement des données
    print("1. Chargement des données...")
    train_df, _ = load_data()
    train_df = initial_feature_engineering(train_df)
    X_train, y_train = get_features_and_target(train_df)
    numeric_features, categorical_features = get_feature_names()
    
    print(f"   Observations: {len(X_train):,}")
    print(f"   Features: {X_train.shape[1]}")
    print()
    
    # 2. Définir les modèles
    print("2. Configuration des modèles...")
    
    models = []
    
    # Ridge Polynomial
    models.append({
        'name': 'Ridge Polynomial',
        'pipeline': Pipeline([
            ('preprocessor', create_polynomial_preprocessor(numeric_features, categorical_features)),
            ('regressor', RidgeCV(alphas=np.logspace(-2, 2, 10)))
        ])
    })
    
    # Random Forest (version allégée pour évaluation)
    models.append({
        'name': 'Random Forest',
        'pipeline': Pipeline([
            ('preprocessor', create_simple_preprocessor(numeric_features, categorical_features)),
            ('regressor', RandomForestRegressor(
                n_estimators=200,  # Réduit pour évaluation rapide
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=5,
                max_features=0.7,
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
                verbose=0
            ))
        ])
    })
    
    # LightGBM
    try:
        from lightgbm import LGBMRegressor
        models.append({
            'name': 'LightGBM',
            'pipeline': Pipeline([
                ('preprocessor', create_simple_preprocessor(numeric_features, categorical_features)),
                ('regressor', LGBMRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=7,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                ))
            ])
        })
    except ImportError:
        print("   ⚠️  LightGBM non installé (pip install lightgbm)")
    
    # XGBoost
    try:
        from xgboost import XGBRegressor
        models.append({
            'name': 'XGBoost',
            'pipeline': Pipeline([
                ('preprocessor', create_simple_preprocessor(numeric_features, categorical_features)),
                ('regressor', XGBRegressor(
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
                ))
            ])
        })
    except ImportError:
        print("   ⚠️  XGBoost non installé (pip install xgboost)")
    
    print(f"   {len(models)} modèles configurés")
    print()
    
    # 3. Évaluer les modèles
    print("3. Évaluation avec validation croisée (3-fold)...")
    print("   (Cela peut prendre 5-10 minutes...)")
    
    results = []
    for model_config in models:
        result = evaluate_model(
            model_config['pipeline'],
            X_train,
            y_train,
            model_config['name'],
            cv_folds=3
        )
        results.append(result)
    
    # 4. Afficher le résumé
    print()
    print("="*80)
    print("RÉSUMÉ DES RÉSULTATS")
    print("="*80)
    print()
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_r2_mean', ascending=False)
    
    print("Classement par R² (test):")
    print()
    for idx, row in results_df.iterrows():
        rank = results_df.index.get_loc(idx) + 1
        print(f"{rank}. {row['model']:20s} R² = {row['test_r2_mean']:.4f} ± {row['test_r2_std']:.4f}  "
              f"RMSE = {row['test_rmse_mean']:.2f}  Écart = {row['overfit']:.4f}")
    
    print()
    
    # Sauvegarder les résultats
    results_df.to_csv('results/evaluation_final.csv', index=False)
    print("✓ Résultats sauvegardés: results/evaluation_final.csv")
    print()
    
    # Recommandation
    best_model = results_df.iloc[0]
    print("="*80)
    print("RECOMMANDATION")
    print("="*80)
    print()
    print(f"Meilleur modèle: {best_model['model']}")
    print(f"R² (test): {best_model['test_r2_mean']:.4f}")
    print(f"RMSE: {best_model['test_rmse_mean']:.2f}")
    print()
    
    if best_model['overfit'] > 0.1:
        print("⚠️  Attention: Sur-apprentissage détecté")
        print("   Considérer une régularisation plus forte ou réduire la complexité")
    else:
        print("✓ Modèle bien calibré, prêt pour la soumission")
    
    print()
    print(f"Pour entraîner ce modèle:")
    model_arg = best_model['model'].lower().replace(' ', '_')
    print(f"  python train_final.py --model {model_arg}")

if __name__ == "__main__":
    main()

