import argparse
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Import des modules personnalisés
from src.data_preparation import load_data, initial_feature_engineering, get_features_and_target, get_test_features, get_feature_names
from src.pipelines import create_simple_preprocessor, create_polynomial_preprocessor

def get_model_and_preprocessor(model_name, numeric_features, categorical_features):
    """
    Retourne le modèle et le pipeline de prétraitement approprié en fonction du nom du modèle.
    """
    # Définition des modèles
    models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15, min_samples_leaf=5),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5, learning_rate=0.1),
        "polynomial_ridge": RidgeCV(alphas=np.logspace(-2, 2, 10))
    }

    if model_name not in models:
        raise ValueError(f"Modèle '{model_name}' non reconnu. Modèles valides : {list(models.keys())}")

    model = models[model_name]

    # Choisir le préprocesseur
    if model_name == "polynomial_ridge":
        print("Utilisation du préprocesseur avec features polynomiales.")
        preprocessor = create_polynomial_preprocessor(numeric_features, categorical_features)
    else:
        print("Utilisation du préprocesseur simple.")
        preprocessor = create_simple_preprocessor(numeric_features, categorical_features)
        
    return model, preprocessor

def main(model_name):
    """
    Fonction principale pour entraîner un modèle et générer la soumission.
    """
    print(f"\n--- Lancement de l'entraînement pour le modèle : {model_name} ---")

    # 1. Chargement et préparation des données
    train_df, test_df = load_data()
    
    train_df = initial_feature_engineering(train_df)
    test_df = initial_feature_engineering(test_df)
    
    X_train, y_train = get_features_and_target(train_df)
    X_test, test_row_ids = get_test_features(test_df)
    
    numeric_features, categorical_features = get_feature_names()

    # 2. Obtenir le modèle et le préprocesseur
    regressor, preprocessor = get_model_and_preprocessor(model_name, numeric_features, categorical_features)

    # 3. Créer le pipeline complet
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])

    # 4. Entraîner le modèle
    print("Entraînement du pipeline complet...")
    full_pipeline.fit(X_train, y_train)
    print("Entraînement terminé.")

    if model_name == "polynomial_ridge":
        best_alpha = full_pipeline.named_steps['regressor'].alpha_
        print(f"Meilleur alpha trouvé par RidgeCV : {best_alpha}")

    # 5. Faire les prédictions
    print("Prédiction sur les données de test...")
    predictions = full_pipeline.predict(X_test)
    predictions = np.clip(predictions, 0, 100) # S'assurer que les prédictions sont dans la plage [0, 100]

    # 6. Créer le fichier de soumission
    submission_df = pd.DataFrame({'row_id': test_row_ids, 'popularity': predictions})
    filename = f"submission_{model_name}.csv"
    submission_df.to_csv(filename, index=False)
    
    print(f"Fichier de soumission '{filename}' généré avec succès.")
    print(submission_df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîner un modèle pour la prédiction de popularité Spotify.")
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True, 
        choices=["linear", "ridge", "random_forest", "gradient_boosting", "polynomial_ridge"],
        help="Le nom du modèle à entraîner."
    )
    args = parser.parse_args()
    main(args.model_name)
