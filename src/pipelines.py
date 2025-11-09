from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, RobustScaler, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.data_preparation import OutlierClipper

def create_simple_preprocessor(numeric_features, ohe_features, target_encode_features, scaler_type='standard'):
    """
    Crée un pipeline de prétraitement pour RF, LGBM, XGB :
    - Scaling pour les variables numériques.
    - One-Hot Encoding pour les catégorielles à faible cardinalité.
    - Target Encoding pour les catégorielles à forte cardinalité (ex: genre).
    """
    
    if scaler_type == 'standard':
        numeric_transformer = StandardScaler()
    elif scaler_type == 'robust':
        numeric_transformer = RobustScaler()
    else:
        raise ValueError(f"scaler_type '{scaler_type}' non reconnu.")

    ohe_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # TargetEncoder gère le 'y' lors du .fit() dans un pipeline
    target_transformer = TargetEncoder(target_type='continuous', smooth='auto', random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('ohe', ohe_transformer, ohe_features),
            ('target_enc', target_transformer, target_encode_features)
        ],
        remainder='drop' # Ignorer les colonnes non spécifiées (ex: 'key' brute)
    )
    return preprocessor

# Préprocesseur spécifique pour CatBoost
def create_catboost_preprocessor(numeric_features, ohe_features, target_encode_features):
    """
    Crée un pipeline de prétraitement pour CatBoost :
    - Scaling pour les variables numériques.
    - Passthrough pour TOUTES les features catégorielles.
    CatBoost gérera l'encodage nativement.
    """
    numeric_transformer = StandardScaler()
    
    # Concatène toutes les features catégorielles
    all_categorical_features = ohe_features + target_encode_features

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            # 'passthrough' signifie ne rien faire, CatBoost les recevra brutes
            ('cat', 'passthrough', all_categorical_features) 
        ],
        remainder='drop'
    )
    return preprocessor


def create_polynomial_preprocessor(numeric_features, categorical_features, degree=2, scaler_type='standard'):
    """
    Crée un pipeline de prétraitement qui génère des features polynomiales
    pour les variables numériques, en plus du OHE pour les catégorielles.
    Utilisé pour le modèle Ridge.
    """
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"scaler_type '{scaler_type}' non reconnu.")

    # Pipeline pour les variables numériques : scaling PUIS création de features polynomiales
    numeric_polynomial_transformer = Pipeline(steps=[
        ('scaler', scaler),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False))
    ])

    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_poly', numeric_polynomial_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    return preprocessor