from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_simple_preprocessor(numeric_features, categorical_features):
    """
    Crée un pipeline de prétraitement simple :
    - Standardisation pour les variables numériques.
    - One-Hot Encoding pour les variables catégorielles.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

def create_polynomial_preprocessor(numeric_features, categorical_features):
    """
    Crée un pipeline de prétraitement qui génère des features polynomiales
    pour les variables numériques, en plus du traitement standard.
    """
    # Pipeline pour les variables numériques : scaling PUIS création de features polynomiales
    numeric_polynomial_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2, include_bias=False))
    ])

    # Transformeur simple pour les variables catégorielles
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_poly', numeric_polynomial_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor
