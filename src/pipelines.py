from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.data_preparation import OutlierClipper

def create_simple_preprocessor(numeric_features, categorical_features, scaler_type='standard'):
    """
    Crée un pipeline de prétraitement simple :
    - Standardisation pour les variables numériques.
    - One-Hot Encoding pour les variables catégorielles.

    Parameters:
    -----------
    numeric_features : list
        Liste des noms de features numériques
    categorical_features : list
        Liste des noms de features catégorielles
    scaler_type : str, default='standard'
        Type de scaler à utiliser: 'standard', 'robust', 'minmax'

    Returns:
    --------
    ColumnTransformer : Le préprocesseur configuré
    """
    # Choisir le scaler approprié
    if scaler_type == 'standard':
        numeric_transformer = StandardScaler()
    elif scaler_type == 'robust':
        # RobustScaler est moins sensible aux outliers
        numeric_transformer = RobustScaler()
    elif scaler_type == 'minmax':
        numeric_transformer = MinMaxScaler()
    else:
        raise ValueError(f"scaler_type '{scaler_type}' non reconnu. Utilisez 'standard', 'robust' ou 'minmax'.")

    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor


def create_polynomial_preprocessor(numeric_features, categorical_features, degree=2, scaler_type='standard'):
    """
    Crée un pipeline de prétraitement qui génère des features polynomiales
    pour les variables numériques, en plus du traitement standard.

    Parameters:
    -----------
    numeric_features : list
        Liste des noms de features numériques
    categorical_features : list
        Liste des noms de features catégorielles
    degree : int, default=2
        Degré des features polynomiales
    scaler_type : str, default='standard'
        Type de scaler à utiliser: 'standard', 'robust', 'minmax'

    Returns:
    --------
    ColumnTransformer : Le préprocesseur configuré
    """
    # Choisir le scaler approprié
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"scaler_type '{scaler_type}' non reconnu.")

    # Pipeline pour les variables numériques : scaling PUIS création de features polynomiales
    numeric_polynomial_transformer = Pipeline(steps=[
        ('scaler', scaler),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False))
    ])

    # Transformeur simple pour les variables catégorielles
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_poly', numeric_polynomial_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor


def create_robust_preprocessor(numeric_features, categorical_features, clip_outliers=True):
    """
    Crée un pipeline de prétraitement robuste aux outliers :
    - Clipping des outliers (optionnel)
    - RobustScaler pour la standardisation
    - One-Hot Encoding pour les variables catégorielles

    Parameters:
    -----------
    numeric_features : list
        Liste des noms de features numériques
    categorical_features : list
        Liste des noms de features catégorielles
    clip_outliers : bool, default=True
        Si True, clippe les outliers avant le scaling

    Returns:
    --------
    ColumnTransformer : Le préprocesseur configuré
    """
    if clip_outliers:
        # Pipeline avec clipping puis scaling robuste
        numeric_transformer = Pipeline(steps=[
            ('clipper', OutlierClipper(lower_quantile=0.01, upper_quantile=0.99)),
            ('scaler', RobustScaler())
        ])
    else:
        numeric_transformer = RobustScaler()

    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor


def create_advanced_preprocessor(numeric_features, categorical_features,
                                 use_polynomial=True, degree=2,
                                 clip_outliers=True, scaler_type='robust'):
    """
    Crée un pipeline de prétraitement avancé combinant plusieurs techniques :
    - Clipping des outliers (optionnel)
    - Scaling robuste ou standard
    - Features polynomiales (optionnel)
    - One-Hot Encoding pour les catégorielles

    Parameters:
    -----------
    numeric_features : list
        Liste des noms de features numériques
    categorical_features : list
        Liste des noms de features catégorielles
    use_polynomial : bool, default=True
        Si True, crée des features polynomiales
    degree : int, default=2
        Degré des features polynomiales
    clip_outliers : bool, default=True
        Si True, clippe les outliers
    scaler_type : str, default='robust'
        Type de scaler: 'standard', 'robust', 'minmax'

    Returns:
    --------
    ColumnTransformer : Le préprocesseur configuré
    """
    # Choisir le scaler
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"scaler_type '{scaler_type}' non reconnu.")

    # Construire le pipeline numérique
    numeric_steps = []

    if clip_outliers:
        numeric_steps.append(('clipper', OutlierClipper(lower_quantile=0.01, upper_quantile=0.99)))

    numeric_steps.append(('scaler', scaler))

    if use_polynomial:
        numeric_steps.append(('poly', PolynomialFeatures(degree=degree, include_bias=False)))

    numeric_transformer = Pipeline(steps=numeric_steps)

    # Transformeur catégoriel
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor
