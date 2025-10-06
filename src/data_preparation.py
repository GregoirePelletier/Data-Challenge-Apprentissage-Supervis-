import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

def load_data(train_path='train_data.csv', test_path='test_data.csv'):
    """
    Charge les données d'entraînement et de test à partir des chemins spécifiés.
    """
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print("Données chargées avec succès.")
        return train_df, test_df
    except FileNotFoundError as e:
        print(f"Erreur de chargement de fichier : {e}")
        print("Veuillez vous assurer que les fichiers de données sont présents.")
        exit()

def initial_feature_engineering(df, apply_log_transform=False, create_interactions=False):
    """
    Applique les transformations de features initiales et communes à un dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        Le dataframe à transformer
    apply_log_transform : bool, default=False
        Si True, applique une transformation logarithmique aux features asymétriques
    create_interactions : bool, default=False
        Si True, crée des features d'interaction basées sur la connaissance du domaine

    Returns:
    --------
    pd.DataFrame : Le dataframe transformé
    """
    # Copie pour éviter les modifications inattendues
    df = df.copy()

    # Gère la variable cyclique 'key' en la transformant en sinus/cosinus
    if 'key' in df.columns:
        df['key_sin'] = np.sin(2 * np.pi * df['key']/12)
        df['key_cos'] = np.cos(2 * np.pi * df['key']/12)
        df = df.drop('key', axis=1)

    # Supprime les colonnes inutiles si elles existent
    cols_to_drop = ['Unnamed: 0']
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # Transformation logarithmique pour les features asymétriques (basé sur l'EDA)
    if apply_log_transform:
        # duration_ms: skewness = 9.94 (très asymétrique)
        if 'duration_ms' in df.columns:
            df['duration_ms_log'] = np.log1p(df['duration_ms'])

        # speechiness: skewness = 4.66
        if 'speechiness' in df.columns:
            df['speechiness_log'] = np.log1p(df['speechiness'])

        # liveness: skewness = 2.11
        if 'liveness' in df.columns:
            df['liveness_log'] = np.log1p(df['liveness'])

        # instrumentalness: skewness = 1.73
        if 'instrumentalness' in df.columns:
            df['instrumentalness_log'] = np.log1p(df['instrumentalness'])

    # Création de features d'interaction basées sur la connaissance du domaine
    if create_interactions:
        # Interaction energy × loudness (corrélation positive forte)
        if 'energy' in df.columns and 'loudness' in df.columns:
            df['energy_loudness'] = df['energy'] * df['loudness']

        # Interaction danceability × energy (titres énergiques et dansants)
        if 'danceability' in df.columns and 'energy' in df.columns:
            df['danceability_energy'] = df['danceability'] * df['energy']

        # Interaction acousticness × instrumentalness (musique acoustique instrumentale)
        if 'acousticness' in df.columns and 'instrumentalness' in df.columns:
            df['acoustic_instrumental'] = df['acousticness'] * df['instrumentalness']

        # Ratio speechiness / duration (densité de paroles)
        if 'speechiness' in df.columns and 'duration_ms' in df.columns:
            df['speech_density'] = df['speechiness'] / (df['duration_ms'] / 1000 + 1)  # +1 pour éviter division par 0

        # Valence × energy (positivité énergique)
        if 'valence' in df.columns and 'energy' in df.columns:
            df['valence_energy'] = df['valence'] * df['energy']

    return df

def get_features_and_target(train_df):
    """Sépare les features (X) et la cible (y) du dataframe d'entraînement."""
    X = train_df.drop(['popularity', 'row_id'], axis=1)
    y = train_df['popularity']
    return X, y

def get_test_features(test_df):
    """Extrait les features du dataframe de test et conserve les row_ids."""
    test_row_ids = test_df['row_id']
    X_test = test_df.drop('row_id', axis=1)
    return X_test, test_row_ids

def get_feature_names(include_log_features=False, include_interactions=False):
    """
    Définit et retourne les listes de noms pour les variables numériques et catégorielles.

    Parameters:
    -----------
    include_log_features : bool, default=False
        Si True, inclut les features transformées logarithmiquement
    include_interactions : bool, default=False
        Si True, inclut les features d'interaction

    Returns:
    --------
    tuple : (numeric_features, categorical_features)
    """
    # Features numériques de base (après transformation de 'key')
    numeric_features = [
        'duration_ms', 'danceability', 'energy', 'loudness',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo', 'key_sin', 'key_cos'
    ]

    # Ajouter les features transformées logarithmiquement si demandé
    if include_log_features:
        log_features = [
            'duration_ms_log', 'speechiness_log',
            'liveness_log', 'instrumentalness_log'
        ]
        numeric_features.extend(log_features)

    # Ajouter les features d'interaction si demandé
    if include_interactions:
        interaction_features = [
            'energy_loudness', 'danceability_energy',
            'acoustic_instrumental', 'speech_density', 'valence_energy'
        ]
        numeric_features.extend(interaction_features)

    # Features catégorielles
    categorical_features_numeric = ['mode', 'time_signature', 'explicit']
    categorical_features_object = ['track_genre']

    all_categorical_features = categorical_features_numeric + categorical_features_object

    return numeric_features, all_categorical_features


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Transformateur personnalisé pour clipper les outliers basé sur les quantiles.
    Utile pour les features avec des valeurs extrêmes (ex: duration_ms).
    """
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        """
        Parameters:
        -----------
        lower_quantile : float, default=0.01
            Quantile inférieur pour le clipping
        upper_quantile : float, default=0.99
            Quantile supérieur pour le clipping
        """
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        """Calcule les bornes de clipping basées sur les quantiles."""
        self.lower_bounds_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.quantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        """Applique le clipping aux données."""
        X_clipped = np.clip(X, self.lower_bounds_, self.upper_bounds_)
        return X_clipped


class FeatureInteractionCreator(BaseEstimator, TransformerMixin):
    """
    Transformateur personnalisé pour créer des interactions entre features spécifiques.
    Plus flexible que PolynomialFeatures car permet de choisir les interactions.
    """
    def __init__(self, interactions=None):
        """
        Parameters:
        -----------
        interactions : list of tuples, default=None
            Liste de tuples (idx1, idx2) indiquant les paires de features à multiplier
            Si None, crée toutes les interactions possibles
        """
        self.interactions = interactions

    def fit(self, X, y=None):
        """Rien à apprendre, retourne self."""
        return self

    def transform(self, X):
        """Crée les features d'interaction."""
        if self.interactions is None:
            # Créer toutes les interactions possibles
            n_features = X.shape[1]
            interactions = [(i, j) for i in range(n_features) for j in range(i+1, n_features)]
        else:
            interactions = self.interactions

        # Créer les nouvelles features
        interaction_features = []
        for idx1, idx2 in interactions:
            interaction_features.append((X[:, idx1] * X[:, idx2]).reshape(-1, 1))

        if interaction_features:
            X_interactions = np.hstack(interaction_features)
            return np.hstack([X, X_interactions])
        else:
            return X
