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

def initial_feature_engineering(df, apply_log_transform=False, create_interactions=False, create_bins=False):
    """
    Applique les transformations de features initiales et communes à un dataframe.
    """
    df = df.copy()

    # Gère la variable cyclique 'key' en la transformant en sinus/cosinus
    if 'key' in df.columns:
        df['key_sin'] = np.sin(2 * np.pi * df['key']/12)
        df['key_cos'] = np.cos(2 * np.pi * df['key']/12)
        # df = df.drop('key', axis=1)

    # Transformation logarithmique pour les features asymétriques
    if apply_log_transform:
        if 'duration_ms' in df.columns:
            df['duration_ms_log'] = np.log1p(df['duration_ms'])
        if 'speechiness' in df.columns:
            df['speechiness_log'] = np.log1p(df['speechiness'])
        if 'liveness' in df.columns:
            df['liveness_log'] = np.log1p(df['liveness'])
        if 'instrumentalness' in df.columns:
            df['instrumentalness_log'] = np.log1p(df['instrumentalness'])

    # Création de features d'interaction
    if create_interactions:
        if 'energy' in df.columns and 'loudness' in df.columns:
            df['energy_loudness'] = df['energy'] * df['loudness']
        if 'danceability' in df.columns and 'energy' in df.columns:
            df['danceability_energy'] = df['danceability'] * df['energy']
        if 'acousticness' in df.columns and 'instrumentalness' in df.columns:
            df['acoustic_instrumental'] = df['acousticness'] * df['instrumentalness']
        if 'speechiness' in df.columns and 'duration_ms' in df.columns:
            # +1 pour éviter division par 0 sur des durées nulles
            df['speech_density'] = df['speechiness'] / (df['duration_ms'] / 1000 + 1)
        if 'valence' in df.columns and 'energy' in df.columns:
            df['valence_energy'] = df['valence'] * df['energy']
        if 'danceability' in df.columns and 'valence' in df.columns:
            df['danceability_valence'] = df['danceability'] * df['valence']
        if 'energy' in df.columns and 'tempo' in df.columns:
            df['energy_tempo'] = df['energy'] * df['tempo']

    # Création de features discrétisées (bins)
    if create_bins:
        # Discrétise la durée en 10 quantiles
        if 'duration_ms' in df.columns:
            df['duration_bin'] = pd.qcut(df['duration_ms'], q=10, labels=False, duplicates='drop').astype(str)
        # Discrétise le tempo en 10 quantiles
        if 'tempo' in df.columns:
            df['tempo_bin'] = pd.qcut(df['tempo'], q=10, labels=False, duplicates='drop').astype(str)

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

def get_feature_names(include_log_features=False, include_interactions=False, include_bins=False):
    """
    Définit et retourne les listes de noms pour les variables numériques,
    catégorielles (OHE) et catégorielles (Target Encoding).
    """
    # Features numériques de base
    numeric_features = [
        'duration_ms', 'danceability', 'energy', 'loudness',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo', 'key_sin', 'key_cos'
    ]

    # Remplacer les features par leur version log si demandé
    if include_log_features:
        log_features_map = {
            'duration_ms': 'duration_ms_log',
            'speechiness': 'speechiness_log',
            'liveness': 'liveness_log',
            'instrumentalness': 'instrumentalness_log'
        }
        numeric_features = [log_features_map.get(f, f) for f in numeric_features]

    if include_interactions:
        interaction_features = [
            'energy_loudness', 'danceability_energy', 'acoustic_instrumental',
            'speech_density', 'valence_energy', 'danceability_valence', 'energy_tempo'
        ]
        numeric_features.extend(interaction_features)

    # Features pour One-Hot Encoding (faible cardinalité)
    ohe_features = ['mode', 'time_signature', 'explicit', 'key']
    
    # Ajout des bins aux features OHE
    if include_bins:
        ohe_features.extend(['duration_bin', 'tempo_bin'])
    
    # Feature pour Target Encoding (haute cardinalité)
    target_encode_features = ['track_genre']

    # Retourner 3 listes
    return numeric_features, ohe_features, target_encode_features


class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Transformateur personnalisé pour clipper les outliers basé sur les quantiles.
    """
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bounds_ = None
        self.upper_bounds_ = None

    def fit(self, X, y=None):
        self.lower_bounds_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.quantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        X_clipped = np.clip(X, self.lower_bounds_, self.upper_bounds_)
        return X_clipped