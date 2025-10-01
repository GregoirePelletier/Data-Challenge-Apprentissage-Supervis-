import pandas as pd
import numpy as np

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

def initial_feature_engineering(df):
    """
    Applique les transformations de features initiales et communes à un dataframe.
    - Gère la variable cyclique 'key'.
    - Supprime les colonnes inutiles.
    """
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

def get_feature_names():
    """
    Définit et retourne les listes de noms pour les variables numériques et catégorielles.
    """
    # Ces listes sont basées sur le dataset APRÈS l'ingénierie de features initiale
    numeric_features = [
        'duration_ms', 'danceability', 'energy', 'loudness', 
        'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
        'valence', 'tempo', 'key_sin', 'key_cos'
    ]
    
    categorical_features_numeric = ['mode', 'time_signature', 'explicit']
    categorical_features_object = ['track_genre']
    
    all_categorical_features = categorical_features_numeric + categorical_features_object
    
    return numeric_features, all_categorical_features
