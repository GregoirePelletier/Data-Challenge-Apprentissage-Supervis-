# Script pour le challenge Kaggle Spotify - Prédiction de Popularité

# =============================================================================
# 1. Import des librairies
# =============================================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

print("Librairies importées.")

# =============================================================================
# 2. Chargement et exploration des données (EDA)
# =============================================================================
# Charger les données
try:
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data.csv')
    print("Données chargées.")
except FileNotFoundError as e:
    print(f"Erreur de chargement de fichier : {e}")
    print("Veuillez vous assurer que les fichiers train_data.csv et test_data.csv sont dans le bon répertoire.")
    exit()

# Afficher les premières lignes et les informations générales
print("\n--- Aperçu des données d'entraînement ---")
print(train_df.head())
print("\n--- Informations sur les données d'entraînement ---")
train_df.info()

# Statistiques descriptives
print("\n--- Statistiques descriptives ---")
print(train_df.describe())

# Analyse de la variable cible 'popularity'
plt.figure(figsize=(10, 6))
sns.histplot(train_df['popularity'], kde=True, bins=30)
plt.title('Distribution de la Popularité')
plt.xlabel('Popularité')
plt.ylabel('Fréquence')
plt.savefig('popularity_distribution.png')
print("\nGraphique de la distribution de la popularité sauvegardé en 'popularity_distribution.png'")

# Corrélation des variables numériques
numeric_cols = train_df.select_dtypes(include=np.number).columns.tolist()
# Exclure row_id qui n'est pas une feature
numeric_cols.remove('row_id')

plt.figure(figsize=(14, 10))
correlation_matrix = train_df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Matrice de corrélation des variables numériques')
plt.savefig('correlation_matrix.png')
print("Graphique de la matrice de corrélation sauvegardé en 'correlation_matrix.png'")


# =============================================================================
# 3. Prétraitement des données et Feature Engineering
# =============================================================================
print("\n--- Prétraitement des données ---")

# Séparer la cible et les features
# On retire 'Unnamed: 0' qui est un artefact du CSV et 'row_id' qui est un identifiant
X = train_df.drop(['popularity', 'row_id', 'Unnamed: 0'], axis=1)
y = train_df['popularity']

# Garder les row_id du jeu de test pour la soumission
test_row_ids = test_df['row_id']
X_test = test_df.drop(['row_id', 'Unnamed: 0'], axis=1)

# Identifier les types de colonnes
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
# 'explicit', 'mode', 'time_signature' sont catégoriques même si numériques
# 'key' sera traitée séparément car elle est cyclique
categorical_features_numeric = ['mode', 'time_signature', 'explicit']
# Mettre à jour la liste des vraies variables numériques
numeric_features = [col for col in numeric_features if col not in categorical_features_numeric and col != 'key']

categorical_features_object = ['track_genre']

# Traitement de la variable cyclique 'key'
# On la transforme en deux dimensions (sin/cos) pour que le modèle comprenne la cyclicité
X['key_sin'] = np.sin(2 * np.pi * X['key']/12)
X['key_cos'] = np.cos(2 * np.pi * X['key']/12)
X_test['key_sin'] = np.sin(2 * np.pi * X_test['key']/12)
X_test['key_cos'] = np.cos(2 * np.pi * X_test['key']/12)

# On ajoute les nouvelles features cycliques aux features numériques et on retire l'ancienne 'key'
numeric_features.extend(['key_sin', 'key_cos'])
X = X.drop('key', axis=1)
X_test = X_test.drop('key', axis=1)

print(f"Variables numériques: {numeric_features}")
print(f"Variables catégoriques (num): {categorical_features_numeric}")
print(f"Variables catégoriques (obj): {categorical_features_object}")

# Créer le pipeline de prétraitement
# 1. Pour les variables numériques : Standardisation
numeric_transformer = StandardScaler()

# 2. Pour les variables catégoriques : One-Hot Encoding
# On combine toutes les features catégoriques pour le OneHotEncoder
categorical_features = categorical_features_numeric + categorical_features_object
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Utiliser ColumnTransformer pour appliquer différents traitements à différentes colonnes
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Garde les autres colonnes (s'il y en a)
)

print("Pipeline de prétraitement défini.")

# =============================================================================
# 4. Définition et entraînement des modèles
# =============================================================================
print("\n--- Entraînement des modèles ---")

# Définir les modèles à évaluer
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures

# Création d'un pipeline amélioré pour les modèles linéaires
# Etape 1: Prétraitement (scaling, one-hot encoding)
# Etape 2: Création de features polynomiales (degré 2)
# Etape 3: Modèle de régression
polynomial_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
    ('regressor', RidgeCV(alphas=np.logspace(-3, 3, 10)))
])

models = {
    "RidgeCV with Polynomial Features": polynomial_pipeline,
    # On garde les anciens modèles pour comparer
    "Linear Regression": Pipeline(steps=[('preprocessor', preprocessor), ('regressor', LinearRegression())]),
    "Ridge Regression (Simple)": Pipeline(steps=[('preprocessor', preprocessor), ('regressor', Ridge(alpha=1.0))]),
    "Random Forest": Pipeline(steps=[('preprocessor', preprocessor), ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15, min_samples_leaf=5))]),
    "Gradient Boosting": Pipeline(steps=[('preprocessor', preprocessor), ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5, learning_rate=0.1))])
}

# Entraîner chaque modèle et générer les soumissions
for name, pipeline in models.items():
    print(f"\n--- Modèle : {name} ---")
    
    # Entraîner le modèle sur TOUTES les données d'entraînement
    print("Entraînement en cours...")
    pipeline.fit(X, y)
    print("Entraînement terminé.")

    if name == "RidgeCV with Polynomial Features":
        # Afficher le meilleur alpha trouvé
        best_alpha = pipeline.named_steps['regressor'].alpha_
        print(f"Meilleur alpha trouvé par RidgeCV : {best_alpha}")
    
    # Faire les prédictions sur le jeu de test
    print("Prédiction sur les données de test...")
    predictions = pipeline.predict(X_test)
    
    # S'assurer que les prédictions sont dans la plage [0, 100]
    predictions = np.clip(predictions, 0, 100)
    
    # Créer le fichier de soumission
    submission_df = pd.DataFrame({'row_id': test_row_ids, 'popularity': predictions})
    
    # Formatter le nom du fichier
    filename = f"submission_{name.replace(' ', '_').lower()}.csv"
    submission_df.to_csv(filename, index=False)
    
    print(f"Fichier de soumission '{filename}' généré avec succès.")
    print(submission_df.head())

print("\n\nProcessus terminé. Tous les modèles ont été entraînés et les fichiers de soumission sont prêts.")
