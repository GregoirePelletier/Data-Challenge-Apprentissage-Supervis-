# Script PowerShell pour entraîner tous les modèles séquentiellement

# Liste des modèles à entraîner
$model_names = @(
    "linear",
    "ridge",
    "random_forest",
    "gradient_boosting",
    "polynomial_ridge"
)

Write-Host "Début de l'entraînement de tous les modèles..."

foreach ($model in $model_names) {
    Write-Host "--------------------------------------------------"
    Write-Host "Entraînement du modèle : $model"
    Write-Host "--------------------------------------------------"
    
    # Exécute le script train.py avec le nom du modèle en argument
    python train.py --model_name $model
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "L'entraînement du modèle $model a échoué. Arrêt du script."
        break
    }
}

Write-Host "--------------------------------------------------"
Write-Host "Tous les modèles ont été entraînés avec succès."
Write-Host "--------------------------------------------------"
