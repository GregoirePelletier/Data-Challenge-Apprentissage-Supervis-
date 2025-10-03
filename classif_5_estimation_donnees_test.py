import pandas as pd
from pathlib import Path


fichier = Path(r"C:\Users\saout\Documents\Data-Challenge-Apprentissage-Supervis\data\test_data.csv")
df_test = pd.read_csv(fichier)

df_test_c = pd.get_dummies(data = df_test, \
                         prefix = liste_var_categ, \
                         columns = liste_var_categ)


df_test_c.set_index('row_id', inplace=True)

df_test_c = df_test_c.reindex(columns=df_c.columns, fill_value=0)

var = [x for x in df_c.columns if x not in target] 



y_pred = random_forest_model.predict(df_test_c[var])

submit = pd.DataFrame({
    "row_id": df_test_c.index,                    # récupère l’index
    "prediction": y_pred
})
# si on te demande des proba :
# proba = pipeline.predict_proba(X_test)[:, 1]
# submit = pd.DataFrame({"row_id": X_test.index, "prob": proba})

submit.to_csv(r"C:\Users\saout\Documents\Data-Challenge-Apprentissage-Supervis-\submission_test.csv", index=False)