# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 18:45:35 2025

@author: saout
"""


from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import xgboost
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from _export_model_report_pdf import export_model_report_pdf

n_estimators_list = [10,50,100,200,300,500,700,1000,1400,1800,2200,2600,3000,4000,5000]
n_learning_rate_list = [0.02, 0.05,0.1,0.3,0.9]
max_depth_list = [3,4,6,8,10]


def f1_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    K = int(dtrain.num_col() / (len(np.unique(y_true)) or 1))  # ou passe K connu
    # plus simple si tu connais K :
    # K = 3
    y_hat = y_pred.reshape(-1, 3).argmax(axis=1)  # 3 classes ici
    from sklearn.metrics import f1_score
    return "f1_macro", f1_score(y_true, y_hat, average="macro")

#______________________________________________________________________________
#______________________________________________________________________________


n = int(len(X_train)*0.8) 

X_train_fit, X_train_eval, y_train_fit, y_train_eval = X_train[:n], X_train[n:], y_train[:n], y_train[n:]

print(XGBClassifier.__module__)

#______________________________________________________________________________
#______________________________________________________________________________

f1_list_train = []
f1_list_test = []
for n_learning_rate in n_learning_rate_list:
    xgb_model = xgboost.XGBClassifier(learning_rate = n_learning_rate, random_state=RANDOM_STATE, objective="multi:softprob", num_class=3, eval_metric="mlogloss" )
    xgb_model.fit( X_train_fit, y_train_fit, eval_set=[(X_train_eval, y_train_eval)],early_stopping_rounds=50, verbose=True)
    predictions_train = xgb_model.predict(X_train) ## The predicted values for the train dataset
    predictions_test = xgb_model.predict(X_test) ## The predicted values for the test dataset
    f1_train = f1_score(y_train,predictions_train,average="macro")
    f1_test = f1_score(y_test,predictions_test,average="macro")
    f1_list_train.append(f1_train)
    f1_list_test.append(f1_test)

plt.title('Train x Test metrics')
plt.xlabel('min_samples_split')
plt.ylabel('f1')
plt.xticks(ticks = range(len(n_learning_rate_list )),labels=n_learning_rate_list) 
plt.plot(f1_list_train)
plt.plot(f1_list_test)
plt.legend(['Train','Test'])
plt.show()

# meilleur choix 0.3

#______________________________________________________________________________
#______________________________________________________________________________

f1_list_train = []
f1_list_test = []
for max_depth in max_depth_list:
    xgb_model = xgboost.XGBClassifier(max_depth = max_depth, random_state=RANDOM_STATE, objective="multi:softprob", num_class=3, eval_metric="mlogloss" )
    xgb_model.fit( X_train_fit, y_train_fit, eval_set=[(X_train_eval, y_train_eval)],early_stopping_rounds=50, verbose=True)

    predictions_train = xgb_model.predict(X_train) 
    predictions_test = xgb_model.predict(X_test) 
    f1_train = f1_score(y_train,predictions_train,average="macro")
    f1_test = f1_score(y_test,predictions_test,average="macro")
    f1_list_train.append(f1_train)
    f1_list_test.append(f1_test)

plt.title('Train x Test metrics')
plt.xlabel('min_samples_split')
plt.ylabel('f1')
plt.xticks(ticks = range(len(max_depth_list )),labels=max_depth_list) 
plt.plot(f1_list_train)
plt.plot(f1_list_test)
plt.legend(['Train','Test'])
plt.show()

#______________________________________________________________________________
#______________________________________________________________________________
# MODELE SELECTIONNE

xgb_model = xgboost.XGBClassifier(
    earning_rate = 0.3,
    max_depth = 10,
    random_state=RANDOM_STATE,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss"     # 👉 early stopping basé sur log-loss
)

xgb_model.fit(
    X_train_fit, y_train_fit,
    eval_set=[(X_train_eval, y_train_eval)],
    early_stopping_rounds=50,
    verbose=True
)


print(f"Metrics train:\n\tf1: {f1_score(y_train, xgb_model.predict(X_train), average='macro'):.4f}\n"
      f"Metrics test:\n\tf1 score: {f1_score(y_test, xgb_model.predict(X_test), average='macro'):.4f}")

# === Prédictions ===

predictions_train = xgb_model.predict(X_train) 
predictions_test = xgb_model.predict(X_test)

# === Matrice de confusion brute ===

cm = confusion_matrix(y_test, predictions_test)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=xgb_model.classes_)
disp.plot(cmap="Greens")
plt.show()

# === Export Resultat ===

f1, acc, name, _ = export_model_report_pdf(xgb_model, X_test, y_test, pdf_path= chemin_sortie+"\\classif_2_3_xgb_model.pdf", 
                                       title = "xgb_model")

print("f1:", f1, "| Modèle:", name)