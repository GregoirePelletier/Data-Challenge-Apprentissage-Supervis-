# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 18:45:35 2025

@author: saout
"""


from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from _export_model_report_pdf import export_model_report_pdf


#______________________________________________________________________________
#______________________________________________________________________________

df = load_train()
df.set_index('row_id', inplace=True)

df_c = pd.get_dummies(data = df, \
                         prefix = liste_var_categ, \
                         columns = liste_var_categ)


var = [x for x in df_c.columns if x not in target] 

X_train, X_test, y_train, y_test = train_test_split(df_c[var], df[target],\
                train_size = 0.8, random_state = RANDOM_STATE)

    
#______________________________________________________________________________
#______________________________________________________________________________


n = int(len(X_train)*0.8) 

X_train_fit, X_train_eval, y_train_fit, y_train_eval = X_train[:n], X_train[n:], y_train[:n], y_train[n:]

print(XGBClassifier.__module__)

#______________________________________________________________________________
#______________________________________________________________________________

n_estimators_list = [10,50,100,200,300,500,700,1000,1400,1800,2200,2600,3000,4000,5000]

f1_list_train = []
f1_list_test = []
for n_estimators in n_estimators_list:
    xgb_model = XGBClassifier (n_estimators = n_estimators, learning_rate = 0.1,verbosity = 1, random_state = RANDOM_STATE)
    xgb_model.fit(X_train_fit,y_train_fit, eval_set = [(X_train_eval,y_train_eval)])#, early_stopping_rounds = 50)

    predictions_train = xgb_model.predict(X_train) ## The predicted values for the train dataset
    predictions_test = xgb_model.predict(X_test) ## The predicted values for the test dataset
    f1_train = f1_score(y_train,predictions_train,average="macro")
    f1_test = f1_score(y_test,predictions_test,average="macro")
    f1_list_train.append(f1_train)
    f1_list_test.append(f1_test)

plt.title('Train x Test metrics')
plt.xlabel('min_samples_split')
plt.ylabel('f1')
plt.xticks(ticks = range(len(n_estimators_list )),labels=n_estimators_list) 
plt.plot(f1_list_train)
plt.plot(f1_list_test)
plt.legend(['Train','Test'])
plt.show()

#______________________________________________________________________________
#______________________________________________________________________________

n_learning_rate_list = [0.02, 0.05,0.1,0.3,0.9]

f1_list_train = []
f1_list_test = []
for n_learning_rate in n_learning_rate_list:
    xgb_model = XGBClassifier (n_estimators = 500, learning_rate = n_learning_rate,verbosity = 1, random_state = RANDOM_STATE)
    xgb_model.fit(X_train_fit,y_train_fit, eval_set = [(X_train_eval,y_train_eval)])#, early_stopping_rounds = 50)

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

#______________________________________________________________________________
#______________________________________________________________________________

max_depth_list = [3,4,6,8,10]

f1_list_train = []
f1_list_test = []
for n_estmax_dmax_depthepthimators in max_depth_list:
    xgb_model = XGBClassifier (n_estimators = n_estimators, max_depth = max_depth ,verbosity = 1, random_state = RANDOM_STATE)
    xgb_model.fit(X_train_fit,y_train_fit, eval_set = [(X_train_eval,y_train_eval)])#, early_stopping_rounds = 50)

    predictions_train = xgb_model.predict(X_train) ## The predicted values for the train dataset
    predictions_test = xgb_model.predict(X_test) ## The predicted values for the test dataset
    f1_train = f1_score(y_train,predictions_train,average="macro")
    f1_test = f1_score(y_test,predictions_test,average="macro")
    f1_list_train.append(f1_train)
    f1_list_test.append(f1_test)

plt.title('Train x Test metrics')
plt.xlabel('min_samples_split')
plt.ylabel('f1')
plt.xticks(ticks = range(len(n_estimators_list )),labels=n_estimators_list) 
plt.plot(f1_list_train)
plt.plot(f1_list_test)
plt.legend(['Train','Test'])
plt.show()

#______________________________________________________________________________
#______________________________________________________________________________

xgb_model = XGBClassifier (n_estimators = 1000, learning_rate = 0.1 ,verbosity = 1, random_state = RANDOM_STATE)
xgb_model.fit(X_train_fit,y_train_fit, eval_set = [(X_train_eval,y_train_eval)])#, early_stopping_rounds = 50)


print(f"Metrics train:\n\tf1: {f1_score(y_train,xgb_model.predict(X_train),average="macro"):.4f}\nMetrics test:\n\tf1 score: {f1_score(y_test,xgb_model.predict(X_test),average="macro"):.4f}")


# === Prédictions ===

predictions_train = xgb_model.predict(X_train) ## The predicted values for the train dataset
predictions_test = xgb_model.predict(X_test) ## The predicted values for the test dataset

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