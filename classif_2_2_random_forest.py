# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 15:40:32 2025

@author: saout
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from _export_model_report_pdf import export_model_report_pdf

# min_samples_leaf : nombre minimum d’échantillons par feuille

min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700]                                          
max_depth_list = [2, 4, 8, 16, 32, 64, None]
n_estimators_list = [10,50,100,500]

#______________________________________________________________________________
#______________________________________________________________________________
# boucle sur  min_samples_split : nombre minimal d’échantillons requis dans un nœud pour pouvoir le diviser (splitter).

f1_list_train = []
f1_list_test = []
for min_samples_split in min_samples_split_list:
    model = RandomForestClassifier(min_samples_split = min_samples_split,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_test = model.predict(X_test) ## The predicted values for the test dataset
    f1_train = f1_score(predictions_train,y_train,average="macro")
    f1_test = f1_score(predictions_test,y_test,average="macro")
    f1_list_train.append(f1_train)
    f1_list_test.append(f1_test)

plt.title('Train x Test metrics')
plt.xlabel('min_samples_split')
plt.ylabel('f1')
plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list) 
plt.plot(f1_list_train)
plt.plot(f1_list_test)
plt.legend(['Train','Test'])
plt.show()

 # meilleur compromis : min_samples_split = 2

#______________________________________________________________________________
#______________________________________________________________________________
# boucle sur  max_depth : profondeur max des arbres

f1_list_train = []
f1_list_test = []
for max_depth in max_depth_list:
    model = RandomForestClassifier(max_depth = max_depth,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_test = model.predict(X_test) ## The predicted values for the test dataset
    f1_train = f1_score(y_train,predictions_train,average="macro")
    f1_test = f1_score(y_test,predictions_test,average="macro")
    f1_list_train.append(f1_train)
    f1_list_test.append(f1_test)

plt.title('Train x Test metrics')
plt.xlabel('max_depth')
plt.ylabel('f1')
plt.xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
plt.plot(f1_list_train)
plt.plot(f1_list_test)
plt.legend(['Train','Test'])
plt.show()

# meilleur compromis 64

#______________________________________________________________________________
#______________________________________________________________________________
# boucle sur  n_estimators

f1_list_train = []
f1_list_test = []
for n_estimators in n_estimators_list:
    model = RandomForestClassifier(n_estimators = n_estimators,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_test = model.predict(X_test) ## The predicted values for the test dataset
    f1_train = f1_score(y_train,predictions_train,average="macro")
    f1_test = f1_score(y_test,predictions_test,average="macro")
    f1_list_train.append(f1_train)
    f1_list_test.append(f1_test)

plt.title('Train x Test metrics')
plt.xlabel('n_estimators')
plt.ylabel('f1')
plt.xticks(ticks = range(len(n_estimators_list )),labels=n_estimators_list)
plt.plot(f1_list_train)
plt.plot(f1_list_test)
plt.legend(['Train','Test'])
plt.show()


#______________________________________________________________________________
#______________________________________________________________________________
# MODELE SELECTIONNE

model_name = "classif_2_2_random_forest_mod"


random_forest_model = RandomForestClassifier(n_estimators = 200,
                                             max_depth = 64, 
                                             max_features = "sqrt",
                                             min_samples_split = 2,
                                             min_samples_leaf = 1).fit(X_train,y_train)


print(f"Metrics train:\n\tf1: {f1_score(y_train, random_forest_model.predict(X_train), average='macro'):.4f}\n"
      f"Metrics test:\n\tf1 score: {f1_score(y_test, random_forest_model.predict(X_test), average='macro'):.4f}")

# === Prédictions ===

predictions_train = random_forest_model.predict(X_train) ## The predicted values for the train dataset
predictions_test = random_forest_model.predict(X_test) ## The predicted values for the test dataset

# === Matrice de confusion brute ===

cm = confusion_matrix(y_test, predictions_test)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=random_forest_model.classes_)
disp.plot(cmap="Greens")
plt.show()

# === Export Resultat ===

f1, acc, name, _ = export_model_report_pdf(random_forest_model, X_test, y_test, pdf_path= chemin_sortie+"\\"+model_name+".pdf", 
                                       title = "random_forest_model")

print("f1:", f1, "| Modèle:", model_name)
print("acc:", acc, "| Modèle:", model_name)