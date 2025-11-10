# Classif 2.1 : Arbre de décision (entraienment & ptimisation)
#               Optimisation sur min_samples_split & max_depth

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from _export_model_report_pdf import export_model_report_pdf
import numpy as np


min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700] ## If the number is an integer, then it is the actual quantity of samples,
max_depth_list = [1,2, 3, 4, 8, 16, 32, 64, None] # None means that there is no depth limit.

#______________________________________________________________________________
#______________________________________________________________________________
# Boucle sur  min_samples_split

f1_list_train = []
f1_list_test = []
for min_samples_split in min_samples_split_list:
    model = DecisionTreeClassifier(min_samples_split = min_samples_split,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)
    f1_train = f1_score(y_train, predictions_train, average="weighted")   
    f1_test = f1_score(y_test, predictions_test, average="weighted")      
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

#______________________________________________________________________________
#______________________________________________________________________________
# boucle sur  max_depth

f1_list_train = []
f1_list_test = []
for max_depth in max_depth_list:
    model = DecisionTreeClassifier(max_depth = max_depth,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)
    f1_train = f1_score(y_train, predictions_train, average="weighted")   
    f1_test = f1_score(y_test, predictions_test, average="weighted")      
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

#______________________________________________________________________________
#______________________________________________________________________________
# MODELE SELECTIONNE

decision_tree_model = DecisionTreeClassifier(min_samples_split = 10,
                                             max_depth = 16,
                                             random_state = RANDOM_STATE).fit(X_train,y_train)

print(f"Metrics train:\n\tf1: {f1_score(y_train,decision_tree_model.predict(X_train),average="weighted"):.4f}\nMetrics test:\n\tf1 score: {f1_score(y_test,decision_tree_model.predict(X_test),average="weighted"):.4f}")
#print(f"Metrics train:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_train),y_train):.4f}\nMetrics test:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_test),y_test):.4f}")

imp_vars = pd.Series(decision_tree_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# === Prédictions ===

predictions_train = decision_tree_model.predict(X_train) ## The predicted values for the train dataset
predictions_test = decision_tree_model.predict(X_test) ## The predicted values for the test dataset

# === Matrice de confusion brute ===

cm = confusion_matrix(y_test, predictions_test)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=decision_tree_model.classes_)
disp.plot(cmap="Greens")
plt.show()

# === Export Resultat ===

f1, acc, name, _ = export_model_report_pdf(decision_tree_model,X_test, y_test,
    pdf_path=str(chemin_sortie / "classif_2_1_decision_tree_model.pdf"),
    title="decision_tree_model"
)

print("f1:", f1, "| Modèle:", name)









