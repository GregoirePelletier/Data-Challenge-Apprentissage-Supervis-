# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 14:24:29 2025

@author: saout
"""



import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency
import pandas as pd
import matplotlib.pyplot as plt


#______________________________________________________________________________
#______________________________________________________________________________


df = load_train()
df.set_index('row_id', inplace=True)

###
sns.pairplot(df.loc[:,liste_var_quanti])

df_corr = df.loc[:,liste_var_quanti].corr()

ax = sns.heatmap(df_corr, xticklabels = df_corr.columns , \
                 yticklabels = df_corr.columns, cmap = 'coolwarm')


# Test indépendance entre deux variables quantitatives / Test de corrélation Pearson
# pearson variables quantitatives

# H0 : Variables indépendantes si p-value > 5%~
# H1 : Variables non indépendantes si p-value < 5%

a = np.empty((len(liste_var_quanti), len(liste_var_quanti)))
a[:] = np.nan
for i, ii in enumerate(liste_var_quanti):
    for j, jj in enumerate(liste_var_quanti):
        a[i, j] = pearsonr(df[ii], df[jj])[1]   # p-value (indice [1])

df_pvalue = pd.DataFrame(a, columns=liste_var_quanti, index=liste_var_quanti).round(5)

cm = sns.light_palette("green", as_cmap=True) 

plt.figure()
sns.heatmap(df_pvalue, annot=False)  # enlève annot si tu veux que des couleurs
plt.title("P-values (Pearson)")
plt.show()



#______________________________________________________________________________
#______________________________________________________________________________

# Test d'indépendance entre deux variables qualitatives / Test du Chi²

# df_count = pd.crosstab(df., df.)
# df_count

# Khi2_obs, p_value, ddl, effectif_theorique = chi2_contingency(df_count)

# from scipy.stats import chi2
# J = df = np.arange(1,5,1)
# I = np.arange(0.05,0.15,0.005)

# a = np.empty((len(J),len(I)))
# a[:] = np.nan

# for i in range(0,len(I)):
#     for j in range(0,len(J)):
#         a[j,i] = chi2.isf(I[i], J[j])
        
# df_chi2 = round(pd.DataFrame(a, columns=I, index = J),5)
# df_chi2

#______________________________________________________________________________
#______________________________________________________________________________

# Test d'indépendance entre une variable qualitative et une quantitative 
# / Test de Fisher avec l'analyse de la variance (ANOVA)

# plt.subplots(figsize=(20,4))
# ax = sns.boxplot(x="Calories", y="Type", data=df)


# import statsmodels.api as sm
# from statsmodels.formula.api import ols
# model = ols('Calories ~ Type', data=df).fit()
# anova_table = sm.stats.anova_lm(model, typ=2)
# anova_table



# bartlett(df.Calories[df.Type == 'Beef'],
#         df.Calories[df.Type == 'Meat'],
#         df.Calories[df.Type == 'Poultry'])


#     H0 : Les variances de chaque groupe sont égales si p-value > 5%
#     ~H1 : Les variances de chaque groupe ne sont pas toutes égales < 5%~





