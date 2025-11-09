# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 09:56:08 2025

@author: saout
"""


import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
from datetime import datetime
from _plot_cat_vs_quants import plot_cat_vs_quants
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
import numpy as np


#______________________________________________________________________________
#______________________________________________________________________________
# Entrees / sorties

chemin_sortie = "C:\\Users\\saout\\Documents\\Data-Challenge-Apprentissage-Supervis-\\sorties"

def load_train():
    fichier = Path(r"C:\Users\saout\Documents\Data-Challenge-Apprentissage-Supervis-\data\train_data.csv")
    return pd.read_csv(fichier)

#______________________________________________________________________________
#______________________________________________________________________________
# Variables GLobales

RANDOM_STATE = 55

liste_var_categ = [   "is_repeated_guest",
                        "hotel",
                        "meal", 
                        "country", 
                        "market_segment", 
                        "distribution_channel", 
                        "reserved_room_type", 
                        "assigned_room_type", 
                        "deposit_type", 
                        "customer_type", 
                        "arrival_date_month",
                       
                        ]

liste_var_quanti = 	[
        		'lead_time', 
                'stays_in_weekend_nights', 
        	  'stays_in_week_nights' ,
        	    'adults', 
        	    'children',
                'babies',   
        	    'previous_cancellations',
               'previous_bookings_not_canceled',   
        	    'booking_changes',  
                'days_in_waiting_list', 
        	    'adr',
                'required_car_parking_spaces', 
        	    'total_of_special_requests',
                 "arrival_date_week_number",
                        "arrival_date_day_of_month",
                        "arrival_date_year",]

target = "reservation_status"
#______________________________________________________________________________
#______________________________________________________________________________

# Import fichiers

df = load_train()
print(df.columns)

df.set_index('row_id', inplace=True)
print(df.head()) 

df.drop(columns=["Unnamed: 0"], axis=1, inplace=True)




# Comptage des catégories variable cible
counts = df[target].value_counts()

# Graphique pie
counts.plot.pie(autopct="%1.1f%%", figsize=(6,6), ylabel="")
plt.show()
#Check-Out (0), Canceled (1), ou No-Show (2)

#______________________________________________________________________________
#______________________________________________________________________________
# features ingeniering

### lead_time
df["lead_time_tronq"] = df["lead_time"].clip(upper=df["lead_time"].quantile(0.99))
df["lead_time_log"] = np.log1p(df["lead_time"])


### market_segment et country
df.loc[~df["market_segment"].isin(["Online TA","Offline TA/TO","Groups","Direct","Corporate"]), "market_segment"] = "Other"

top_countries = ["PRT","GBR","FRA","ESP","DEU","ITA","IRL","BEL","BRA","USA","NLD","CHE","CN","AUT"]
df.loc[~df["country"].isin(top_countries), "country"] = "Other"

### ADR
df["party_size"] = df["adults"] + df["children"].fillna(0) + df["babies"].fillna(0)
df["adr_per_person"] = df["adr"] / df["party_size"].replace(0, np.nan)
df["adr_per_person"] = df["adr_per_person"].fillna(0)


### ratio annulation
total_prev = df["previous_cancellations"] + df["previous_bookings_not_canceled"]
df["prev_cancel_ratio"] = np.where(total_prev > 0,
                                   df["previous_cancellations"] / total_prev,
                                   0)

### arrival_date_month
month_map = {
    "January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
    "July":7,"August":8,"September":9,"October":10,"November":11,"December":12
}
df["month_num"] = df["arrival_date_month"].map(month_map)

# sin/cos pour la saisonnalité
df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

# Saisonnalité hebdo (ISO week 1–53)
w = df["arrival_date_week_number"].astype(int).clip(1, 53)
df["week_sin"] = np.sin(2 * np.pi * w / 52)
df["week_cos"] = np.cos(2 * np.pi * w / 52)

# 
df["room_changed"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)


liste_var_categ = [
    "is_repeated_guest", "hotel", "meal", "country",
    "market_segment", "distribution_channel",
    "reserved_room_type", "assigned_room_type",
    "room_changed",
    "deposit_type", "customer_type"
]

liste_var_quanti = [
    #"lead_time_tronq", 
    "lead_time_log",
    "stays_in_weekend_nights", "stays_in_week_nights",
    "adults", "children", "babies",
    "previous_cancellations", "previous_bookings_not_canceled",
    "booking_changes", "days_in_waiting_list", #"adr",
      "adr_per_person",
    "required_car_parking_spaces", "total_of_special_requests",
    "arrival_date_day_of_month", "arrival_date_year",
    "month_sin", "month_cos", "week_sin", "week_cos",
    "party_size", "prev_cancel_ratio"
]

#______________________________________________________________________________
#______________________________________________________________________________
# dummies categ + echantillonage train/test

df = df[liste_var_categ + liste_var_quanti + ["reservation_status"]]

df_c = pd.get_dummies(data = df, \
                         prefix = liste_var_categ, \
                         columns = liste_var_categ)


var = [x for x in df_c.columns if x != target] ## Removing our target variable

X_train, X_test, y_train, y_test = train_test_split(df_c[var], df[target],\
                train_size = 0.8, random_state = RANDOM_STATE)


print(f'train samples: {len(X_train)}\ntest samples: {len(X_test)}')
print(f'target proportion train : {sum(y_train)/len(y_train):.4f}')
print(f'target proportion test : {sum(y_test)/len(y_test):.4f}')






'''
df["country"].value_counts()


df["country"].value_counts().head(10)
(df["country"].value_counts(normalize=True) * 100).round(2)

df["is_portuguese"] = np.where(df["country"] == "PRT", "Portugal", "Étranger")
df["is_portuguese"].value_counts(normalize=True) * 100

import matplotlib.pyplot as plt

df["country"].value_counts().head(10).plot(kind="bar", color="steelblue")
plt.title("Top 10 pays d'origine des clients")
plt.ylabel("Nombre de réservations")
plt.xlabel("Pays")
plt.show()
'''



