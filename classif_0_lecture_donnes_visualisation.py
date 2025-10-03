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

#______________________________________________________________________________
#______________________________________________________________________________
# Entrees / sorties

chemin_sortie = "C:\\Users\\saout\\Documents\\Data-Challenge-Apprentissage-Supervis\\sorties"

def load_train():
    fichier = Path(r"C:\Users\saout\Documents\Data-Challenge-Apprentissage-Supervis\data\train_data.csv")
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
                        "arrival_date_week_number",
                        "arrival_date_day_of_month",
                        "arrival_date_year"
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
        	    'total_of_special_requests',]

target = "reservation_status"

#______________________________________________________________________________
#______________________________________________________________________________

# Import fichiers

df = load_train()
print(df.columns)

df.set_index('row_id', inplace=True)
print(df.head()) 


# Comptage des catégories variable cible
counts = df[target].value_counts()

# Graphique pie
counts.plot.pie(autopct="%1.1f%%", figsize=(6,6), ylabel="")
plt.show()
#Check-Out (0), Canceled (1), ou No-Show (2)

#______________________________________________________________________________
#______________________________________________________________________________

# Visualisation des données

###############################
### VARIABLES CATÉGORIELLES ###
###############################


pdf_path = chemin_sortie+"\\classif_0_histplot_target.pdf"
with PdfPages(pdf_path) as pdf:
    # (facultatif) métadonnées
    info = pdf.infodict()
    info["Title"] = "Histplots vs cible"
    info["Author"] = "Ton script"
    info["Subject"] = f"Proportions de {target} par catégories"
    info["CreationDate"] = datetime.now()

    for x in liste_var_categ:
        d = df[[x, target]].dropna().copy()
        if d.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(
            data=d,
            x=x,
            hue=target,
            multiple="fill",
            stat="proportion",
            discrete=True,
            shrink=0.9,
            ax=ax
        )
        ax.set_ylabel("Proportion")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
        ax.set_title(f"{x} vs {target} — proportions (empilées)")
        fig.tight_layout()

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

print("PDF écrit :", pdf_path)

# zoom deposit_type
df['deposit_type'].unique()
eph = df.loc[ df['deposit_type']=='Non Refund',['deposit_type','reservation_status','customer_type','market_segment']]
# --> Non Refund sont presque toutes annulées


###############################
### VARIABLES QUANTITATIVES ###
###############################

              
plot_cat_vs_quants(df, target, quants=liste_var_quanti, pdf_path=chemin_sortie+"\\classif_0_cat_vs_quants.pdf",
                               order=None, palette="tab10", inner="quartile",
                               show_kde=True)

