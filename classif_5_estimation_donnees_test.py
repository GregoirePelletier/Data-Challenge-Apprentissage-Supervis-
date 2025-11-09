import pandas as pd
from pathlib import Path

nom_modele = "2"

fichier = Path(r"C:\Users\saout\Documents\Data-Challenge-Apprentissage-Supervis-\data\test_data.csv")
df_test = pd.read_csv(fichier)

df_test.drop(columns=["Unnamed: 0"], axis=1, inplace=True)


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
# features ingeniering

### lead_time
df_test["lead_time_tronq"] = df_test["lead_time"].clip(upper=df_test["lead_time"].quantile(0.99))
df_test["lead_time_log"] = np.log1p(df_test["lead_time"])


### market_segment et country
df_test.loc[~df_test["market_segment"].isin(["Online TA","Offline TA/TO","Groups","Direct","Corporate"]), "market_segment"] = "Other"

top_countries = ["PRT","GBR","FRA","ESP","DEU","ITA","IRL","BEL","BRA","USA","NLD","CHE","CN","AUT"]
df_test.loc[~df_test["country"].isin(top_countries), "country"] = "Other"

### ADR
df_test["party_size"] = df_test["adults"] + df_test["children"].fillna(0) + df_test["babies"].fillna(0)
df_test["adr_per_person"] = df_test["adr"] / df_test["party_size"].replace(0, np.nan)
df_test["adr_per_person"] = df_test["adr_per_person"].fillna(0)


### ratio annulation
total_prev = df_test["previous_cancellations"] + df_test["previous_bookings_not_canceled"]
df_test["prev_cancel_ratio"] = np.where(total_prev > 0,
                                   df_test["previous_cancellations"] / total_prev,
                                   0)

### arrival_date_month
month_map = {
    "January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
    "July":7,"August":8,"September":9,"October":10,"November":11,"December":12
}
df_test["month_num"] = df_test["arrival_date_month"].map(month_map)

# sin/cos pour la saisonnalité
df_test["month_sin"] = np.sin(2 * np.pi * df_test["month_num"] / 12)
df_test["month_cos"] = np.cos(2 * np.pi * df_test["month_num"] / 12)

# Saisonnalité hebdo (ISO week 1–53)
w = df_test["arrival_date_week_number"].astype(int).clip(1, 53)
df_test["week_sin"] = np.sin(2 * np.pi * w / 52)
df_test["week_cos"] = np.cos(2 * np.pi * w / 52)

# 
df_test["room_changed"] = (df_test["reserved_room_type"] != df_test["assigned_room_type"]).astype(int)



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


df_test = df_test[liste_var_categ + liste_var_quanti + ["row_id"]] 


df_test_c = pd.get_dummies(data = df_test, \
                         prefix = liste_var_categ, \
                         columns = liste_var_categ)



df_test_c = df_test_c.set_index("row_id").reindex(columns=X_train.columns, fill_value=0)




y_pred = random_forest_model.predict(df_test_c[var])




submit = pd.DataFrame({
    "row_id": df_test_c.index,                    # récupère l’index
    "prediction": y_pred
})

# proba = pipeline.predict_proba(X_test)[:, 1]
# submit = pd.DataFrame({"row_id": X_test.index, "prob": proba})

submit.to_csv(r"C:\Users\saout\Documents\Data-Challenge-Apprentissage-Supervis-\sorties\{nom_modele}.csv", index=False)