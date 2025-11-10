import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
import numpy as np

# Ensure OpenMP compatibility for joblib/sklearn
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Global Parameters (from classif_0_parametrage.py)
RANDOM_STATE = 55
CHEMIN_SORTIE = Path("C:/Users/saout/Documents/Data-Challenge-Apprentissage-Supervis-/sorties")
DATA_PATH = Path("C:/Users/saout/Documents/Data-Challenge-Apprentissage-Supervis-/data")

def load_and_preprocess_data(file_name="train_data.csv"):
    """
    Loads data and performs initial feature engineering and one-hot encoding.
    Based on classif_0_parametrage.py.
    """
    file_path = DATA_PATH / file_name
    df = pd.read_csv(file_path)

    if "row_id" in df.columns:
        df.set_index('row_id', inplace=True)
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], axis=1, inplace=True)

    # Define initial feature lists
    liste_var_categ = [
        "is_repeated_guest", "hotel", "meal", "country", "market_segment",
        "distribution_channel", "reserved_room_type", "assigned_room_type",
        "deposit_type", "customer_type", "arrival_date_month",
    ]
    liste_var_quanti = [
        'lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights',
        'adults', 'children', 'babies', 'previous_cancellations',
        'previous_bookings_not_canceled', 'booking_changes', 'days_in_waiting_list',
        'adr', 'required_car_parking_spaces', 'total_of_special_requests',
        "arrival_date_week_number", "arrival_date_day_of_month", "arrival_date_year",
    ]
    target = "reservation_status"

    # Feature Engineering
    df["lead_time_tronq"] = df["lead_time"].clip(upper=df["lead_time"].quantile(0.99))
    df["lead_time_log"] = np.log1p(df["lead_time"])

    df.loc[~df["market_segment"].isin(["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate"]), "market_segment"] = "Other"
    top_countries = ["PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "IRL", "BEL", "BRA", "USA", "NLD", "CHE", "CN", "AUT"]
    df.loc[~df["country"].isin(top_countries), "country"] = "Other"

    df["party_size"] = df["adults"] + df["children"].fillna(0) + df["babies"].fillna(0)
    df["adr_per_person"] = df["adr"] / df["party_size"].replace(0, np.nan)
    df["adr_per_person"] = df["adr_per_person"].fillna(0)

    total_prev = df["previous_cancellations"] + df["previous_bookings_not_canceled"]
    df["prev_cancel_ratio"] = np.where(total_prev > 0, df["previous_cancellations"] / total_prev, 0)

    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
    }
    df["month_num"] = df["arrival_date_month"].map(month_map)
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)

    w = df["arrival_date_week_number"].astype(int).clip(1, 53)
    df["week_sin"] = np.sin(2 * np.pi * w / 52)
    df["week_cos"] = np.cos(2 * np.pi * w / 52)

    df["room_changed"] = (df["reserved_room_type"] != df["assigned_room_type"]).astype(int)

    # Updated feature lists after FE
    liste_var_categ = [
        "is_repeated_guest", "hotel", "meal", "country",
        "market_segment", "distribution_channel",
        "reserved_room_type", "assigned_room_type",
        "room_changed",
        "deposit_type", "customer_type"
    ]
    liste_var_quanti = [
        "lead_time_log",
        "stays_in_weekend_nights", "stays_in_week_nights",
        "adults", "children", "babies",
        "previous_cancellations", "previous_bookings_not_canceled",
        "booking_changes", "days_in_waiting_list",
        "adr_per_person",
        "required_car_parking_spaces", "total_of_special_requests",
        "arrival_date_day_of_month", "arrival_date_year",
        "month_sin", "month_cos", "week_sin", "week_cos",
        "party_size", "prev_cancel_ratio"
    ]

    # Select features and apply one-hot encoding
    if file_name == "train_data.csv":
        df_processed = df[liste_var_categ + liste_var_quanti + [target]].copy()
    else:
        df_processed = df[liste_var_categ + liste_var_quanti].copy()

    df_encoded = pd.get_dummies(data=df_processed, prefix=liste_var_categ, columns=liste_var_categ)

    return df_encoded, target, liste_var_categ, liste_var_quanti

def train_test_split_data(df_encoded, target_col, random_state=RANDOM_STATE):
    """
    Splits the encoded DataFrame into training and testing sets.
    """
    var = [x for x in df_encoded.columns if x != target_col]
    X = df_encoded[var]
    y = df_encoded[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=random_state)
    return X_train, X_test, y_train, y_test, var

def run_pipeline_standard_models(X_train, X_test, y_train, y_test, feature_cols):
    """
    Executes the standard model pipeline (0 -> 2 -> 5).
    This will use the RandomForestClassifier from classif_2_2_random_forest.py as an example.
    """
    from src.models.random_forest_model import train_random_forest, evaluate_model as evaluate_rf_model
    from src.models.xgboost_model import train_xgboost, evaluate_model as evaluate_xgb_model
    from src.utils._export_model_report_pdf import export_model_report_pdf

    print("\n--- Running Standard Model Pipeline ---")

    # Train Random Forest Model
    print("\n--- Training Random Forest Model ---")
    rf_model, rf_f1, rf_params = train_random_forest(X_train, y_train, X_test, y_test, output_dir=CHEMIN_SORTIE)
    print(f"Random Forest Best F1: {rf_f1:.4f}")
    print("Random Forest Best Params:", rf_params)

    # Evaluate Random Forest Model
    pdf_output_path_rf = CHEMIN_SORTIE / "random_forest_model_report.pdf"
    evaluate_rf_model(rf_model, X_test, y_test, title="Random Forest Model Evaluation", output_path=pdf_output_path_rf)
    
    # Predictions on test data for Random Forest
    df_test_raw = pd.read_csv(DATA_PATH / "test_data.csv")
    df_test_processed, _, _, _ = load_and_preprocess_data(file_name="test_data.csv")
    df_test_aligned = df_test_processed.set_index(df_test_raw['row_id']).reindex(columns=X_train.columns, fill_value=0)
    y_pred_rf = rf_model.predict(df_test_aligned)

    submit_rf = pd.DataFrame({
        "row_id": df_test_aligned.index,
        "prediction": y_pred_rf
    })
    output_path_rf_submission = CHEMIN_SORTIE / "submission_standard_rf.csv"
    submit_rf.to_csv(output_path_rf_submission, index=False)
    print(f"Standard Random Forest predictions saved to {output_path_rf_submission}")

    # Train XGBoost Model (as an alternative standard model)
    print("\n--- Training XGBoost Model ---")
    xgb_model, xgb_f1, xgb_params = train_xgboost(X_train, y_train, X_test, y_test, output_dir=CHEMIN_SORTIE)
    print(f"XGBoost Best F1: {xgb_f1:.4f}")
    print("XGBoost Best Params:", xgb_params)

    # Evaluate XGBoost Model
    pdf_output_path_xgb = CHEMIN_SORTIE / "xgboost_model_report.pdf"
    evaluate_xgb_model(xgb_model, X_test, y_test, title="XGBoost Model Evaluation", output_path=pdf_output_path_xgb)

    # Predictions on test data for XGBoost
    y_pred_xgb = xgb_model.predict(df_test_aligned)

    submit_xgb = pd.DataFrame({
        "row_id": df_test_aligned.index,
        "prediction": y_pred_xgb
    })
    output_path_xgb_submission = CHEMIN_SORTIE / "submission_standard_xgb.csv"
    submit_xgb.to_csv(output_path_xgb_submission, index=False)
    print(f"Standard XGBoost predictions saved to {output_path_xgb_submission}")


def run_pipeline_expert_models(df_full, X_train, X_test, y_train, y_test, liste_var_quanti, liste_var_categ):
    """
    Executes the expert model pipeline (0 -> 3 -> 4 -> 5_experts).
    """
    import prince
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import ParameterSampler

    print("\n--- Running Expert Model Pipeline ---")

    # --- Step 3: Clustering (FAMD + KMeans) ---
    cols_quanti = liste_var_quanti
    cols_quali = liste_var_categ

    # Ensure df_full has the necessary columns and types for FAMD
    X_famd_input = df_full[cols_quanti + cols_quali].copy()
    for c in cols_quanti:
        X_famd_input[c] = pd.to_numeric(X_famd_input[c], errors="coerce").fillna(0)
    for c in cols_quali:
        X_famd_input[c] = X_famd_input[c].astype("category")

    famd = prince.FAMD(n_components=6, random_state=RANDOM_STATE)
    famd.fit(X_famd_input)
    coord = famd.row_coordinates(X_famd_input)

    scaler_famd = StandardScaler().fit(coord)
    Z = scaler_famd.transform(coord)

    km_route = KMeans(n_clusters=3, n_init=30, random_state=RANDOM_STATE)
    km_route.fit(Z)

    # Save clustering models
    joblib.dump(famd, CHEMIN_SORTIE / "famd.pkl")
    joblib.dump(scaler_famd, CHEMIN_SORTIE / "scaler_famd.pkl")
    joblib.dump(km_route, CHEMIN_SORTIE / "km_route.pkl")
    joblib.dump(X_train.columns.tolist(), CHEMIN_SORTIE / "ohe_columns.pkl") # for test data alignment

    # Assign clusters to the full DataFrame
    df_full.loc[X_famd_input.index, "cluster_famd"] = km_route.labels_.astype(int)
    
    # Apply the rule: cluster 2 -> 0
    df_full["cluster_famd"] = np.where(df_full["cluster_famd"] == 2, 0, df_full["cluster_famd"])

    # --- Step 4: Expert Model Training ---
    c_tr = df_full.loc[X_train.index, "cluster_famd"].astype(int)
    c_te = df_full.loc[X_test.index, "cluster_famd"].astype(int)

    grid = {
        "n_estimators": [200, 400, 800, 1200],
        "max_depth": [None, 6, 10, 16, 24, 32],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.3, 0.5, 0.8],
        "bootstrap": [True, False]
    }

    models = {}
    for k in sorted(c_tr.unique()): # Iterate through unique clusters
        tr_idx = c_tr[c_tr == k].index
        Xtr, ytr = X_train.loc[tr_idx], y_train.loc[tr_idx]

        if len(tr_idx) == 0:
            print(f"Skipping cluster {k}: no training samples.")
            continue

        best, best_f1 = None, -1
        print(f"Training expert model for cluster {k}...")
        for p in ParameterSampler(grid, n_iter=3, random_state=RANDOM_STATE): # Further reduced n_iter
            rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced", **p)
            rf.fit(Xtr, ytr)
            # Evaluate on a subset of X_test belonging to this cluster, if available
            te_idx_k = c_te[c_te == k].index
            if len(te_idx_k) > 0:
                f1 = f1_score(y_test.loc[te_idx_k], rf.predict(X_test.loc[te_idx_k]), average="macro")
                if f1 > best_f1:
                    best_f1, best = f1, rf
            else: # If no test samples for this cluster, just pick the last trained model
                best_f1, best = 0.0, rf # Assign a default F1 if no test data for evaluation
        models[k] = best
        print(f"Cluster {k} — meilleur F1_macro: {best_f1:.3f}")
        joblib.dump(best, CHEMIN_SORTIE / f"rf_cluster{k}.pkl")

    # --- Step 5_experts: Prediction on Test Data ---
    df_test_raw = pd.read_csv(DATA_PATH / "test_data.csv")
    df_test_processed, _, _, _ = load_and_preprocess_data(file_name="test_data.csv")
    df_test_aligned = df_test_processed.set_index(df_test_raw['row_id']).reindex(columns=X_train.columns, fill_value=0)

    # Re-apply FE for FAMD input on test data
    df_famd_input_test = df_test_raw[cols_quanti + cols_quali].copy()
    for c in cols_quanti:
        df_famd_input_test[c] = pd.to_numeric(df_famd_input_test[c], errors="coerce").fillna(0.0).astype("float64")
    for c in cols_quali:
        # Ensure categories are aligned with training FAMD
        if c in famd.cat_cols_:
            idx = famd.cat_cols_.get_loc(c)
            allowed_categories = pd.Index(famd.cat_scaler_.categories_[idx])
            col = df_famd_input_test[c].astype(object)
            fb = "Other" if "Other" in allowed_categories else allowed_categories[0]
            col = col.where(col.notna(), fb)
            col = col.where(col.isin(allowed_categories), fb)
            df_famd_input_test[c] = pd.Categorical(col, categories=allowed_categories)
        else:
            df_famd_input_test[c] = df_famd_input_test[c].astype("category")

    df_famd_input_test.index = df_test_raw["row_id"].values

    coord_test = famd.row_coordinates(df_famd_input_test)
    Z_test = pd.DataFrame(scaler_famd.transform(coord_test), index=df_famd_input_test.index)
    clusters_test = km_route.predict(Z_test).astype(int)
    clusters_test = np.where(clusters_test == 2, 0, clusters_test) # Apply 2->0 rule

    pred_s = pd.Series(0, index=df_test_aligned.index, dtype=int)

    for k in models.keys():
        idx_k = df_test_aligned.index[clusters_test == k]
        if len(idx_k) > 0:
            pred_s.loc[idx_k] = models[k].predict(df_test_aligned.loc[idx_k])

    output_path = CHEMIN_SORTIE / "submission_experts.csv"
    submit = pd.DataFrame({"row_id": pred_s.index, "prediction": pred_s.values})
    submit.to_csv(output_path, index=False)
    print(f"Expert model predictions saved to {output_path}")


def main(pipeline_type="standard"):
    """
    Main function to run the selected pipeline.
    pipeline_type: "standard" or "experts"
    """
    print(f"Starting pipeline: {pipeline_type}")

    # Load and preprocess training data
    df_encoded_train, target_col, liste_var_categ, liste_var_quanti = load_and_preprocess_data(file_name="train_data.csv")
    X_train, X_test, y_train, y_test, feature_cols = train_test_split_data(df_encoded_train, target_col)

    # To get the original df with cluster_famd for expert pipeline
    df_full_original, _, _, _ = load_and_preprocess_data(file_name="train_data.csv")
    
    if pipeline_type == "standard":
        run_pipeline_standard_models(X_train, X_test, y_train, y_test, feature_cols)
    elif pipeline_type == "experts":
        run_pipeline_expert_models(df_full_original, X_train, X_test, y_train, y_test, liste_var_quanti, liste_var_categ)
    else:
        print("Invalid pipeline type. Choose 'standard' or 'experts'.")

if __name__ == "__main__":
    # To run the standard pipeline by default:
    main(pipeline_type="standard")

    # Uncomment the line below to run the expert models pipeline:
    # main(pipeline_type="experts")