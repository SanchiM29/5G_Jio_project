import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
def calculate_residuals(results_df):

    results_df = results_df.copy()
    results_df["residual"] = results_df["y"] - results_df["yhat"]

    return results_df

def evaluate_model(results_df):

    mae = mean_absolute_error(results_df["y"], results_df["yhat"])
    rmse = np.sqrt(mean_squared_error(results_df["y"], results_df["yhat"]))

    return mae, rmse

def detect_anomalies(results_df, min_points=5):

    results_df = results_df.copy()

    #Point anomaly ( yhat lower )
    results_df["point_anomaly"] = results_df["y"] < results_df["yhat_lower"]

    #groups of consecutive values
    results_df["group"] = (
        results_df["point_anomaly"] != results_df["point_anomaly"].shift()
    ).cumsum()

    #Count group sizes
    group_sizes = results_df.groupby("group")["point_anomaly"].transform("size")

    # Final anomaly 
    results_df["final_anomaly"] = (
        (results_df["point_anomaly"] == True) &
        (group_sizes >= min_points)
    )

    #Flag
    results_df["anomaly_flag"] = results_df["final_anomaly"].astype(int)

    anomaly_count = results_df["anomaly_flag"].sum()

    return results_df, anomaly_count
