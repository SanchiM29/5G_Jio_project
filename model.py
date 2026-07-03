import pandas as pd
import pickle
from prophet import Prophet
import os

def prepare_prophet_data(df, time_col="Timestamp", kpi="Throughput 4G (Mbps)"):

    if time_col not in df.columns or kpi not in df.columns:
        raise ValueError("Columns not found in dataframe")

    df = df.copy()

    # Convert to datetime
    df[time_col] = pd.to_datetime(df[time_col])

    # Sort
    df = df.sort_values(time_col)

    prophet_df = df[[time_col, kpi]].rename(
        columns={time_col: "ds", kpi: "y"}
    )

    return prophet_df

def clean_prophet_data(prophet_df):

    prophet_df = prophet_df.dropna().copy()
    prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)

    return prophet_df

def define_windows(train_days=25, predict_days=1):

    train_window = pd.Timedelta(days=train_days)
    predict_window = pd.Timedelta(days=predict_days)

    return train_window, predict_window

def rolling_prophet_forecast(
    prophet_df,
    train_window,
    predict_window,
    model_save_path="models"
):

    os.makedirs(model_save_path, exist_ok=True)

    start_date = prophet_df["ds"].min() + train_window
    end_date = prophet_df["ds"].max()

    current_date = start_date

    all_predictions = []
    model_dates = []

    while current_date < end_date:

        train_start = current_date - train_window
        train_end = current_date

        test_start = current_date
        test_end = current_date + predict_window

        train_df = prophet_df[
            (prophet_df["ds"] >= train_start) &
            (prophet_df["ds"] < train_end)
        ]

        test_df = prophet_df[
            (prophet_df["ds"] >= test_start) &
            (prophet_df["ds"] < test_end)
        ]

        if len(test_df) == 0:
            break

        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            interval_width=0.8,              
            changepoint_prior_scale=0.05    
        )

        model.fit(train_df)

        forecast = model.predict(test_df[["ds"]])

        result = test_df.copy()

        result["yhat"] = forecast["yhat"].values
        result["yhat_lower"] = forecast["yhat_lower"].values
        result["yhat_upper"] = forecast["yhat_upper"].values

        all_predictions.append(result)

        # Save model
        model_name = f"{model_save_path}/prophet_model_{current_date.date()}.pkl"

        with open(model_name, "wb") as f:
            pickle.dump(model, f)

        model_dates.append(current_date)

        current_date += pd.Timedelta(days=1)
    final_df = pd.concat(all_predictions, ignore_index=True)
    final_df = final_df.drop_duplicates(subset=["ds"]).sort_values("ds")

    return final_df, model_dates
