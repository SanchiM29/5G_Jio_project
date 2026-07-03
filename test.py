import pandas as pd
#from preprocess import preprocess_data, convert_to_5min, prepare_single_cell
from model import prepare_prophet_data, clean_prophet_data, define_windows, rolling_prophet_forecast
from anomaly import calculate_residuals, detect_anomalies
from event import create_events, classify_severity, generate_alarm_messages
import streamlit as st

df = pd.read_csv("telecom_data.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])


#df = preprocess_data(df)
prophet_df = prepare_prophet_data(
    df,
    time_col="Timestamp",
    kpi="Throughput 4G (Mbps)"
)

prophet_df = clean_prophet_data(prophet_df)


train_window, predict_window = define_windows()

results_df, model_dates = rolling_prophet_forecast(
    prophet_df,
    train_window,
    predict_window
)

results_df = calculate_residuals(results_df)
results_df, _ = detect_anomalies(results_df)
events_df = create_events(results_df)
events_df = classify_severity(events_df)
events_df = generate_alarm_messages(events_df)
results_df.to_csv("rolling_predictions.csv", index=False)
events_df.to_csv("anomaly_events.csv", index=False)

st.session_state["results_df"] = results_df
st.session_state["events_df"] = events_df

# IMPORTANT for RAG
events_df.to_csv("anomaly_events.csv", index=False)

print("\n PIPELINE SUCCESSFUL\n")
print(events_df.head())
