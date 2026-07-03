import pandas as pd

def create_events(results_df):

    df = results_df[results_df["final_anomaly"]].copy()

    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    df["time_diff"] = df["ds"].diff()

    df["new_event"] = (
        df["time_diff"] > pd.Timedelta(minutes=5)
    ).astype(int)

    df["event_id"] = df["new_event"].cumsum()

    events = df.groupby("event_id").agg({
        "ds": ["min", "max"],
        "y": "min",
        "yhat_lower": "mean"
    }).reset_index()

    events.columns = [
        "event_id",
        "start_time",
        "end_time",
        "min_throughput",
        "expected_lower"
    ]

    events["duration_min"] = (
        (events["end_time"] - events["start_time"]).dt.total_seconds() / 60
    )
    events["percent_drop"] = (
        (events["expected_lower"] - events["min_throughput"])
        / events["expected_lower"]
    ) * 100

    events["percent_drop"] = events["percent_drop"].round(2)

    return events

def classify_severity(events_df):

    def severity(p):
        if p > 50:
            return "CRITICAL"
        elif p > 30:
            return "HIGH"
        else:
            return "MEDIUM"

    events_df = events_df.copy()
    events_df["severity"] = events_df["percent_drop"].apply(severity)

    return events_df

def generate_alarm_messages(events_df):

    def generate_alarm(row):
        return f"""
NETWORK ANOMALY ALERT

Event ID: {row['event_id']}
Start Time: {row['start_time']}
End Time: {row['end_time']}

Duration: {round(row['duration_min'], 1)} minutes

Severity: {row['severity']}

Drop: {row['percent_drop']}%

Impact:
Throughput dropped below expected range.

Recommended Action:
• Check congestion
• Verify network load
• Inspect logs
"""

    events_df = events_df.copy()
    events_df["alarm_message"] = events_df.apply(generate_alarm, axis=1)

    return events_df
