import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# your modules
#from preprocess import preprocess_data, convert_to_5min, prepare_single_cell
from model import prepare_prophet_data, clean_prophet_data, define_windows, rolling_prophet_forecast
from anomaly import calculate_residuals, detect_anomalies
from event import create_events, classify_severity, generate_alarm_messages
from rag import explain

if "pipeline_ran" not in st.session_state:
    st.session_state["pipeline_ran"] = False

if "results_df" not in st.session_state:
    st.session_state["results_df"] = None

if "events_df" not in st.session_state:
    st.session_state["events_df"] = None

st.set_page_config(layout="wide")
st.title("4G Anomaly Detection Dashboard")

uploaded_file = st.file_uploader("Upload Telecom Dataset", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    import plotly.express as px

    fig_trend = px.line(
        df,
        x="Timestamp",
        y="Throughput 4G (Mbps)",
        title="Raw 4G Throughput Trend",
        template="plotly_dark"
    )

    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("EDA")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", len(df))
    col2.metric("Columns", len(df.columns))
    col3.metric("KPI Columns", len(df.select_dtypes(include="number").columns))

    st.subheader("Time Continuity Check")
    st.subheader("Raw Throughput Trend")

# BUTTON
if st.button("Run Full Pipeline"):
    st.session_state["pipeline_ran"] = True

# RUN PIPELINE
if st.session_state["pipeline_ran"]:

    with st.spinner("Processing..."):

        prophet_df = prepare_prophet_data(df)
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

        # SAVE STATE
        st.session_state["results_df"] = results_df
        st.session_state["events_df"] = events_df

        # SAVE FOR RAG
        events_df.to_csv("anomaly_events.csv", index=False)

        st.success("Pipeline Completed")

    # LOAD FROM STATE
    results_df = st.session_state["results_df"]
    events_df = st.session_state["events_df"]

    st.subheader("Forecast + Anomalies")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results_df["ds"],
        y=results_df["y"],
        mode="lines",
        name="Actual"
    ))

    fig.add_trace(go.Scatter(
        x=results_df["ds"],
        y=results_df["yhat"],
        mode="lines",
        name="Predicted"
    ))

    fig.add_trace(go.Scatter(
        x=results_df["ds"],
        y=results_df["yhat_upper"],
        mode="lines",
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=results_df["ds"],
        y=results_df["yhat_lower"],
        fill='tonexty',
        mode="lines",
        fillcolor='rgba(255,165,0,0.15)',
        line=dict(width=0),
        name="Confidence Interval"
    ))

    anomalies = results_df[results_df["final_anomaly"]]

    fig.add_trace(go.Scatter(
        x=anomalies["ds"],
        y=anomalies["y"],
        mode="markers",
        marker=dict(color="red", size=6),
        name="Final Anomalies"
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Anomaly Events")
    st.dataframe(events_df)

    st.subheader("AI Explanation")

    if len(events_df) == 0:
        st.info("No anomaly events detected")

    else:

        selected_event = st.selectbox(
            "Select Event",
            events_df.index
        )

        event_row = events_df.loc[selected_event]

        st.write(event_row)

        # QUESTION OPTIONS
        question_options = [
            "What happened during this anomaly?",
            "What is the most likely cause?",
            "How severe is this anomaly?",
            "What actions should be taken?",
            "Summarize this anomaly event",
        ]

        selected_question = st.selectbox(
            "Choose Analysis Question",
            question_options
        )

        if st.button("Generate Explanation"):

            try:

                query = f"""
Selected Anomaly Event:
{str(event_row)}

User Question:
{selected_question}
"""

                explanation = explain(query)

                st.success(explanation)

            except Exception as e:
                st.error(f"Error: {e}")
