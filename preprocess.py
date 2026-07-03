import pandas as pd
def preprocess_data(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["cell_id", "timestamp"])
    return df

def convert_to_5min(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"]) 
    final_data = []

    for cell, group in df.groupby("cell_id"):
        group = group.set_index("timestamp")
        agg = group.resample("5min").agg({
            "throughput_mbps": "median",
            "latency_ms": "median",
            "packet_loss_pct": "max",
            "prb_utilization_pct": "max"
        })

        agg["cell_id"] = cell
        final_data.append(agg.reset_index())
    return pd.concat(final_data, ignore_index=True)


def prepare_single_cell(df_5min, cell_id="CELL_001"):
    df_cell = df_5min[df_5min["cell_id"] == cell_id].copy()

    df_cell = df_cell.sort_values("timestamp")

    df_cell["throughput_mbps"] = df_cell["throughput_mbps"].interpolate()
    df_cell["latency_ms"] = df_cell["latency_ms"].interpolate()
    df_cell["prb_utilization_pct"] = df_cell["prb_utilization_pct"].interpolate()
    df_cell["packet_loss_pct"] = df_cell["packet_loss_pct"].interpolate()

    df_cell = df_cell.dropna()

    return df_cell
