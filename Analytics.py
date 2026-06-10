import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from database import get_detections, get_kpis


# ── KPIs ───────────────────────────────────────────────────────────────────────

def get_kpis():
    from database import get_kpis as _get_kpis
    return _get_kpis()


# ── Charts ─────────────────────────────────────────────────────────────────────

def chart_person_over_time():
    df = get_detections(300)
    if df.empty:
        return _empty_fig("No data — run detection or load demo data")

    df = df.sort_values("id")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["person_count"],
        mode="lines+markers", name="Persons",
        line=dict(color="#00C896", width=2),
        marker=dict(size=4)
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["car_count"],
        mode="lines", name="Vehicles",
        line=dict(color="#FF8C42", width=2, dash="dot")
    ))
    fig.update_layout(
        **_dark_layout("Person & Vehicle Trend Over Time"),
        xaxis_title="Time", yaxis_title="Count"
    )
    return fig


def chart_anomaly_timeline():
    df = get_detections(300)
    if df.empty:
        return _empty_fig("No data yet")

    df = df.sort_values("id")
    colors = df["anomaly"].map({1: "#FF4B6E", 0: "#00C896"})
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["timestamp"],
        y=df["person_count"],
        marker_color=colors,
        name="Person Count",
        hovertemplate="Time: %{x}<br>Persons: %{y}<extra></extra>"
    ))
    fig.update_layout(
        **_dark_layout("Anomaly Timeline (Red = Crowd Alert)"),
        xaxis_title="Time", yaxis_title="Person Count"
    )
    return fig


def chart_detection_pie():
    df = get_detections(500)
    if df.empty:
        return _empty_fig("No data yet")

    totals = {
        "Persons": int(df["person_count"].sum()),
        "Vehicles": int(df["car_count"].sum()),
        "Other Objects": int(df["other_count"].sum()),
    }
    fig = px.pie(
        values=list(totals.values()),
        names=list(totals.keys()),
        color_discrete_sequence=["#00C896", "#FF8C42", "#7B88FF"],
        hole=0.4,
        title="Detection Distribution"
    )
    fig.update_layout(**_dark_layout("Detection Distribution"))
    return fig


def chart_heatmap_hourly():
    df = get_detections(1000)
    if df.empty:
        return _empty_fig("No data yet")

    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["ts"])
    df["hour"] = df["ts"].dt.hour
    df["day"] = df["ts"].dt.day_name()

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = df.pivot_table(index="day", columns="hour", values="person_count",
                           aggfunc="mean", fill_value=0)
    pivot = pivot.reindex([d for d in day_order if d in pivot.index])

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=pivot.index,
        colorscale="Viridis",
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Avg Persons: %{z:.1f}<extra></extra>"
    ))
    fig.update_layout(**_dark_layout("Hourly Activity Heatmap (Avg Persons)"))
    return fig


def chart_confidence_histogram():
    df = get_detections(500)
    if df.empty:
        return _empty_fig("No data yet")

    fig = px.histogram(
        df, x="confidence", nbins=30,
        color_discrete_sequence=["#7B88FF"],
        title="Detection Confidence Distribution"
    )
    fig.update_layout(
        **_dark_layout("Confidence Distribution"),
        xaxis_title="Confidence Score",
        yaxis_title="Frame Count"
    )
    return fig


def chart_fine_tune_loss():
    """Show training/validation loss curve (uses stored or simulated data)."""
    import os, json

    log_path = "fine_tuned_models/training_log.json"
    if os.path.exists(log_path):
        with open(log_path) as f:
            log = json.load(f)
        epochs = log.get("epochs", [])
        train_loss = log.get("train_loss", [])
        val_loss = log.get("val_loss", [])
    else:
        # Simulate for display
        epochs = list(range(1, 21))
        train_loss = [1.2 * (0.85 ** i) + np.random.uniform(0, 0.05) for i in epochs]
        val_loss = [1.3 * (0.87 ** i) + np.random.uniform(0, 0.07) for i in epochs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, name="Train Loss",
                             line=dict(color="#00C896", width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, name="Val Loss",
                             line=dict(color="#FF4B6E", width=2, dash="dot")))
    fig.update_layout(
        **_dark_layout("Fine-tuning Loss Curve"),
        xaxis_title="Epoch", yaxis_title="Loss"
    )
    return fig


def generate_demo_data():
    from database import generate_demo_data as _gen
    _gen()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _dark_layout(title=""):
    return dict(
        title=title,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#13151f",
        font=dict(color="#e8e8e8"),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor="#1a1d27")
    )


def _empty_fig(msg="No data"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(color="#aaa", size=16))
    fig.update_layout(**_dark_layout())
    return fig