import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from database import get_detections, get_events, get_stats_summary

COLORS = {
    "person": "#00C896",
    "car": "#FF8C42",
    "anomaly": "#FF4B6E",
    "other": "#A78BFA",
    "bg": "rgba(0,0,0,0)"
}

def _base_layout(title=""):
    return dict(
        title=dict(text=title, font=dict(size=15)),
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color="#e0e0e0", size=12),
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(gridcolor="#333", showgrid=True),
        yaxis=dict(gridcolor="#333", showgrid=True),
        legend=dict(orientation="h", y=-0.2)
    )

def chart_person_over_time():
    df = get_detections(300)
    if df.empty:
        return _empty_chart("No data yet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["person_count"],
        mode="lines+markers", name="Persons",
        line=dict(color=COLORS["person"], width=2),
        marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["car_count"],
        mode="lines", name="Vehicles",
        line=dict(color=COLORS["car"], width=2, dash="dot")))
    fig.update_layout(**_base_layout("Person & Vehicle Count Over Time"))
    return fig

def chart_anomaly_timeline():
    df = get_detections(300)
    if df.empty:
        return _empty_chart("No data yet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    anomalies = df[df["anomaly"] == 1]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["timestamp"], y=df["person_count"],
        name="Person count", marker_color=COLORS["person"], opacity=0.5))
    if not anomalies.empty:
        fig.add_trace(go.Scatter(x=anomalies["timestamp"], y=anomalies["person_count"],
            mode="markers", name="Anomaly",
            marker=dict(color=COLORS["anomaly"], size=12, symbol="x")))
    fig.update_layout(**_base_layout("Anomaly Detection Timeline"))
    return fig

def chart_detection_pie():
    stats = get_stats_summary()
    labels = ["Persons", "Vehicles", "Anomalies"]
    values = [
        stats["total_persons_detected"],
        stats["total_cars_detected"],
        stats["total_anomalies"]
    ]
    if sum(values) == 0:
        return _empty_chart("No detections yet")
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker=dict(colors=[COLORS["person"], COLORS["car"], COLORS["anomaly"]]),
        hole=0.4, textinfo="percent+label"
    ))
    fig.update_layout(**_base_layout("Detection Distribution"))
    return fig

def chart_heatmap_hourly():
    df = get_detections(1000)
    if df.empty:
        return _empty_chart("No data yet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day_name()
    pivot = df.pivot_table(values="person_count", index="day", columns="hour", aggfunc="mean", fill_value=0)
    days_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = pivot.reindex([d for d in days_order if d in pivot.index])
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h}:00" for h in pivot.columns],
        y=pivot.index.tolist(),
        colorscale="Viridis",
        colorbar=dict(title="Avg persons")
    ))
    fig.update_layout(**_base_layout("Crowd Density Heatmap (Hour vs Day)"))
    return fig

def chart_confidence_histogram():
    df = get_detections(500)
    if df.empty or df["confidence"].sum() == 0:
        return _empty_chart("No confidence data")
    fig = go.Figure(go.Histogram(
        x=df["confidence"], nbinsx=20,
        marker_color=COLORS["person"], opacity=0.8
    ))
    fig.update_layout(**_base_layout("Detection Confidence Distribution"))
    return fig

def chart_fine_tune_loss():
    from database import get_fine_tune_logs
    df = get_fine_tune_logs()
    if df.empty:
        return _empty_chart("No fine-tuning data yet")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["epoch"], y=df["loss"],
        mode="lines+markers", name="Loss",
        line=dict(color=COLORS["anomaly"], width=2)))
    fig.add_trace(go.Scatter(x=df["epoch"], y=df["accuracy"],
        mode="lines+markers", name="Accuracy",
        line=dict(color=COLORS["person"], width=2)))
    fig.update_layout(**_base_layout("Fine-tuning: Loss & Accuracy"))
    return fig

def get_kpis():
    stats = get_stats_summary()
    df = get_detections(50)
    current_persons = int(df["person_count"].iloc[0]) if not df.empty else 0
    current_anomaly = bool(df["anomaly"].iloc[0]) if not df.empty else False
    return {
        "current_persons": current_persons,
        "current_anomaly": current_anomaly,
        "total_frames_analyzed": stats["total_frames"],
        "total_anomalies": stats["total_anomalies"],
        "avg_persons": stats["avg_persons_per_frame"],
        "total_vehicles": stats["total_cars_detected"]
    }

def _empty_chart(msg="No data"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="#888"))
    fig.update_layout(**_base_layout())
    return fig

def generate_demo_data():
    """Insert synthetic data for demo/testing purposes."""
    import random
    from database import insert_detection, insert_event
    from datetime import datetime, timedelta
    base = datetime.now() - timedelta(hours=6)
    for i in range(120):
        t = base + timedelta(minutes=i * 3)
        pc = random.randint(0, 15)
        cc = random.randint(0, 5)
        oc = random.randint(0, 3)
        anomaly = pc >= 8
        conf = random.uniform(0.55, 0.97)
        conn = __import__("database").get_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO detections (timestamp, person_count, car_count, other_count, anomaly, confidence, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (t.strftime("%Y-%m-%d %H:%M:%S"), pc, cc, oc, int(anomaly), round(conf, 3), "demo"))
        conn.commit()
        conn.close()
        if anomaly:
            insert_event("CROWD_ANOMALY", f"Demo: {pc} persons at {t.strftime('%H:%M')}", "warning")