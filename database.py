import sqlite3
import pandas as pd
import random
from datetime import datetime, timedelta

DB_PATH = "assbi.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT (datetime('now','localtime')),
            person_count INTEGER DEFAULT 0,
            car_count INTEGER DEFAULT 0,
            other_count INTEGER DEFAULT 0,
            anomaly INTEGER DEFAULT 0,
            confidence REAL DEFAULT 0.0,
            source TEXT DEFAULT 'youtube'
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT (datetime('now','localtime')),
            event_type TEXT,
            description TEXT,
            severity TEXT DEFAULT 'info'
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS dataset_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            split TEXT,
            label_path TEXT,
            added_at TEXT DEFAULT (datetime('now','localtime'))
        )
    """)
    conn.commit()
    conn.close()


def insert_detection(person_count, car_count, other_count, anomaly, confidence, source="youtube"):
    conn = get_conn()
    conn.execute(
        "INSERT INTO detections (person_count,car_count,other_count,anomaly,confidence,source) VALUES (?,?,?,?,?,?)",
        (person_count, car_count, other_count, int(anomaly), round(confidence, 3), source)
    )
    conn.commit()
    conn.close()


def insert_event(event_type, description, severity="info"):
    conn = get_conn()
    conn.execute(
        "INSERT INTO events (event_type, description, severity) VALUES (?,?,?)",
        (event_type, description, severity)
    )
    conn.commit()
    conn.close()


def get_detections(limit=200):
    conn = get_conn()
    df = pd.read_sql_query(
        f"SELECT * FROM detections ORDER BY id DESC LIMIT {limit}",
        conn
    )
    conn.close()
    return df


def get_events(limit=20):
    conn = get_conn()
    df = pd.read_sql_query(
        f"SELECT * FROM events ORDER BY id DESC LIMIT {limit}",
        conn
    )
    conn.close()
    return df


def get_kpis():
    conn = get_conn()
    c = conn.cursor()

    row = c.execute("SELECT * FROM detections ORDER BY id DESC LIMIT 1").fetchone()
    current_persons = row["person_count"] if row else 0
    current_anomaly = bool(row["anomaly"]) if row else False

    total_frames = c.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    total_anomalies = c.execute("SELECT COUNT(*) FROM detections WHERE anomaly=1").fetchone()[0]
    avg_row = c.execute("SELECT AVG(person_count) FROM detections").fetchone()[0]
    avg_persons = round(avg_row, 1) if avg_row else 0.0
    total_vehicles = c.execute("SELECT SUM(car_count) FROM detections").fetchone()[0] or 0

    conn.close()
    return {
        "current_persons": current_persons,
        "current_anomaly": current_anomaly,
        "total_frames_analyzed": total_frames,
        "total_anomalies": total_anomalies,
        "avg_persons": avg_persons,
        "total_vehicles": total_vehicles,
    }


def get_stats_summary():
    conn = get_conn()
    c = conn.cursor()
    total_frames = c.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    total_anomalies = c.execute("SELECT COUNT(*) FROM detections WHERE anomaly=1").fetchone()[0]
    total_persons = c.execute("SELECT SUM(person_count) FROM detections").fetchone()[0] or 0
    conn.close()
    return {
        "total_frames": total_frames,
        "total_anomalies": total_anomalies,
        "total_persons_detected": total_persons,
    }


def clear_all():
    conn = get_conn()
    conn.execute("DELETE FROM detections")
    conn.execute("DELETE FROM events")
    conn.commit()
    conn.close()


def generate_demo_data():
    """Insert 120 realistic demo records for testing."""
    conn = get_conn()
    base_time = datetime.now() - timedelta(hours=6)
    for i in range(120):
        ts = (base_time + timedelta(minutes=i * 3)).strftime("%Y-%m-%d %H:%M:%S")
        hour = (base_time + timedelta(minutes=i * 3)).hour
        # Rush hour simulation
        if 7 <= hour <= 9 or 16 <= hour <= 18:
            persons = random.randint(4, 14)
        else:
            persons = random.randint(0, 6)
        cars = random.randint(0, 5)
        others = random.randint(0, 3)
        anomaly = 1 if persons >= 8 else 0
        conf = round(random.uniform(0.55, 0.95), 3)
        conn.execute(
            "INSERT INTO detections (timestamp,person_count,car_count,other_count,anomaly,confidence,source) VALUES (?,?,?,?,?,?,?)",
            (ts, persons, cars, others, anomaly, conf, "demo")
        )
        if anomaly:
            conn.execute(
                "INSERT INTO events (timestamp,event_type,description,severity) VALUES (?,?,?,?)",
                (ts, "CROWD_ANOMALY", f"{persons} persons detected at Abbey Road", "warning")
            )
    conn.commit()
    conn.close()