import sqlite3
import pandas as pd
from datetime import datetime

DB_PATH = "assbi.db"

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            person_count INTEGER DEFAULT 0,
            car_count INTEGER DEFAULT 0,
            other_count INTEGER DEFAULT 0,
            anomaly INTEGER DEFAULT 0,
            confidence REAL DEFAULT 0.0,
            source TEXT DEFAULT 'stream',
            frame_path TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            event_type TEXT,
            description TEXT,
            severity TEXT DEFAULT 'info'
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS fine_tune_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            epoch INTEGER,
            loss REAL,
            accuracy REAL,
            model_path TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_detection(person_count, car_count, other_count, anomaly, confidence, source="stream", frame_path=None):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO detections (timestamp, person_count, car_count, other_count, anomaly, confidence, source, frame_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), person_count, car_count, other_count, int(anomaly), confidence, source, frame_path))
    conn.commit()
    conn.close()

def insert_event(event_type, description, severity="info"):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO events (timestamp, event_type, description, severity)
        VALUES (?, ?, ?, ?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), event_type, description, severity))
    conn.commit()
    conn.close()

def insert_fine_tune_log(epoch, loss, accuracy, model_path=""):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        INSERT INTO fine_tune_logs (timestamp, epoch, loss, accuracy, model_path)
        VALUES (?, ?, ?, ?, ?)
    """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, loss, accuracy, model_path))
    conn.commit()
    conn.close()

def get_detections(limit=500):
    conn = get_conn()
    df = pd.read_sql_query(f"SELECT * FROM detections ORDER BY id DESC LIMIT {limit}", conn)
    conn.close()
    return df

def get_events(limit=100):
    conn = get_conn()
    df = pd.read_sql_query(f"SELECT * FROM events ORDER BY id DESC LIMIT {limit}", conn)
    conn.close()
    return df

def get_fine_tune_logs():
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM fine_tune_logs ORDER BY id ASC", conn)
    conn.close()
    return df

def get_stats_summary():
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM detections")
    total = c.fetchone()[0]
    c.execute("SELECT SUM(person_count), SUM(car_count), SUM(anomaly) FROM detections")
    row = c.fetchone()
    c.execute("SELECT AVG(person_count) FROM detections")
    avg_persons = c.fetchone()[0] or 0
    conn.close()
    return {
        "total_frames": total,
        "total_persons_detected": int(row[0] or 0),
        "total_cars_detected": int(row[1] or 0),
        "total_anomalies": int(row[2] or 0),
        "avg_persons_per_frame": round(avg_persons, 2)
    }

def clear_all():
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM detections")
    c.execute("DELETE FROM events")
    conn.commit()
    conn.close()