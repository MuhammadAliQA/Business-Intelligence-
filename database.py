import sqlite3
import pandas as pd

DB = "assbi.db"


# ───────────────────────── CONNECTION ─────────────────────────

def conn():
    return sqlite3.connect(DB)


# ───────────────────────── INIT DB ─────────────────────────

def init_db():
    with conn() as c:
        cur = c.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS detections(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            person_count INTEGER,
            car_count INTEGER,
            other_count INTEGER,
            anomaly INTEGER,
            confidence REAL,
            source TEXT
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
            event_type TEXT,
            description TEXT,
            severity TEXT
        )
        """)

        c.commit()


# ───────────────────────── INSERT DETECTION ─────────────────────────

def insert_detection(p, c, o, a, conf, source):
    with conn() as db:
        db.execute("""
            INSERT INTO detections
            (person_count, car_count, other_count, anomaly, confidence, source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (p, c, o, int(a), float(conf), source))
        db.commit()


# ───────────────────────── INSERT EVENT ─────────────────────────

def insert_event(event_type, description, severity="info"):
    with conn() as db:
        db.execute("""
            INSERT INTO events (event_type, description, severity)
            VALUES (?, ?, ?)
        """, (event_type, description, severity))
        db.commit()


# ───────────────────────── GET DATA ─────────────────────────

def get_detections(limit=200):
    with conn() as db:
        return pd.read_sql_query(
            "SELECT * FROM detections ORDER BY id DESC LIMIT ?",
            db,
            params=(limit,)
        )


def get_events(limit=50):
    with conn() as db:
        return pd.read_sql_query(
            "SELECT * FROM events ORDER BY id DESC LIMIT ?",
            db,
            params=(limit,)
        )


# ───────────────────────── KPI ENGINE ─────────────────────────

def get_kpis():
    with conn() as db:
        cur = db.cursor()

        last = cur.execute("""
            SELECT person_count, car_count, anomaly
            FROM detections
            ORDER BY id DESC
            LIMIT 1
        """).fetchone()

        total_frames = cur.execute("SELECT COUNT(*) FROM detections").fetchone()[0]

        total_anomalies = cur.execute(
            "SELECT COUNT(*) FROM detections WHERE anomaly=1"
        ).fetchone()[0]

        avg_persons = cur.execute(
            "SELECT AVG(person_count) FROM detections"
        ).fetchone()[0] or 0

        total_vehicles = cur.execute(
            "SELECT SUM(car_count) FROM detections"
        ).fetchone()[0] or 0

        return {
            "current_persons": last[0] if last else 0,
            "current_anomaly": bool(last[2]) if last else False,
            "total_frames_analyzed": total_frames,
            "total_anomalies": total_anomalies,
            "avg_persons": round(avg_persons, 2),
            "total_vehicles": total_vehicles
        }


# ───────────────────────── STATS SUMMARY ─────────────────────────

def get_stats_summary():
    with conn() as db:
        cur = db.cursor()

        return {
            "total_frames": cur.execute("SELECT COUNT(*) FROM detections").fetchone()[0],
            "total_anomalies": cur.execute("SELECT COUNT(*) FROM detections WHERE anomaly=1").fetchone()[0],
            "total_persons_detected": cur.execute("SELECT SUM(person_count) FROM detections").fetchone()[0] or 0
        }


# ───────────────────────── CLEAR DB ─────────────────────────

def clear_all():
    with conn() as db:
        db.execute("DELETE FROM detections")
        db.execute("DELETE FROM events")
        db.commit()