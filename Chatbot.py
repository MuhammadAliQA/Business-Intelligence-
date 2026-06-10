import requests
from database import get_kpis, get_detections, get_events

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

QUICK_QUESTIONS = [
    "📊 Current detection summary",
    "🚨 How many anomalies today?",
    "📈 What's the crowd trend?",
    "🎯 Detection accuracy?",
    "⏰ Peak activity hours?",
    "🚗 Total vehicles detected?",
    "🔴 Last anomaly event",
    "💡 Recommendations?",
]


def _build_system_prompt():
    """Build context-aware system prompt from live DB data."""
    kpis = get_kpis()
    df = get_detections(50)
    events_df = get_events(10)

    recent_data = ""
    if not df.empty:
        recent_data = df[["timestamp", "person_count", "car_count", "anomaly", "confidence"]].tail(10).to_string(index=False)

    recent_events = ""
    if not events_df.empty:
        recent_events = events_df[["timestamp", "event_type", "description"]].to_string(index=False)

    return f"""You are ASSBI AI Assistant — an intelligent Business Intelligence chatbot for a smart surveillance platform.

## Platform Overview
ASSBI (AI Smart Surveillance & Business Intelligence) monitors live video streams (primarily Abbey Road, London) using YOLOv8 object detection and OpenCV.

## Live KPI Data (as of now)
- Current persons in frame: {kpis['current_persons']}
- Current anomaly status: {'⚠️ YES — CROWD DETECTED' if kpis['current_anomaly'] else '✅ Normal'}
- Total frames analyzed: {kpis['total_frames_analyzed']}
- Total anomaly events: {kpis['total_anomalies']}
- Average persons per frame: {kpis['avg_persons']}
- Total vehicles detected: {kpis['total_vehicles']}

## Recent Detections (last 10 frames)
{recent_data if recent_data else 'No detections yet — run the Live Detection module first.'}

## Recent Events (last 10)
{recent_events if recent_events else 'No events recorded yet.'}

## Your Role
- Answer questions about the surveillance data above
- Provide BI insights: trends, patterns, recommendations
- Explain how the detection system works (YOLOv8, anomaly thresholds, fine-tuning)
- Suggest actions when anomalies are detected
- Be concise, professional, and data-driven
- If asked about data you don't have, say so honestly

Anomaly threshold: ≥8 persons in a single frame triggers a crowd alert.
"""


def chat(history: list, api_key: str) -> str:
    """
    Send conversation history to Groq API and return AI response.
    history: list of {"role": "user"/"assistant", "content": "..."}
    """
    if not api_key or not api_key.strip():
        return "⚠️ Please enter your Groq API key in the sidebar."

    system_prompt = _build_system_prompt()

    messages = [{"role": "system", "content": system_prompt}] + history

    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.5,
    }

    try:
        resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    except requests.exceptions.HTTPError as e:
        if resp.status_code == 401:
            return "❌ Invalid Groq API key. Please check and re-enter it in the sidebar."
        elif resp.status_code == 429:
            return "⚠️ Rate limit reached. Please wait a moment and try again."
        else:
            return f"❌ API error {resp.status_code}: {str(e)}"

    except requests.exceptions.Timeout:
        return "⏱️ Request timed out. Please try again."

    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"