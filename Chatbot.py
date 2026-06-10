from groq import Groq
from database import get_stats_summary, get_detections, get_events

SYSTEM_PROMPT = """You are ASSBI Assistant — an expert AI analyst for the AI-Powered Smart Surveillance and Business Intelligence platform.

You help users understand:
- Live surveillance data (person counts, vehicle counts, crowd anomalies)
- Business intelligence insights and KPIs
- System performance and fine-tuning results
- Privacy, ethics, and GDPR compliance in AI surveillance
- Technical questions about YOLOv8, OpenCV, and the data pipeline

When answering, be concise, data-driven, and professional.
If real-time stats are provided, use them in your answer.
Always mention relevant ethical considerations when discussing surveillance data.
"""

def build_context():
    """Pull latest stats from DB to inject into chatbot context."""
    stats = get_stats_summary()
    df = get_detections(10)
    events_df = get_events(5)

    recent = ""
    if not df.empty:
        last = df.iloc[0]
        recent = (f"Latest frame: {last['person_count']} persons, "
                  f"{last['car_count']} vehicles, "
                  f"anomaly={'YES' if last['anomaly'] else 'NO'}, "
                  f"confidence={last['confidence']}. ")

    event_summary = ""
    if not events_df.empty:
        event_summary = "Recent events: " + "; ".join(
            f"{r['event_type']} ({r['severity']})" for _, r in events_df.iterrows()
        )

    return (
        f"LIVE SYSTEM STATS — "
        f"Total frames analyzed: {stats['total_frames']}. "
        f"Total persons detected: {stats['total_persons_detected']}. "
        f"Total vehicles: {stats['total_cars_detected']}. "
        f"Total anomalies: {stats['total_anomalies']}. "
        f"Avg persons/frame: {stats['avg_persons_per_frame']}. "
        f"{recent}{event_summary}"
    )

def chat(messages: list, api_key: str) -> str:
    """
    messages: list of {"role": "user"/"assistant", "content": "..."}
    Returns assistant reply string.
    """
    client = Groq(api_key=api_key)

    context = build_context()
    system_with_context = SYSTEM_PROMPT + f"\n\n[CURRENT DATA]\n{context}"

    groq_messages = [{"role": "system", "content": system_with_context}] + messages

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=groq_messages,
        max_tokens=800,
        temperature=0.5
    )
    return response.choices[0].message.content


QUICK_QUESTIONS = [
    "What is the current crowd status?",
    "How many anomalies were detected today?",
    "Explain how YOLOv8 detects persons",
    "What are the privacy concerns with this system?",
    "How can I improve detection accuracy?",
    "Summarize today's surveillance report",
    "What does a high anomaly count mean?",
    "How does fine-tuning improve the model?"
]