import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"


def chat(history, api_key):
    if not api_key:
        return "❌ API key yo‘q"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": history[-10:],
        "temperature": 0.4,
        "max_tokens": 400
    }

    try:
        r = requests.post(GROQ_URL, headers=headers, json=payload, timeout=20)

        if r.status_code != 200:
            return f"API ERROR: {r.text}"

        return r.json()["choices"][0]["message"]["content"]

    except Exception as e:
        return f"ERROR: {str(e)}"


QUICK_QUESTIONS = [
    "How many persons detected?",
    "Is there any anomaly?",
    "Traffic analysis summary?",
    "System status?"
]