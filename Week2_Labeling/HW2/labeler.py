import requests
import streamlit as st

# ---- Settings ----------------------------------------------------------
BUSINESS = "a ride-sharing company"

LABELS = [
    "driver_behavior",
    "wait_time",
    "pricing",
    "vehicle_quality",
    "safety",
    "positive"
]

GUIDE = """Label the comment with exactly one of these labels:

driver_behavior: Complaints about driver attitude or professionalism.
wait_time: Complaints about waiting time or pickup issues.
pricing: Complaints about fares or unexpected charges.
vehicle_quality: Complaints about cleanliness, comfort, or vehicle condition.
safety: Complaints about unsafe driving or safety concerns.
positive: Praise or satisfaction without significant complaints.

If a comment mentions safety concerns, label it safety.
If multiple other issues appear, label by the main problem.
If the comment is unclear or unrelated, reply unclear.
Reply with only the label, nothing else."""

MODEL = "openai/gpt-oss-20b"
URL = "https://api.groq.com/openai/v1/chat/completions"
# ------------------------------------------------------------------------


def label_comment(text):
    key = st.secrets["GROQ_API_KEY"]
    prompt = f"You label customer comments for {BUSINESS}.\n{GUIDE}\n\nComment: {text}"
    response = requests.post(
        URL,
        headers={"Authorization": f"Bearer {key}"},
        json={"model": MODEL, "temperature": 0,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=30,
    )
    response.raise_for_status()
    answer = response.json()["choices"][0]["message"]["content"]
    label = answer.strip().lower()
    return label if label in LABELS else "unclear"
