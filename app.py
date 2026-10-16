```python
import os
import json
import time
import random
from datetime import datetime
from zoneinfo import ZoneInfo

from flask import Flask, request, render_template, Response
from google import genai

app = Flask(__name__)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

API_KEY = os.environ.get("GEMINI_API_KEY")

MODEL = "gemini-3.6-flash"
BACKUP_MODEL = "gemini-3.5-flash-lite"

TIMEZONE = ZoneInfo("Asia/Kolkata")

client = genai.Client(api_key=API_KEY) if API_KEY else None


# --------------------------------------------------
# PROMPT
# --------------------------------------------------

def build_prompt(question, history):
    today = datetime.now(TIMEZONE).strftime("%d %B %Y")

    return f"""
You are MRG AI, a helpful, friendly, and intelligent AI assistant.

IDENTITY:
- Founder: Manju R Gowda.
- Developed by Manju R Gowda.

TODAY'S DATE: {today}

INSTRUCTIONS:
1. Answer the user's actual question clearly.
2. Use simple, friendly, natural language.
3. Use conversation history when relevant.
4. Be accurate and never invent facts, news, links, or sources.
5. Answer educational and general knowledge questions directly.
6. Do not claim to have searched the internet.
7. For current information you cannot verify, explain that limitation.
8. Distinguish facts from uncertainty.
9. Format answers clearly.
10. Avoid unnecessary repetition.

CONVERSATION HISTORY:
{json.dumps(history, ensure_ascii=False)}

USER QUESTION:
{question}

Provide one clear and useful answer.
"""


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.route("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL,
        "backup_model": BACKUP_MODEL,
        "api_key_configured": client is not None,
        "web_search": "Disabled",
    }


# --------------------------------------------------
# GEMINI RESPONSE
# --------------------------------------------------

def generate_answer(model, prompt):
    """Generate an answer and retry temporary server errors."""

    if client is None:
        return None

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )

            answer = getattr(response, "text", None)

            if answer and answer.strip():
                return answer.strip()

            print(f"GEMINI ERROR ({model}): Empty response.")

        except Exception as error:
            error_text = str(error).lower()

            print(
                f"GEMINI ERROR ({model}, "
                f"attempt {attempt + 1}/3): {error}"
            )

            temporary_errors = (
                "503",
                "unavailable",
                "429",
                "resource_exhausted",
                "500",
                "502",
                "504",
                "timeout",
                "temporarily",
                "high demand",
            )

            if not any(term in error_text for term in temporary_errors):
                break

            if attempt < 2:
                delay = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)

    return None


# --------------------------------------------------
# CHAT ENDPOINT
# --------------------------------------------------

@app.route("/ask", methods=["POST"])
def ask():

    if client is None:
        return Response(
            "MRG AI is not configured. Please check the "
            "GEMINI_API_KEY setting in Render.",
            status=500,
            mimetype="text/plain",
        )

    question = request.form.get("question", "").strip()

    if not question:
        return Response(
            "Please enter a question.",
            status=400,
            mimetype="text/plain",
        )

    try:
        history = json.loads(
            request.form.get("history", "[]")
        )

        if not isinstance(history, (list, dict)):
            history = []

    except (json.JSONDecodeError, TypeError):
        history = []

    prompt = build_prompt(question, history)

    def generate():

        answer = generate_answer(MODEL, prompt)

        if not answer and BACKUP_MODEL != MODEL:
            print("Trying backup model:", BACKUP_MODEL)
            answer = generate_answer(BACKUP_MODEL, prompt)

        if answer:
            yield answer
        else:
            yield (
                "MRG AI is temporarily unable to respond. 💙 "
                "Google's AI service may be busy. Please try again "
                "in a little while. If the problem continues, "
                "check the latest Render logs and Gemini API quota."
            )

    return Response(
        generate(),
        mimetype="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# --------------------------------------------------
# START APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
    )
```
