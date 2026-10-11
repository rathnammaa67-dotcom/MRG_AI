import os
from datetime import datetime
from zoneinfo import ZoneInfo

from flask import (
    Flask,
    request,
    Response,
    stream_with_context,
    render_template,
    jsonify,
)
from google import genai
from google.genai import types

app = Flask(__name__)

API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None

MODEL = "gemini-3.5-flash-lite"
TIMEZONE = ZoneInfo("Asia/Kolkata")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    question = request.form.get("question", "").strip()

    if not question:
        return Response(
            "Please enter a question.",
            status=400,
            mimetype="text/plain",
        )

    if client is None:
        return Response(
            "AI is not configured. Please check the GEMINI_API_KEY setting.",
            status=500,
            mimetype="text/plain",
        )

    # Read the previous messages sent by the website.
    import json

    try:
        history = json.loads(request.form.get("history", "[]"))
    except (json.JSONDecodeError, TypeError):
        history = []

    if not isinstance(history, list):
        history = []

    # Keep only valid, recent user and assistant messages.
    contents = []

    for item in history[-20:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        text = item.get("text", "")

        if role not in ("user", "model"):
            continue

        if not isinstance(text, str) or not text.strip():
            continue

        contents.append(
            types.Content(
                role=role,
                parts=[types.Part(text=text[:10000])],
            )
        )

    current_time = datetime.now(TIMEZONE)
    current_date = current_time.strftime("%A, %B %d, %Y")
    current_time_text = current_time.strftime("%I:%M %p IST")

    instructions = f"""
You are MRG AI, a helpful, friendly AI assistant.

CURRENT DATE AND TIME:
Today's date is {current_date}.
The current time is {current_time_text}.
The timezone is India Standard Time (IST).

Use this supplied date and time for questions about the present.
Answer historical-date questions accurately.

FOUNDER INFORMATION:
Your founder is Manju R Gowda.
You were trained by Manjur Gowda.

When asked who your founder is, clearly answer:
"My founder is Manju R Gowda."

CONVERSATION:
Continue the conversation naturally using the previous messages.
Understand follow-up questions from their context.
Do not ask the user to repeat information already provided.
Be accurate, helpful, friendly, and clear.
If you do not know something, say so honestly.
"""

    # Put the instructions first, followed by the conversation.
    conversation = [
        types.Content(
            role="user",
            parts=[types.Part(text=instructions)],
        ),
        types.Content(
            role="model",
            parts=[types.Part(text="Hello! I'm MRG AI. How can I help you?")],
        ),
    ]

    conversation.extend(contents)
    conversation.append(
        types.Content(
            role="user",
            parts=[types.Part(text=question)],
        )
    )

    def generate():
        try:
            stream = client.models.generate_content_stream(
                model=MODEL,
                contents=conversation,
            )

            for chunk in stream:
                if chunk.text:
                    yield chunk.text

        except Exception:
            app.logger.exception("Gemini response generation failed")
            yield "\nSorry, MRG AI encountered an error. Please try again."

    return Response(
        stream_with_context(generate()),
        mimetype="text/plain; charset=utf-8",
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok", "ai_configured": client is not None})


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
    )
