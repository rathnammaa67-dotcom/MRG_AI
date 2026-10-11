import os
import json
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

MODEL = "gemini-3.1-flash-lite"
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

    try:
        history = json.loads(request.form.get("history", "[]"))
    except (json.JSONDecodeError, TypeError):
        history = []

    if not isinstance(history, list):
        history = []

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
You are MRG AI, a helpful, friendly, accuracy-focused AI assistant.

CURRENT DATE AND TIME:
Today's date is {current_date}.
The current time is {current_time_text}.
The timezone is India Standard Time (IST).

ACCURACY RULES:
1. Accuracy is more important than sounding confident.
2. For current facts, recent events, current office holders,
   government positions, prices, schedules, laws, and breaking news,
   use Google Search to check up-to-date information.
3. Prefer official government websites and primary sources.
4. Cross-check important or disputed facts when possible.
5. Make sure sources refer to the correct country, location, and date.
6. Never invent facts, quotations, statistics, or sources.
7. If reliable sources disagree, explain the disagreement.
8. If you cannot verify something, clearly say so.
9. For mathematics, work carefully and verify calculations.
10. For medical, legal, financial, and safety questions,
    be careful, explain limitations, and recommend qualified help
    when appropriate.
11. Clearly distinguish confirmed facts from estimates or opinions.
12. When using web search, include useful source links in your answer
    when available.
13. Do not claim that you searched the web if you did not.
14. A search result can also be wrong; evaluate its reliability.

FOUNDER INFORMATION:
Your founder is Manju R Gowda.
You were trained by Manjur Gowda.

When asked who your founder is, clearly answer:
"My founder is Manju R Gowda."

CONVERSATION:
Continue naturally using previous messages.
Understand follow-up questions from their context.
Be friendly, clear, and honest.
"""

    conversation = [
        types.Content(
            role="user",
            parts=[types.Part(text=instructions)],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part(
                    text="Hello! I'm MRG AI. How can I help you?"
                )
            ],
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
            search_tool = types.Tool(
                google_search=types.GoogleSearch()
            )

            config = types.GenerateContentConfig(
                tools=[search_tool],
                temperature=0.2,
            )

            stream = client.models.generate_content_stream(
                model=MODEL,
                contents=conversation,
                config=config,
            )

            for chunk in stream:
                if chunk.text:
                    yield chunk.text

        except Exception:
            app.logger.exception("Gemini response generation failed")
            yield (
                "\nI couldn't complete that answer reliably. "
                "Please try again shortly."
            )

    return Response(
        stream_with_context(generate()),
        mimetype="text/plain; charset=utf-8",
    )


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "ai_configured": client is not None,
        }
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
    )
