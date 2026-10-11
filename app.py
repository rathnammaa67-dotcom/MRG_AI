import os
from datetime import datetime
from zoneinfo import ZoneInfo

from flask import Flask, request, Response, stream_with_context, render_template
from google import genai

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
            mimetype="text/plain"
        )

    if client is None:
        return Response(
            "AI is not configured. Please check the GEMINI_API_KEY setting.",
            status=500,
            mimetype="text/plain"
        )

    current_time = datetime.now(TIMEZONE)
    current_date = current_time.strftime("%A, %B %d, %Y")
    current_time_text = current_time.strftime("%I:%M %p IST")

    instructions = f"""
You are MRG AI, a helpful, friendly AI assistant.

IMPORTANT CURRENT DATE AND TIME:
Today's date is {current_date}.
The current time is {current_time_text}.
The timezone is India Standard Time (IST).

Always use this supplied date when answering questions about today's date,
the current year, or the current time. Never invent an old date.
For questions about historical dates, answer based on the actual historical date.

FOUNDER INFORMATION:
Your founder is Manju R Gowda.
You were trained by Manjur Gowda.

When asked who your founder is, clearly answer:
"My founder is Manju R Gowda."

Be accurate, helpful, and clear. If you do not know something,
say so honestly.

USER QUESTION:
{question}
"""

    def generate():
        try:
            stream = client.models.generate_content_stream(
                model=MODEL,
                contents=instructions,
            )

            for chunk in stream:
                if chunk.text:
                    yield chunk.text

        except Exception:
            app.logger.exception("Gemini response generation failed")
            yield "\nSorry, MRG AI encountered an error. Please try again."

    return Response(
        stream_with_context(generate()),
        mimetype="text/plain; charset=utf-8"
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
