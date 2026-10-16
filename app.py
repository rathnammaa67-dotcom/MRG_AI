import os
import json
import time
import random
from datetime import datetime
from zoneinfo import ZoneInfo

from flask import Flask, request, render_template, Response
from google import genai

app = Flask(__name__)

# Gemini configuration
API_KEY = os.environ.get("GEMINI_API_KEY")

PRIMARY_MODEL = "gemini-3.5-flash-lite"
BACKUP_MODEL = "gemini-3.1-flash-lite"

TIMEZONE = ZoneInfo("Asia/Kolkata")

client = genai.Client(api_key=API_KEY) if API_KEY else None


def build_prompt(question, history):
    today = datetime.now(TIMEZONE).strftime("%d %B %Y")

    return f"""
You are MRG AI, a helpful, friendly, intelligent AI assistant.

IDENTITY:
- Founder: Manju R Gowda.
- Developed by Manju R Gowda.

TODAY'S DATE: {today}

INSTRUCTIONS:
1. Answer the user's actual question clearly.
2. Use simple, natural language.
3. Use conversation history when relevant.
4. Never invent facts, sources, or links.
5. Answer educational and general-knowledge questions directly.
6. Do not claim to have searched the internet.
7. Explain uncertainty about information you cannot verify.
8. Format answers clearly.
9. Do not repeat the same information unnecessarily.

CONVERSATION HISTORY:
{json.dumps(history, ensure_ascii=False)}

USER QUESTION:
{question}

Provide one clear, useful answer.
"""


def is_temporary_error(error):
    """Identify errors that may resolve after waiting."""
    message = str(error).lower()

    temporary_markers = [
        "503",
        "unavailable",
        "429",
        "resource_exhausted",
        "500",
        "502",
        "504",
        "timeout",
        "timed out",
        "high demand",
        "temporarily",
        "internal server error",
    ]

    return any(marker in message for marker in temporary_markers)


def generate_with_retry(model, prompt, max_attempts=2):
    """
    Retry temporary failures with increasing delays.
    Return the answer on success, or None on failure.
    """
    if client is None:
        return None

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )

            answer = getattr(response, "text", None)

            if answer and answer.strip():
                print(f"SUCCESS: {model}")
                return answer.strip()

            print(f"EMPTY RESPONSE: {model}")

        except Exception as error:
            print(
                f"GEMINI ERROR ({model}, "
                f"attempt {attempt + 1}/{max_attempts}): {error}"
            )

            if not is_temporary_error(error):
                break

        if attempt < max_attempts - 1:
            # Wait before retrying a temporary failure.
            delay = min(2 ** (attempt + 1), 8)
            delay += random.uniform(0, 1)
            print(f"Retrying {model} in approximately {delay:.1f} seconds.")
            time.sleep(delay)

    return None


def generate_answer(prompt):
    """Try the primary model, then the backup model."""

    answer = generate_with_retry(
        PRIMARY_MODEL,
        prompt,
        max_attempts=2,
    )

    if answer:
        return answer

    if BACKUP_MODEL and BACKUP_MODEL != PRIMARY_MODEL:
        print(f"PRIMARY MODEL FAILED. Trying {BACKUP_MODEL}.")

        answer = generate_with_retry(
            BACKUP_MODEL,
            prompt,
            max_attempts=2,
        )

        if answer:
            return answer

    return None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return {
        "status": "ok",
        "primary_model": PRIMARY_MODEL,
        "backup_model": BACKUP_MODEL,
        "api_key_configured": client is not None,
    }


@app.route("/ask", methods=["POST"])
def ask():
    if client is None:
        return Response(
            "MRG AI is not configured. Please check GEMINI_API_KEY "
            "in your Render environment settings.",
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
        history = json.loads(request.form.get("history", "[]"))

        if not isinstance(history, (list, dict)):
            history = []

    except (json.JSONDecodeError, TypeError):
        history = []

    prompt = build_prompt(question, history)

    def stream_answer():
        try:
            answer = generate_answer(prompt)

            if answer:
                yield answer
            else:
                yield (
                    "I'm sorry, bestfriend! 💙 MRG AI is experiencing "
                    "heavy demand right now. Please wait a little and "
                    "try your question again."
                )

        except Exception as error:
            print(f"UNEXPECTED ANSWER ERROR: {error}")

            yield (
                "MRG AI encountered a temporary problem. 💙 "
                "Please try again shortly."
            )

    return Response(
        stream_answer(),
        mimetype="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
    )import os
import json
import time
import random
from datetime import datetime
from zoneinfo import ZoneInfo

from flask import Flask, request, render_template, Response
from google import genai

app = Flask(__name__)

# Gemini configuration
API_KEY = os.environ.get("GEMINI_API_KEY")

PRIMARY_MODEL = "gemini-3.5-flash-lite"
BACKUP_MODEL = "gemini-3.1-flash-lite"

TIMEZONE = ZoneInfo("Asia/Kolkata")

client = genai.Client(api_key=API_KEY) if API_KEY else None


def build_prompt(question, history):
    today = datetime.now(TIMEZONE).strftime("%d %B %Y")

    return f"""
You are MRG AI, a helpful, friendly, intelligent AI assistant.

IDENTITY:
- Founder: Manju R Gowda.
- Developed by Manju R Gowda.

TODAY'S DATE: {today}

INSTRUCTIONS:
1. Answer the user's actual question clearly.
2. Use simple, natural language.
3. Use conversation history when relevant.
4. Never invent facts, sources, or links.
5. Answer educational and general-knowledge questions directly.
6. Do not claim to have searched the internet.
7. Explain uncertainty about information you cannot verify.
8. Format answers clearly.
9. Do not repeat the same information unnecessarily.

CONVERSATION HISTORY:
{json.dumps(history, ensure_ascii=False)}

USER QUESTION:
{question}

Provide one clear, useful answer.
"""


def is_temporary_error(error):
    """Identify errors that may resolve after waiting."""
    message = str(error).lower()

    temporary_markers = [
        "503",
        "unavailable",
        "429",
        "resource_exhausted",
        "500",
        "502",
        "504",
        "timeout",
        "timed out",
        "high demand",
        "temporarily",
        "internal server error",
    ]

    return any(marker in message for marker in temporary_markers)


def generate_with_retry(model, prompt, max_attempts=2):
    """
    Retry temporary failures with increasing delays.
    Return the answer on success, or None on failure.
    """
    if client is None:
        return None

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )

            answer = getattr(response, "text", None)

            if answer and answer.strip():
                print(f"SUCCESS: {model}")
                return answer.strip()

            print(f"EMPTY RESPONSE: {model}")

        except Exception as error:
            print(
                f"GEMINI ERROR ({model}, "
                f"attempt {attempt + 1}/{max_attempts}): {error}"
            )

            if not is_temporary_error(error):
                break

        if attempt < max_attempts - 1:
            # Wait before retrying a temporary failure.
            delay = min(2 ** (attempt + 1), 8)
            delay += random.uniform(0, 1)
            print(f"Retrying {model} in approximately {delay:.1f} seconds.")
            time.sleep(delay)

    return None


def generate_answer(prompt):
    """Try the primary model, then the backup model."""

    answer = generate_with_retry(
        PRIMARY_MODEL,
        prompt,
        max_attempts=2,
    )

    if answer:
        return answer

    if BACKUP_MODEL and BACKUP_MODEL != PRIMARY_MODEL:
        print(f"PRIMARY MODEL FAILED. Trying {BACKUP_MODEL}.")

        answer = generate_with_retry(
            BACKUP_MODEL,
            prompt,
            max_attempts=2,
        )

        if answer:
            return answer

    return None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return {
        "status": "ok",
        "primary_model": PRIMARY_MODEL,
        "backup_model": BACKUP_MODEL,
        "api_key_configured": client is not None,
    }


@app.route("/ask", methods=["POST"])
def ask():
    if client is None:
        return Response(
            "MRG AI is not configured. Please check GEMINI_API_KEY "
            "in your Render environment settings.",
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
        history = json.loads(request.form.get("history", "[]"))

        if not isinstance(history, (list, dict)):
            history = []

    except (json.JSONDecodeError, TypeError):
        history = []

    prompt = build_prompt(question, history)

    def stream_answer():
        try:
            answer = generate_answer(prompt)

            if answer:
                yield answer
            else:
                yield (
                    "I'm sorry, bestfriend! 💙 MRG AI is experiencing "
                    "heavy demand right now. Please wait a little and "
                    "try your question again."
                )

        except Exception as error:
            print(f"UNEXPECTED ANSWER ERROR: {error}")

            yield (
                "MRG AI encountered a temporary problem. 💙 "
                "Please try again shortly."
            )

    return Response(
        stream_answer(),
        mimetype="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
    )
