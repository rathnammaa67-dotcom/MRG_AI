import os
import json
import time
from datetime import datetime

from flask import Flask, request, render_template, Response, stream_with_context
from google import genai
from ddgs import DDGS

app = Flask(__name__)

API_KEY = os.environ.get("GEMINI_API_KEY")

# Primary model and backup model
MODEL = "gemini-3.1-flash-lite"
BACKUP_MODEL = "gemini-2.5-flash"

client = genai.Client(api_key=API_KEY) if API_KEY else None


def search_web(question):
    """Try web search quickly; continue without it if it fails."""
    try:
        def search_web(question):
            results = list(
                ddgs.text(
                    query=question + " official information",
                    region="in-en",
                    safesearch="moderate",
                    max_results=4,
                )
            )

        formatted = []

        for item in results:
            title = item.get("title", "")
            body = item.get("body", "")
            url = item.get("href", "")

            if title and body:
                formatted.append(
                    f"Title: {title}\n"
                    f"Information: {body}\n"
                    f"Source: {url}"
                )

        return "\n\n".join(formatted)

    except Exception as error:
        print("WEB SEARCH ERROR:", error)
        return ""


def build_prompt(question, history, search_results):
    today = datetime.now().strftime("%d %B %Y")

    if search_results:
        web_information = search_results
    else:
        web_information = (
            "Web search was unavailable. Do not pretend to have "
            "verified current information. Clearly state uncertainty "
            "for facts that may have changed."
        )

    return f"""
You are MRG AI, a helpful and friendly AI assistant.

IDENTITY:
- Your founder is Manju R Gowda.
- You were trained and developed by Manjur Gowda.

TODAY'S DATE: {today}

RULES:
1. Answer the user's actual question clearly.
2. For changing facts, use the search results when available.
3. Prefer official sources and check publication dates.
4. Never invent facts, links, or sources.
5. If current information cannot be verified, say so.
6. Use simple, friendly language.
7. Use conversation history only as context.

CONVERSATION HISTORY:
{json.dumps(history, ensure_ascii=False)}

WEB SEARCH RESULTS:
{web_information}

USER QUESTION:
{question}

Give one clear answer.
"""


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL,
        "api_key_configured": client is not None,
    }


def generate_answer(model, prompt):
    """Try a model with short retries for temporary server errors."""
    last_error = None

    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )

            answer = getattr(response, "text", None)

            if answer:
                return answer

            last_error = "The model returned an empty answer."

        except Exception as error:
            last_error = str(error)
            print(f"GEMINI ERROR ({model}, attempt {attempt + 1}):", error)

            # Retry only likely temporary service/rate-limit errors.
            error_text = last_error.lower()
            temporary = any(
                term in error_text
                for term in (
                    "503",
                    "unavailable",
                    "429",
                    "resource_exhausted",
                    "500",
                    "timeout",
                    "temporarily",
                )
            )

            if not temporary:
                break

            if attempt == 0:
                time.sleep(1.5)

    return None


@app.route("/ask", methods=["POST"])
def ask():
    if client is None:
        return Response(
            "The AI is not configured. Please check GEMINI_API_KEY in Render.",
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

    # Web search must not prevent the AI from answering.
    search_results = search_web(question)
    prompt = build_prompt(question, history, search_results)

    def generate():
        answer = generate_answer(MODEL, prompt)

        if not answer and BACKUP_MODEL != MODEL:
            print("Trying backup model:", BACKUP_MODEL)
            answer = generate_answer(BACKUP_MODEL, prompt)

        if answer:
            yield answer
        else:
            yield (
                "MRG AI is temporarily busy. 💙 "
                "Please try again in a minute. If this continues, "
                "check the Gemini model availability and API quota "
                "in Google AI Studio."
            )

    return Response(
        stream_with_context(generate()),
        mimetype="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache"},
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
    )
