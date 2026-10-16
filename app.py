import os
import json
import time
from datetime import datetime

from flask import Flask, request, render_template, Response, stream_with_context
from google import genai
from ddgs import DDGS

app = Flask(__name__)

# API configuration
API_KEY = os.environ.get("GEMINI_API_KEY")

MODEL = "gemini-3.8-flash"
BACKUP_MODEL = "gemini-3.7-flash"

client = genai.Client(api_key=API_KEY) if API_KEY else None


# --------------------------------------------------
# WEB SEARCH
# --------------------------------------------------

def search_web(question):
    """Search the web and return formatted results."""

    try:
        with DDGS(timeout=10) as ddgs:
            results = list(
                ddgs.text(
                    query=question + " latest news",
                    region="in-en",
                    safesearch="moderate",
                    max_results=5,
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

        search_results = "\n\n".join(formatted)

        if not search_results:
            print("WEB SEARCH: No results returned.")

        return search_results

    except Exception as error:
        print("WEB SEARCH ERROR:", error)
        return ""


# --------------------------------------------------
# PROMPT
# --------------------------------------------------

def build_prompt(question, history, search_results):

    today = datetime.now().strftime("%d %B %Y")

    if search_results:
        web_information = search_results
    else:
        web_information = (
            "No verified web search results are available. "
            "Do not claim that live information was searched. "
            "Be transparent when current facts cannot be verified."
        )

    return f"""
You are MRG AI, a helpful, friendly, and intelligent AI assistant.

IDENTITY:
- Your founder is Manju R Gowda.
- You were trained and developed by Manjur Gowda.

TODAY'S DATE: {today}

IMPORTANT RULES:
1. Answer the user's actual question clearly.
2. Use provided web search results for current information.
3. Never pretend to have searched the web if no results are available.
4. Never invent news, facts, links, or sources.
5. Prefer reliable and official sources when available.
6. Distinguish verified facts from uncertainty.
7. Use simple, friendly language.
8. Use conversation history only as context.
9. Do not describe old information as breaking news.
10. Answer in a helpful and natural way.

CONVERSATION HISTORY:
{json.dumps(history, ensure_ascii=False)}

WEB SEARCH RESULTS:
{web_information}

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
    }


# --------------------------------------------------
# GEMINI RESPONSE
# --------------------------------------------------

def generate_answer(model, prompt):
    """Generate an answer with a retry for temporary errors."""

    if client is None:
        return None

    for attempt in range(2):

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )

            answer = getattr(response, "text", None)

            if answer:
                return answer

            print(f"GEMINI ERROR ({model}): Empty response.")

        except Exception as error:

            error_text = str(error).lower()

            print(
                f"GEMINI ERROR ({model}, attempt {attempt + 1}):",
                error,
            )

            temporary_errors = (
                "503",
                "unavailable",
                "429",
                "resource_exhausted",
                "500",
                "timeout",
                "temporarily",
            )

            if not any(term in error_text for term in temporary_errors):
                break

            if attempt == 0:
                time.sleep(2)

    return None


# --------------------------------------------------
# CHAT ENDPOINT
# --------------------------------------------------

@app.route("/ask", methods=["POST"])
def ask():

    if client is None:
        return Response(
            "MRG AI is not configured. Check GEMINI_API_KEY in Render.",
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

    # Search the web before generating the answer.
    search_results = search_web(question)

    prompt = build_prompt(
        question,
        history,
        search_results,
    )

    def generate():

        answer = generate_answer(
            MODEL,
            prompt,
        )

        # Try the backup model if the primary model fails.
        if not answer and BACKUP_MODEL != MODEL:

            print("Trying backup model:", BACKUP_MODEL)

            answer = generate_answer(
                BACKUP_MODEL,
                prompt,
            )

        if answer:
            yield answer

        else:
            yield (
                "MRG AI is temporarily busy. 💙 "
                "Please try again shortly. If the problem continues, "
                "check your Gemini API quota and model availability "
                "in Google AI Studio."
            )

    return Response(
        stream_with_context(generate()),
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
    
