import os
import json
from datetime import datetime
from flask import Flask, request, render_template, Response, stream_with_context
from google import genai
from ddgs import DDGS

app = Flask(__name__)

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = "gemini-3.1-flash-lite"

client = genai.Client(api_key=API_KEY) if API_KEY else None


def search_web(question):
    """Search the web for information relevant to the question."""
    try:
        with DDGS(timeout=10) as ddgs:
            results = list(
                ddgs.text(
                    query=question + " latest official information",
                    region="in-en",
                    safesearch="moderate",
                    max_results=6,
                )
            )

        if not results:
            return "No web search results were found."

        formatted = []
        for item in results:
            title = item.get("title", "")
            body = item.get("body", "")
            url = item.get("href", "")

            if title and body and url:
                formatted.append(
                    f"Title: {title}\n"
                    f"Information: {body}\n"
                    f"Source: {url}"
                )

        return "\n\n".join(formatted) or "No useful results found."

    except Exception as error:
        print("WEB SEARCH ERROR:", error)
        return "Web search is temporarily unavailable."


def build_prompt(question, history, search_results):
    today = datetime.now().strftime("%d %B %Y")

    return f"""
You are MRG AI, a helpful AI assistant.

IDENTITY:
- Your founder is Manju R Gowda.
- You were trained and developed by Manjur Gowda.

TODAY'S DATE: {today}

IMPORTANT RULES:
1. Answer the user's actual question clearly and accurately.
2. For questions about current office holders, news, prices, sports,
   weather, laws, or other changing facts, examine the web search
   results before answering.
3. Prefer official government websites, official organizations,
   and reputable news sources over Wikipedia, blogs, or old articles.
4. Check dates and distinguish old information from current information.
5. Search results can be outdated or incorrect. Never assume that
   the first result is correct.
6. If the available results do not establish the current answer,
   clearly say that you could not verify it. Never guess.
7. Do not invent sources, dates, names, or facts.
8. Include relevant source links in your answer when available.
9. Use simple, friendly language.
10. Do not repeat the same answer twice.
11. Conversation history is context, not proof that a claim is true.

CONVERSATION HISTORY:
{history}

WEB SEARCH RESULTS:
{search_results}

USER QUESTION:
{question}

Provide one clear answer. Use the search results critically and
state uncertainty when the evidence is insufficient.
"""


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    return {"status": "ok", "model": MODEL}


@app.route("/ask", methods=["POST"])
def ask():
    if client is None:
        return Response(
            "Error: GEMINI_API_KEY is not configured in the server settings.",
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

    raw_history = request.form.get("history", "")

    try:
        history = json.loads(raw_history) if raw_history else []
        if not isinstance(history, (list, dict)):
            history = []
    except (json.JSONDecodeError, TypeError):
        history = []

    search_results = search_web(question)
    prompt = build_prompt(question, history, search_results)

    def generate():
        try:
            response = client.models.generate_content_stream(
                model=MODEL,
                contents=prompt,
            )

            for chunk in response:
                text = getattr(chunk, "text", None)
                if text:
                    yield text

        except Exception as error:
            print("GEMINI ERROR:", error)
            yield (
                "\n\nSorry, I couldn't complete the answer. "
                "Please try again in a moment."
            )

    return Response(
        stream_with_context(generate()),
        mimetype="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-cache"},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))


