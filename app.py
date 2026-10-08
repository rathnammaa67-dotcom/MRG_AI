from flask import Flask, render_template, request, Response, stream_with_context
from google import genai
import os

app = Flask(__name__)

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():

    question = request.form.get("question", "").strip()

    if not question:
        return "Please enter a question.", 400

    def generate():

        try:
            response = client.models.generate_content_stream(
                model="gemini-3.1-flash-lite",
                contents=question,
                config={
                    "max_output_tokens": 300,
                    "temperature": 0.3
                }
            )

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception:
            yield (
                "⚠️ Manju's server is currently busy. "
                "Please wait a few seconds and try again."
            )

    return Response(
        stream_with_context(generate()),
        mimetype="text/plain"
    )


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
