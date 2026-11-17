"""Tourism chatbot API: TF-IDF similarity matches a traveller's question to destinations."""
import json
import re
from flask import Flask, jsonify, request, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, static_folder="static")
with open("destinations.json", encoding="utf-8") as f:
    PLACES = json.load(f)

docs = [f"{p['name']} {p['region']} {p['tags']} {p['info']}" for p in PLACES]
def tokenize(text):
    """Lowercase words with a simple plural strip so 'beaches' matches 'beach'."""
    words = re.findall(r"[a-z]+", text.lower())
    return [w[:-2] if w.endswith("es") and len(w) > 4 else w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words]


vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False)
matrix = vectorizer.fit_transform(docs)

GREETING = re.compile(r"\b(hi|hello|salam|hey|assalam)\b", re.I)
SEASON = re.compile(r"\b(when|best time|season|weather)\b", re.I)


def reply(message: str) -> dict:
    message = message.strip()
    if not message:
        return {"text": "Ask me about places in Pakistan, like 'quiet mountain trip for a family'."}
    if GREETING.search(message) and len(message.split()) <= 3:
        return {"text": "Assalam o Alaikum! Tell me what kind of trip you want: mountains, beaches, history or food."}

    lowered = message.lower()
    for p in PLACES:  # direct questions about a named place
        key = p["name"].lower().split(" and ")[0].split()[0].replace("-daro", "")
        if key in lowered:
            when = f" Best time: {p['best_time']}." if SEASON.search(message) else ""
            return {"text": f"{p['info']}{when}", "places": [p["name"]]}

    scores = cosine_similarity(vectorizer.transform([message]), matrix)[0]
    ranked = scores.argsort()[::-1][:3]
    ranked = [i for i in ranked if scores[i] > 0.05]
    if not ranked:
        return {"text": "I could not match that. Try words like mountains, lake, beach, history, snow or family."}
    lines = [f"{PLACES[i]['name']} ({PLACES[i]['region']}): best in {PLACES[i]['best_time']}." for i in ranked]
    return {"text": "Here are my top picks:\n" + "\n".join(lines), "places": [PLACES[i]["name"] for i in ranked]}


@app.post("/api/chat")
def chat():
    data = request.get_json(silent=True) or {}
    return jsonify(reply(data.get("message", "")))


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    app.run(debug=True)
