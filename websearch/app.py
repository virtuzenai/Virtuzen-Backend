import requests
import google.generativeai as genai
from flask import Flask, request, jsonify
import json
import time
from hashlib import md5
from datetime import datetime, timedelta

# API Keys (set these in Render's Environment Variables)
GOOGLE_API_KEY = "AIzaSyDeeLRl61zVi52zFe41jCjCbNUvCdbFPM8"
SEARCH_ENGINE_ID = "b3c8857ed6b5744fa"
GEMINI_API_KEY = "AIzaSyBeMGH5FDILyFBiFkiDpIX1srFaZ5ELR8M"

# Cache setup
CACHE_FILE = "search_cache.json"
CACHE = {}
CACHE_EXPIRY_HOURS = 24

# Flask app
app = Flask(__name__)

# Initialize Gemini
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    app.logger.error(f"Gemini setup failed: {e}")
    raise

# Load cache
def load_cache():
    global CACHE
    try:
        with open(CACHE_FILE, "r") as f:
            raw_cache = json.load(f)
        now = datetime.now().timestamp()
        CACHE = {}
        for k, v in raw_cache.items():
            if isinstance(v, dict) and "timestamp" in v:
                if now - v["timestamp"] < CACHE_EXPIRY_HOURS * 3600:
                    CACHE[k] = v
            else:
                app.logger.info("Skipping old cache entry.")
    except (FileNotFoundError, json.JSONDecodeError):
        CACHE = {}

# Save cache
def save_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(CACHE, f)

# Search function
def execute_search_query(search_term):
    cache_key = md5(search_term.encode()).hexdigest()
    if cache_key in CACHE:
        return CACHE[cache_key]["data"]

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": search_term,
        "sort": "date",
        "num": 5,
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        CACHE[cache_key] = {
            "data": data,
            "timestamp": datetime.now().timestamp()
        }
        save_cache()
        return data
    except requests.RequestException as e:
        app.logger.error(f"Search failed: {e}")
        return None

# Rank results
def rank_results(items):
    if not items:
        return items
    trusted_domains = ["wikipedia.org", "britannica.com", "edu", "gov", "bbc.com"]
    banned_domains = ["signup", "login", "advertise"]
    ranked = []
    for index, item in enumerate(items):
        score = 1.0
        link = item.get("link", "").lower()
        snippet = item.get("snippet", "")
        if any(bad in link for bad in banned_domains) or len(snippet) < 20:
            continue
        if any(good in link for good in trusted_domains):
            score += 0.6
        if len(snippet) > 50:
            score += 0.4
        ranked.append((score, index, item))
    ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [item for _, _, item in ranked]

# Generate summary
def generate_summary(search_term, search_data):
    cache_key = md5((search_term + json.dumps(search_data)).encode()).hexdigest()
    if cache_key in CACHE:
        return CACHE[cache_key]["summary"]

    if not search_data or "items" not in search_data:
        return "Nada worth summing up here!"

    items = rank_results(search_data.get("items", [])[:5])
    if not items:
        return "The web’s being stingy—nothing solid to share."

    aggregated_details = ""
    for item in items:
        title = item.get("title", "No Title")
        snippet = item.get("snippet", "No snippet")[:100]
        link = item.get("link", "No link")
        aggregated_details += f"Title: {title}\nSnippet: {snippet}\nLink: {link}\n\n"

    prompt = (
        f"User Query: {search_term}\n"
        f"Search Results:\n{aggregated_details}\n"
        "Hey, imagine we’re chilling on a porch swing, swapping stories. "
        "Wrap these results into a short, fun tale like you’re telling me something cool you just found. "
        "Hit the high points, keep it lively, and skip the dry stuff unless it’s a must. "
        "End with a ‘what’s your take?’ to keep it rolling!"
    )

    try:
        response = model.generate_content(prompt)
        summary = response.text
        CACHE[cache_key] = {
            "summary": summary,
            "timestamp": datetime.now().timestamp()
        }
        save_cache()
        return summary
    except Exception as e:
        app.logger.error(f"Summary failed: {e}")
        return "Couldn’t spin a yarn this time."

# API endpoint
@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '').strip()
    if not query or len(query) < 3:
        return jsonify({"error": "Hey, give me a bit more to work with—query’s too short!"}), 400

    search_results = execute_search_query(query)
    if not search_results:
        return jsonify({"error": "Aw, the web’s not cooperating today."}), 500

    summary = generate_summary(query, search_results)
    ranked_items = rank_results(search_results.get("items", [])[:5])
    results = [{"title": item["title"], "snippet": item.get("snippet", "No snippet"), "link": item["link"]} for item in ranked_items]

    return jsonify({
        "summary": summary,
        "results": results
    })

if __name__ == "__main__":
    load_cache()
    app.run(host="0.0.0.0", port=8080)
