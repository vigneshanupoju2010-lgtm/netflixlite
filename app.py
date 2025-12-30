# app.py
from flask import Flask, request, render_template, jsonify
import pickle
import requests
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()


TMDB_API_KEY = os.environ.get('TMDB_API_KEY')
TMDB_IMAGE_BASE = 'https://image.tmdb.org/t/p/w300'

app = Flask(__name__)

# load recommender
with open('model.pkl', 'rb') as f:
    recommender = pickle.load(f)


def get_poster_from_tmdb(title):
    api_key = TMDB_API_KEY
    if not api_key:
        return None

    # Clean title (removes year like (2008))
    clean_title = title.split("(")[0].strip()

    # TMDB search
    search_url = (
        f"https://api.themoviedb.org/3/search/movie"
        f"?api_key={api_key}&query={clean_title}"
    )
    response = requests.get(search_url).json()

    results = response.get("results", [])
    if results:
        poster_path = results[0].get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"

    return None

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    titles = data.get('titles', [])
    recs = recommender.recommend_from_titles(titles, top_n=5)

    # attach poster urls
    for r in recs:
        r['poster'] = get_poster_from_tmdb(r['title'])

    return jsonify({'input': titles, 'recommendations': recs})


@app.route("/titles", methods=["GET"])
def titles():
    try:
        titles_list = recommender.movies['title'].astype(str).tolist()
        return jsonify({"titles": titles_list})
    except:
        return jsonify({"titles": []})

_links_df = pd.read_csv("data/links.csv")
_links_df['tmdbId'] = pd.to_numeric(_links_df['tmdbId'], errors='coerce').fillna(0).astype(int)
movie_to_tmdb = dict(zip(_links_df['movieId'], _links_df['tmdbId']))


@app.route("/movie/<int:movie_id>", methods=["GET"])
def movie_details(movie_id):

    if not TMDB_API_KEY:
        return jsonify({"error": "TMDB_API_KEY missing"}), 500

    tmdb_id = movie_to_tmdb.get(movie_id)
    if not tmdb_id:
        return jsonify({"error": "No TMDB ID found"}), 404

    base = "https://api.themoviedb.org/3"
    params = {"api_key": TMDB_API_KEY, "language": "en-US"}

    # Movie details
    details = requests.get(f"{base}/movie/{tmdb_id}", params=params).json()

    # Videos (for trailer)
    videos = requests.get(f"{base}/movie/{tmdb_id}/videos", params=params).json()
    youtube_key = None
    for v in videos.get("results", []):
        if v["site"] == "YouTube" and v["type"].lower() in ("trailer", "teaser"):
            youtube_key = v["key"]
            break

    poster_path = details.get("poster_path")
    poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None

    return jsonify({
        "movieId": movie_id,
        "title": details.get("title"),
        "overview": details.get("overview"),
        "genres": [g["name"] for g in details.get("genres", [])],
        "release_date": details.get("release_date"),
        "tmdb_rating": details.get("vote_average"),
        "poster": poster_url,
        "youtube_key": youtube_key
    })

if __name__ == '__main__':
    app.run(debug=True)
