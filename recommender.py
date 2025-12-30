# recommender.py
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class MovieRecommender:
    def __init__(self, movies_csv='data/movies.csv', ratings_csv='data/ratings.csv'):
        self.movies = pd.read_csv(movies_csv)          # expects movieId,title
        self.ratings = pd.read_csv(ratings_csv)        # expects userId,movieId,rating
        self.movie_user_matrix = None
        self.similarity = None
        self._prepare()

    def _prepare(self):
        # create user-item pivot table (movies x users)
        pivot = self.ratings.pivot_table(index='movieId', columns='userId', values='rating').fillna(0)
        self.movie_user_matrix = pivot
        # compute item-item cosine similarity
        self.similarity = cosine_similarity(pivot.values)
        # mapping movieId -> row index in the matrix
        self.movieid_to_index = {movie_id: idx for idx, movie_id in enumerate(pivot.index)}
        self.index_to_movieid = {idx: movie_id for movie_id, idx in self.movieid_to_index.items()}

    def recommend_from_titles(self, titles, top_n=5):
        """
        titles: list of movie titles user likes (strings)
        returns: list of recommended movieIds sorted by score
        """
        # find movieIds for the provided titles (best-effort, case-insensitive)
        title_to_row = {row['title'].lower(): row['movieId'] for _, row in self.movies.iterrows()}
        selected_ids = []
        for t in titles:
            t_lower = t.strip().lower()
            if t_lower in title_to_row:
                selected_ids.append(title_to_row[t_lower])
            else:
                # fallback: try substring match
                matches = [m for m in title_to_row if t_lower in m]
                if matches:
                    selected_ids.append(title_to_row[matches[0]])

        # if none found, return empty
        if not selected_ids:
            return []

        # for each selected movie, get similarity vector
        sim_vectors = []
        for mid in selected_ids:
            if mid in self.movieid_to_index:
                idx = self.movieid_to_index[mid]
                sim_vectors.append(self.similarity[idx])
        if not sim_vectors:
            return []

        # aggregate similarity (average)
        agg = np.mean(sim_vectors, axis=0)

        # create (movie_index, score) pairs and sort
        scored = list(enumerate(agg))
        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)

        # build recommendations skipping selected movies and low-score items
        recs = []
        selected_indices = {self.movieid_to_index[mid] for mid in selected_ids if mid in self.movieid_to_index}
        for idx, score in scored_sorted:
            if idx in selected_indices:
                continue
            movie_id = self.index_to_movieid[idx]
            title = self.movies.loc[self.movies['movieId']==movie_id, 'title'].values
            if len(title):
                recs.append({'movieId': int(movie_id), 'title': title[0], 'score': float(score)})
            if len(recs) >= top_n:
                break
        return recs

# helper to save/load model
def build_and_save_model(movies_csv='data/movies.csv', ratings_csv='data/ratings.csv', out='model.pkl'):
    rec = MovieRecommender(movies_csv, ratings_csv)
    with open(out, 'wb') as f:
        pickle.dump(rec, f)
    print('Model saved to', out)
