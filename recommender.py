"""Movie Recommendation System (MovieLens 100K).

Content-based (genre cosine similarity), collaborative filtering (SVD from the
Surprise library) and a hybrid blend of the two.

Usage:
    python recommender.py --user 196 --top 10
    python recommender.py --evaluate
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

DATA_DIR = Path("data/ml-100k")
GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


# Steps 2-3: collect and preprocess data
def load_data(data_dir: Path = DATA_DIR):
    ratings_path, items_path = data_dir / "u.data", data_dir / "u.item"
    if not ratings_path.exists() or not items_path.exists():
        raise FileNotFoundError(
            f"MovieLens 100K not found in {data_dir}. Download it from "
            "https://www.kaggle.com/datasets/grouplens/movielens-100k-dataset "
            "and place u.data and u.item there."
        )
    ratings = pd.read_csv(
        ratings_path, sep="\t", names=["user_id", "movie_id", "rating", "timestamp"]
    )
    movies = pd.read_csv(
        items_path, sep="|", encoding="latin-1", header=None,
        names=["movie_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRES,
    )
    movies = movies.drop(columns=["video_release_date", "imdb_url"])
    movies["release_date"] = movies["release_date"].fillna("unknown")
    ratings = ratings.dropna().drop_duplicates(["user_id", "movie_id"])
    return ratings, movies


def explore(ratings: pd.DataFrame, movies: pd.DataFrame) -> None:
    """Step 4: quick exploratory analysis of user-item interactions."""
    n_users, n_items = ratings.user_id.nunique(), ratings.movie_id.nunique()
    sparsity = 1 - len(ratings) / (n_users * n_items)
    print(f"Users: {n_users}  Movies: {n_items}  Ratings: {len(ratings)}  Sparsity: {sparsity:.2%}")
    print("Rating distribution:\n", ratings.rating.value_counts().sort_index())
    top = ratings.groupby("movie_id").size().nlargest(5).index
    print("Most rated:", movies.set_index("movie_id").loc[top, "title"].tolist())


# Step 5-6: similarity models and engine
def content_scores(user_id, ratings, movies) -> np.ndarray:
    """Mean genre similarity between every movie and the movies the user liked (>=4)."""
    genre_matrix = movies[GENRES].values
    liked = ratings[(ratings.user_id == user_id) & (ratings.rating >= 4)].movie_id
    liked_idx = movies.index[movies.movie_id.isin(liked)]
    if len(liked_idx) == 0:
        return np.zeros(len(movies))
    return cosine_similarity(genre_matrix[liked_idx], genre_matrix).mean(axis=0)


def train_svd(ratings: pd.DataFrame):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[["user_id", "movie_id", "rating"]], reader)
    trainset = data.build_full_trainset()
    algo = SVD(n_factors=50, random_state=42)
    algo.fit(trainset)
    return algo


# Step 7: generate recommendations
def recommend(user_id, ratings, movies, algo, top_n=10, mode="hybrid", alpha=0.7):
    seen = set(ratings[ratings.user_id == user_id].movie_id)
    cf = np.array([algo.predict(user_id, m).est for m in movies.movie_id]) / 5.0
    cb = content_scores(user_id, ratings, movies)
    if mode == "content":
        score = cb
    elif mode == "collaborative":
        score = cf
    else:
        score = alpha * cf + (1 - alpha) * cb
    out = movies[["movie_id", "title"]].copy()
    out["score"] = score
    out = out[~out.movie_id.isin(seen)]
    return out.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)


# Step 8: evaluate
def evaluate(ratings: pd.DataFrame) -> None:
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[["user_id", "movie_id", "rating"]], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    algo = SVD(n_factors=50, random_state=42)
    algo.fit(trainset)
    predictions = algo.test(testset)
    print("SVD hold-out results:")
    accuracy.rmse(predictions)
    accuracy.mae(predictions)


def main():
    parser = argparse.ArgumentParser(description="Movie Recommendation System")
    parser.add_argument("--user", type=int, default=196, help="user id (1-943)")
    parser.add_argument("--top", type=int, default=10, help="number of recommendations")
    parser.add_argument("--mode", choices=["content", "collaborative", "hybrid"], default="hybrid")
    parser.add_argument("--evaluate", action="store_true", help="report RMSE / MAE")
    parser.add_argument("--explore", action="store_true", help="print dataset summary")
    args = parser.parse_args()

    ratings, movies = load_data()
    if args.explore:
        explore(ratings, movies)
    if args.evaluate:
        evaluate(ratings)
        return
    algo = train_svd(ratings)
    print(f"\nTop {args.top} {args.mode} recommendations for user {args.user}:\n")
    print(recommend(args.user, ratings, movies, algo, args.top, args.mode).to_string(index=False))


if __name__ == "__main__":
    main()
