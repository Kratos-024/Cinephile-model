import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# with open("movie_recommender.pkl", "rb") as f:
#     saved = pickle.load(f)

# model = saved["model"]
# vectorizer = saved["vectorizer"]
# movie_index = saved["movie_index"] 

# movies_df = pd.read_csv("movie_copy_2000s.tsv", sep=",")  # try comma first
app = FastAPI(title="Cinephile Recommender")

class MovieRequest(BaseModel):
    movie_title: str



with open("movie_recommender_2.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
movie_index = data["movie_index"]
best_final_df_with_only_movie_after_2000s = data["dataframe"]
X = data["vectorizer"].transform(best_final_df_with_only_movie_after_2000s["combined"]) 


def recommend(title):
    title = title.lower()
    if title not in movie_index:
        return "Movie not found."

    idx = movie_index[title]
    distances, indices = model.kneighbors(X[idx], n_neighbors=5)

    recommended = []
    for i in indices[0][1:]:
        recommended.append(best_final_df_with_only_movie_after_2000s.iloc[i]['title'])

    return {"recommended":recommended}


@app.post("/recommend")
def recommend_movie(req: MovieRequest):
    title = req.movie_title.lower()

    recommended = recommend(title)

    return {"recommendations": recommended}
