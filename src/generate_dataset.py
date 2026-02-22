# src/generate_dataset.py

import pandas as pd
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

movies_path = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
credits_path = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")

movies = pd.read_csv(movies_path)
credits = pd.read_csv(credits_path)

credits = credits[["movie_id", "cast", "crew"]]

df = movies.merge(
    credits,
    left_on="id",
    right_on="movie_id"
)

# -----------------------------
# Safe JSON Parser
# -----------------------------

def safe_json(text):
    try:
        return json.loads(text)
    except:
        return []

# -----------------------------
# Feature Extraction
# -----------------------------

def get_genres(g):
    g = safe_json(g)
    if len(g) == 0:
        return "Unknown", "Unknown"
    if len(g) == 1:
        return g[0]["name"], "Unknown"
    return g[0]["name"], g[1]["name"]

def get_main_actor(c):
    c = safe_json(c)
    return c[0]["name"] if c else "Unknown"

def get_director(c):
    c = safe_json(c)
    for person in c:
        if person.get("job") == "Director":
            return person.get("name")
    return "Unknown"

def get_country(p):
    p = safe_json(p)
    return p[0]["name"] if p else "Unknown"

df["Genre"], df["Second_Genre"] = zip(*df["genres"].apply(get_genres))
df["Main_Actor"] = df["cast"].apply(get_main_actor)
df["Director"] = df["crew"].apply(get_director)
df["Production_Country"] = df["production_countries"].apply(get_country)

df["Release_Year"] = pd.to_datetime(
    df["release_date"], errors="coerce"
).dt.year

# -----------------------------
# Category Builders
# -----------------------------

def budget_cat(b):
    if b < 10_000_000:
        return "Low"
    elif b < 100_000_000:
        return "Medium"
    else:
        return "High"

def revenue_cat(r):
    if r < 50_000_000:
        return "Low"
    elif r < 300_000_000:
        return "Medium"
    else:
        return "High"

def runtime_cat(rt):
    if rt < 90:
        return "Short"
    elif rt < 140:
        return "Medium"
    else:
        return "Long"

def vote_count_cat(v):
    if v < 500:
        return "Low"
    elif v < 2000:
        return "Medium"
    else:
        return "High"

def popularity_cat(p):
    if p < 10:
        return "Low"
    elif p < 30:
        return "Medium"
    else:
        return "High"

def rating_cat(r):
    if r < 5:
        return "Low"
    elif r < 7:
        return "Medium"
    else:
        return "High"

df["Budget_Category"] = df["budget"].apply(budget_cat)
df["Revenue_Category"] = df["revenue"].apply(revenue_cat)
df["Runtime_Category"] = df["runtime"].apply(runtime_cat)
df["Vote_Count_Category"] = df["vote_count"].apply(vote_count_cat)
df["Popularity_Category"] = df["popularity"].apply(popularity_cat)
df["Rating_Category"] = df["vote_average"].apply(rating_cat)

df["Title"] = df["title"]
df["Language"] = df["original_language"]

final_df = df[
    [
        "Title",
        "Genre",
        "Second_Genre",
        "Release_Year",
        "Budget_Category",
        "Revenue_Category",
        "Runtime_Category",
        "Rating_Category",
        "Vote_Count_Category",
        "Popularity_Category",
        "Language",
        "Production_Country",
        "Main_Actor",
        "Director",
    ]
]

final_df = final_df.dropna()
final_df = final_df.drop_duplicates()

# Remove incomplete / unknown data
final_df = final_df[
    (final_df["Director"] != "Unknown") &
    (final_df["Main_Actor"] != "Unknown") &
    (final_df["Genre"] != "Unknown") &
    (final_df["Second_Genre"] != "Unknown") &
    (final_df["Production_Country"] != "Unknown")
]

save_path = os.path.join(DATA_DIR, "movie_cleaned.csv")
final_df.to_csv(save_path, index=False)

print("Dataset regenerated successfully.")
print("Total movies:", len(final_df))