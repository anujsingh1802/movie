"""Advanced movie assistant extensions.

Features:
1) Natural-language + voice query handling.
2) Recommendation models (content-based + optional collaborative filtering).
3) Interactive visualizations (Plotly + PyVis network).
4) Explainability helpers (LIME + information gain text).
"""

from __future__ import annotations

import importlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "movie_cleaned.csv"


@dataclass
class RecommenderArtifacts:
    df: pd.DataFrame
    tfidf_matrix: Any
    similarity_matrix: np.ndarray


def _optional_import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(
            f"Optional dependency '{module_name}' is required for this feature. "
            f"Install it and retry. Original error: {exc}"
        ) from exc


def load_movies(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.fillna("Unknown")
    for col in df.columns:
        df[col] = df[col].astype(str)
    return df


def _extract_entities(query: str, df: pd.DataFrame) -> dict[str, list[str]]:
    """Use spaCy NER when available, with fallback title/director/actor matching."""
    entities = {"titles": [], "persons": [], "genres": []}

    if importlib.util.find_spec("spacy") and importlib.util.find_spec("en_core_web_sm"):
        spacy = _optional_import("spacy")
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(query)
        for ent in doc.ents:
            if ent.label_ in {"WORK_OF_ART", "ORG"}:
                entities["titles"].append(ent.text)
            if ent.label_ == "PERSON":
                entities["persons"].append(ent.text)

    query_l = query.lower()
    for title in df["Title"].head(500).tolist():
        if title.lower() in query_l:
            entities["titles"].append(title)
    for person in pd.concat([df["Director"], df["Main_Actor"]]).unique().tolist():
        if person.lower() in query_l:
            entities["persons"].append(person)

    known_genres = set(df["Genre"].str.lower().unique()) | set(df["Second_Genre"].str.lower().unique())
    entities["genres"] = [g for g in known_genres if g in query_l]

    entities["titles"] = sorted(set(entities["titles"]))
    entities["persons"] = sorted(set(entities["persons"]))
    entities["genres"] = sorted(set(entities["genres"]))
    return entities


def answer_query(query: str, df: Optional[pd.DataFrame] = None, top_n: int = 5) -> dict[str, Any]:
    """Answer natural-language movie queries from local dataset.

    Examples:
      - "Who directed Parasite?"
      - "List Sci-Fi movies with high ratings"
    """
    df = load_movies() if df is None else df
    entities = _extract_entities(query, df)
    q = query.lower().strip()

    # Rule-guided parsing for deterministic retrieval.
    if q.startswith("who directed") or "directed" in q:
        title = entities["titles"][0] if entities["titles"] else q.replace("who directed", "").strip(" ?")
        matches = df[df["Title"].str.lower() == title.lower()]
        if len(matches) == 0:
            matches = df[df["Title"].str.contains(title, case=False, regex=False)]
        if len(matches) == 0:
            return {"query": query, "answer": f"No movie found for title '{title}'.", "rows": []}

        row = matches.iloc[0]
        return {
            "query": query,
            "answer": f"{row['Title']} was directed by {row['Director']}.",
            "rows": matches[["Title", "Director", "Genre", "Rating_Category"]].head(top_n).to_dict("records"),
        }

    if "list" in q or "show" in q:
        filtered = df.copy()
        if entities["genres"]:
            genre = entities["genres"][0]
            filtered = filtered[
                filtered["Genre"].str.lower().eq(genre) | filtered["Second_Genre"].str.lower().eq(genre)
            ]

        rating_match = re.search(r"(high|very high|medium|low)\s+rating", q)
        if rating_match:
            rating = rating_match.group(1).title()
            filtered = filtered[filtered["Rating_Category"].str.contains(rating, case=False, regex=False)]

        records = filtered[["Title", "Genre", "Second_Genre", "Rating_Category", "Director"]].head(top_n)
        answer = f"Found {len(filtered)} matches. Showing top {min(top_n, len(records))}."
        return {"query": query, "answer": answer, "rows": records.to_dict("records")}

    # Optional Hugging Face QA for free-form follow-up on top records.
    if importlib.util.find_spec("transformers"):
        transformers = _optional_import("transformers")
        qa = transformers.pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
        context = " ".join(
            f"{r.Title} is a {r.Genre} movie directed by {r.Director} starring {r.Main_Actor}."
            for r in df.head(100).itertuples()
        )
        out = qa(question=query, context=context)
        return {"query": query, "answer": out.get("answer", "No answer."), "rows": []}

    return {
        "query": query,
        "answer": "Could not map query deterministically. Install transformers for free-form QA fallback.",
        "rows": [],
    }


def capture_voice_query(timeout: int = 5, phrase_time_limit: int = 10) -> str:
    """Capture microphone audio with speech_recognition and return transcript."""
    sr = _optional_import("speech_recognition")

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Listening...")
        audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

    try:
        text = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        text = ""
    return text


def transcribe_with_whisper(audio_path: str, model_name: str = "base") -> str:
    whisper = _optional_import("whisper")
    model = whisper.load_model(model_name)
    out = model.transcribe(audio_path)
    return out.get("text", "").strip()


def listen_and_answer(df: Optional[pd.DataFrame] = None) -> dict[str, Any]:
    """End-to-end voice pipeline: capture audio -> text -> answer_query."""
    query = capture_voice_query()
    if not query:
        return {"query": "", "answer": "Sorry, I could not understand the audio.", "rows": []}
    return answer_query(query, df=df)


def train_content_recommender(df: Optional[pd.DataFrame] = None) -> RecommenderArtifacts:
    df = load_movies() if df is None else df.copy()
    df["content_text"] = (
        df["Genre"]
        + " "
        + df["Second_Genre"]
        + " "
        + df["Director"]
        + " "
        + df["Main_Actor"]
        + " "
        + df["Language"]
        + " "
        + df["Production_Country"]
    )

    sklearn_text = _optional_import("sklearn.feature_extraction.text")
    sklearn_pairwise = _optional_import("sklearn.metrics.pairwise")

    tfidf = sklearn_text.TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    tfidf_matrix = tfidf.fit_transform(df["content_text"])
    sim = sklearn_pairwise.cosine_similarity(tfidf_matrix, tfidf_matrix)

    return RecommenderArtifacts(df=df, tfidf_matrix=tfidf_matrix, similarity_matrix=sim)


def recommend_movies(movie_id: int | str, artifacts: RecommenderArtifacts, top_n: int = 5) -> pd.DataFrame:
    """Return top-N similar movies by content similarity.

    movie_id can be integer row index or exact title string.
    """
    df = artifacts.df
    if isinstance(movie_id, str):
        matches = df.index[df["Title"].str.lower() == movie_id.lower()].tolist()
        if not matches:
            raise ValueError(f"Movie title not found: {movie_id}")
        idx = matches[0]
    else:
        idx = int(movie_id)

    sim_scores = list(enumerate(artifacts.similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top = [i for i, _ in sim_scores[1 : top_n + 1]]

    out = df.loc[top, ["Title", "Genre", "Second_Genre", "Director", "Main_Actor"]].copy()
    out["similarity"] = [artifacts.similarity_matrix[idx, i] for i in top]
    return out.reset_index(drop=True)


def train_surprise_knn_example(ratings_csv_path: str):
    """Optional collaborative filtering example using Surprise KNNBasic."""
    surprise = _optional_import("surprise")
    Dataset = surprise.Dataset
    Reader = surprise.Reader
    KNNBasic = surprise.KNNBasic

    ratings = pd.read_csv(ratings_csv_path)
    reader = Reader(rating_scale=(ratings["rating"].min(), ratings["rating"].max()))
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
    trainset = data.build_full_trainset()

    algo = KNNBasic(sim_options={"name": "cosine", "user_based": False})
    algo.fit(trainset)
    return algo


def create_budget_vs_rating_scatter(df: Optional[pd.DataFrame] = None, output_html: Optional[str] = None):
    df = load_movies() if df is None else df.copy()
    budget_map = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}
    rating_map = {"Low": 1, "Medium": 2, "High": 3, "Very High": 4}

    df["BudgetScore"] = df["Budget_Category"].map(budget_map).fillna(0)
    df["RatingScore"] = df["Rating_Category"].map(rating_map).fillna(0)

    px = _optional_import("plotly.express")

    fig = px.scatter(
        df,
        x="BudgetScore",
        y="RatingScore",
        hover_name="Title",
        color="Genre",
        title="Budget vs Rating (interactive)",
        labels={"BudgetScore": "Budget Category (ordinal)", "RatingScore": "Rating Category (ordinal)"},
    )

    if output_html:
        fig.write_html(output_html)
    return fig


def create_collaboration_network(df: Optional[pd.DataFrame] = None, output_html: str = "movie_collab_graph.html") -> str:
    """Create actor-director collaboration graph and export interactive HTML via pyvis."""
    pyvis_network = _optional_import("pyvis.network")
    nx = _optional_import("networkx")

    df = load_movies() if df is None else df
    graph = nx.Graph()

    for row in df.itertuples(index=False):
        actor = str(row.Main_Actor)
        director = str(row.Director)
        movie = str(row.Title)
        if not actor or not director:
            continue
        graph.add_node(actor, group="actor")
        graph.add_node(director, group="director")
        if graph.has_edge(actor, director):
            graph[actor][director]["weight"] += 1
            graph[actor][director]["movies"].append(movie)
        else:
            graph.add_edge(actor, director, weight=1, movies=[movie])

    net = pyvis_network.Network(height="750px", width="100%", bgcolor="#111111", font_color="white")
    net.from_nx(graph)
    net.show(output_html)
    return output_html


def explain_recommendation_with_lime(query_title: str, recommended_title: str, df: Optional[pd.DataFrame] = None) -> dict[str, Any]:
    """Explain recommendation relevance using LIME on a surrogate regressor."""
    lime_tabular = _optional_import("lime.lime_tabular")

    df = load_movies() if df is None else df.copy()
    artifacts = train_content_recommender(df)

    query_idx = df.index[df["Title"].str.lower() == query_title.lower()].tolist()
    rec_idx = df.index[df["Title"].str.lower() == recommended_title.lower()].tolist()
    if not query_idx or not rec_idx:
        raise ValueError("Both query_title and recommended_title must exist in dataset")

    q = query_idx[0]

    feature_cols = ["Genre", "Second_Genre", "Director", "Main_Actor", "Language", "Production_Country", "Rating_Category"]
    X = df[feature_cols]
    y = artifacts.similarity_matrix[q]

    sklearn_compose = _optional_import("sklearn.compose")
    sklearn_preproc = _optional_import("sklearn.preprocessing")
    sklearn_pipeline = _optional_import("sklearn.pipeline")
    sklearn_ensemble = _optional_import("sklearn.ensemble")

    model = sklearn_pipeline.Pipeline(
        steps=[
            (
                "prep",
                sklearn_compose.ColumnTransformer(
                    transformers=[("cat", sklearn_preproc.OneHotEncoder(handle_unknown="ignore"), feature_cols)]
                ),
            ),
            ("rf", sklearn_ensemble.RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )
    model.fit(X, y)

    explainer = lime_tabular.LimeTabularExplainer(
        training_data=model.named_steps["prep"].transform(X).toarray(),
        feature_names=model.named_steps["prep"].get_feature_names_out().tolist(),
        mode="regression",
    )

    row_vec = model.named_steps["prep"].transform(X.iloc[[rec_idx[0]]]).toarray()[0]
    exp = explainer.explain_instance(row_vec, model.named_steps["rf"].predict, num_features=8)

    return {
        "query_title": query_title,
        "recommended_title": recommended_title,
        "top_contributors": exp.as_list(),
        "narrative": f"Recommendation is driven mostly by shared metadata patterns between '{query_title}' and '{recommended_title}'.",
    }


def explain_information_gain(parent_count: int, yes_count: int, no_count: int) -> dict[str, Any]:
    """Standalone information gain explanation for decision-tree questions.

    Gain = Entropy(parent) - weighted_entropy(children)
    where Entropy(group) = log2(group_size) for uniform class uncertainty.
    """
    if parent_count <= 0:
        raise ValueError("parent_count must be > 0")

    parent_entropy = np.log2(parent_count)
    p_yes = yes_count / parent_count
    p_no = no_count / parent_count
    weighted_children = 0.0
    if yes_count > 0:
        weighted_children += p_yes * np.log2(yes_count)
    if no_count > 0:
        weighted_children += p_no * np.log2(no_count)

    gain = float(parent_entropy - weighted_children)
    text = (
        f"Gain = Entropy(parent) - weighted_entropy(children) = {parent_entropy:.4f} "
        f"- {weighted_children:.4f} = {gain:.4f} bits"
    )
    return {
        "parent_entropy": float(parent_entropy),
        "weighted_children_entropy": float(weighted_children),
        "gain": gain,
        "explanation": text,
    }


if __name__ == "__main__":
    movies = load_movies()

    # --- 1) Text query examples ---
    print(answer_query("Who directed Parasite?", movies))
    print(answer_query("List drama movies with high ratings", movies, top_n=3))

    # --- 2) Recommendation example ---
    artifacts = train_content_recommender(movies)
    print(recommend_movies("Inception", artifacts, top_n=5).head())

    # --- 3) Visualization examples ---
    create_budget_vs_rating_scatter(movies, output_html="budget_vs_rating.html")
    create_collaboration_network(movies, output_html="collab_graph.html")
    print("Saved: budget_vs_rating.html, collab_graph.html")

    # --- 4) Explainability examples ---
    print(explain_information_gain(parent_count=3776, yes_count=1890, no_count=1886))