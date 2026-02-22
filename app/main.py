from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock
import math
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.engine import MovieEngine, load_dataset
from src.src.advanced_movie_assistant import answer_query, recommend_movies, train_content_recommender

app = FastAPI(title="Movie Guessing AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


SESSION_TTL_MINUTES = 30


class StartResponse(BaseModel):
@@ -56,60 +58,129 @@ class AskResponse(BaseModel):
    information_gain: float = Field(description="Expected entropy reduction in bits")
    explanation: str
    questions_asked: int


class ResultResponse(BaseModel):
    session_id: str
    result: str
    remaining: int
    questions_asked: int


class AnswerRequest(BaseModel):
    session_id: str
    feature: str
    value: str
    answer: bool


class AnswerResponse(BaseModel):
    session_id: str
    remaining: int
    questions_asked: int


class NLPQueryRequest(BaseModel):
    query: str = Field(min_length=1)


class NLPQueryResponse(BaseModel):
    answer: str
    rows: list[dict[str, Any]]


class RecommendationItem(BaseModel):
    title: str
    explanation: str


class RecommendationResponse(BaseModel):
    recommendations: list[RecommendationItem]


class StatsResponse(BaseModel):
    total_movies: int
    theoretical_min: int
    actual_average: float
    feature_importance: dict[str, float]


@dataclass
class GameSession:
    engine: MovieEngine
    lock: RLock = field(default_factory=RLock)
    last_access: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


sessions: Dict[str, GameSession] = {}
sessions_lock = RLock()
DATASET = load_dataset()
RECOMMENDER_ARTIFACTS = train_content_recommender(df=DATASET)


def _compute_feature_importance(df) -> dict[str, float]:
    parent_count = len(df)
    feature_scores: dict[str, float] = {}
    for feature in (col for col in df.columns if col != "Title"):
        counts = df[feature].value_counts(dropna=False)
        best_gain = 0.0
        for count in counts:
            yes_count = int(count)
            no_count = parent_count - yes_count
            if yes_count == 0 or no_count == 0:
                continue
            gain = MovieEngine.information_gain(parent_count, yes_count, no_count)
            if gain > best_gain:
                best_gain = gain
        feature_scores[feature] = round(best_gain, 4)
    return dict(sorted(feature_scores.items(), key=lambda item: item[1], reverse=True))


def _simulate_average_questions(df) -> float:
    sample_size = min(len(df), 250)
    sampled = df.sample(n=sample_size, random_state=42) if sample_size < len(df) else df

    steps: list[int] = []
    for _, target in sampled.iterrows():
        engine = MovieEngine(df=df)
        while len(engine.remaining) > 1:
            question = engine.get_best_question()
            if question is None:
                break
            answer = str(target[question.feature]) == question.value
            engine.apply_answer(question.feature, question.value, answer)
        steps.append(engine.questions_asked)
    return round(sum(steps) / len(steps), 2)


STATS = StatsResponse(
    total_movies=len(DATASET),
    theoretical_min=math.ceil(math.log2(len(DATASET))),
    actual_average=_simulate_average_questions(DATASET),
    feature_importance=_compute_feature_importance(DATASET),
)


def _prune_sessions() -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=SESSION_TTL_MINUTES)
    with sessions_lock:
        stale_ids = [sid for sid, session in sessions.items() if session.last_access < cutoff]
        for sid in stale_ids:
            sessions.pop(sid, None)


def _get_session(session_id: str) -> GameSession:
    _prune_sessions()
    with sessions_lock:
        session = sessions.get(session_id)

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    session.last_access = datetime.now(timezone.utc)
    return session


def _start_game() -> StartResponse:
    _prune_sessions()
    session_id = str(uuid4())
@@ -178,26 +249,63 @@ def _answer(payload: AnswerRequest) -> AnswerResponse:
            raise HTTPException(
                status_code=409,
                detail="No movies remain after this answer. Answers are inconsistent.",
            )

        return AnswerResponse(
            session_id=payload.session_id,
            remaining=remaining,
            questions_asked=session.engine.questions_asked,
        )


@app.post("/answer", response_model=AnswerResponse)
def answer_question(payload: AnswerRequest) -> AnswerResponse:
    return _answer(payload)


@app.post("/answer/legacy", response_model=AnswerResponse)
def answer_question_legacy(
    session_id: str = Query(...),
    feature: str = Query(...),
    value: str = Query(...),
    answer: bool = Query(...),
) -> AnswerResponse:
    # Backward-compatible endpoint for query-string POST callers.
    return _answer(AnswerRequest(session_id=session_id, feature=feature, value=value, answer=answer))


@app.post("/nlp-query", response_model=NLPQueryResponse)
def nlp_query(payload: NLPQueryRequest) -> NLPQueryResponse:
    try:
        result = answer_query(payload.query, df=DATASET)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process NLP query: {exc}") from exc

    return NLPQueryResponse(answer=str(result.get("answer", "")), rows=result.get("rows", []))


@app.get("/recommend", response_model=RecommendationResponse)
def recommend(movie: str = Query(..., min_length=1)) -> RecommendationResponse:
    try:
        recommendations = recommend_movies(movie_id=movie, artifacts=RECOMMENDER_ARTIFACTS)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {exc}") from exc

    items = [
        RecommendationItem(
            title=str(row["Title"]),
            explanation=(
                f"Similar profile ({row['Genre']}/{row['Second_Genre']}) with director {row['Director']} "
                f"and actor {row['Main_Actor']}."
            ),
        )
        for row in recommendations.to_dict("records")
    ]
    return RecommendationResponse(recommendations=items)


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    return STATS