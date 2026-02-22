from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import RLock
from typing import Dict, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.engine import MovieEngine, load_dataset

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
    message: str
    session_id: str
    remaining: int


class AskResponse(BaseModel):
    session_id: str
    question: str
    feature: str
    value: str
    remaining: int
    yes_count: int
    no_count: int
    information_gain: float = Field(description="Expected entropy reduction in bits")
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


@dataclass
class GameSession:
    engine: MovieEngine
    lock: RLock = field(default_factory=RLock)
    last_access: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


sessions: Dict[str, GameSession] = {}
sessions_lock = RLock()
DATASET = load_dataset()


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
    session = GameSession(engine=MovieEngine(df=DATASET))

    with sessions_lock:
        sessions[session_id] = session

    return StartResponse(message="Game started", session_id=session_id, remaining=len(session.engine.remaining))


@app.post("/start", response_model=StartResponse)
def start_game_post() -> StartResponse:
    return _start_game()


@app.get("/start", response_model=StartResponse)
def start_game_get() -> StartResponse:
    # Backward-compatible for existing frontend calls.
    return _start_game()


@app.get("/ask", response_model=AskResponse | ResultResponse)
def ask_question(session_id: str):
    session = _get_session(session_id)

    with session.lock:
        result = session.engine.get_result()
        if result is not None:
            return ResultResponse(
                session_id=session_id,
                result=result,
                remaining=len(session.engine.remaining),
                questions_asked=session.engine.questions_asked,
            )

        question = session.engine.get_best_question()
        if question is None:
            raise HTTPException(status_code=409, detail="No informative question available")

        return AskResponse(
            session_id=session_id,
            question=question.question_text,
            feature=question.feature,
            value=question.value,
            remaining=len(session.engine.remaining),
            yes_count=question.yes_count,
            no_count=question.no_count,
            information_gain=round(question.information_gain, 4),
            questions_asked=session.engine.questions_asked,
        )


def _answer(payload: AnswerRequest) -> AnswerResponse:
    session = _get_session(payload.session_id)

    with session.lock:
        try:
            remaining = session.engine.apply_answer(payload.feature, payload.value, payload.answer)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if remaining <= 0:
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