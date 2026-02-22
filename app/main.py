from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.engine import MovieEngine
import os

# ----------------------------
# App Initialization
# ----------------------------

app = FastAPI(title="Movie Guessing AI")

# ----------------------------
# CORS (optional but safe)
# ----------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Paths
# ----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

# ----------------------------
# Mount Static Files
# ----------------------------

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# ----------------------------
# Initialize Engine
# ----------------------------

engine = MovieEngine()

# ----------------------------
# API Routes
# ----------------------------

@app.get("/start")
def start_game():
    engine.reset()
    return {"message": "Game started"}


@app.get("/ask")
def ask_question():

    # Safety reset
    if engine.remaining is None or len(engine.remaining) == 0:
        engine.reset()

    result = engine.get_result()

    if result:
        return {"result": result}

    q = engine.get_best_question()

    if q is None:
        return {"error": "No question available"}

    feature, value = q

    return {
        "feature": str(feature),
        "value": str(value),
        "remaining": int(len(engine.remaining))
    }


@app.post("/answer")
def answer_question(feature: str, value: str, answer: bool):

    engine.apply_answer(feature, value, answer)

    return {
        "remaining": int(len(engine.remaining)),
        "questions_asked": engine.questions_asked
    }