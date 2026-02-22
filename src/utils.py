import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


def save_model(obj, filename):
    joblib.dump(obj, os.path.join(MODEL_DIR, filename))


def load_model(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))