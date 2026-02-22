# src/preprocess.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils import DATA_DIR
import os


def load_and_engineer():

    df = pd.read_csv(os.path.join(DATA_DIR, "movie_cleaned.csv"))

    # --------------------------
    # Use ALL discriminative features
    # --------------------------

    features = [
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

    X = df[features].copy()
    y = df["Title"]

    encoders = {}

    for col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)

    return X, y_encoded, encoders, target_encoder