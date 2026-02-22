import pandas as pd
import os
from utils import DATA_DIR

df = pd.read_csv(os.path.join(DATA_DIR, "movie_cleaned.csv"))

question_order = [
    "Genre",
    "Release_Year",
    "Main_Actor",
    "Director",
    "Second_Genre",
    "Budget_Category",
    "Revenue_Category",
    "Runtime_Category",
    "Rating_Category",
    "Vote_Count_Category",
    "Popularity_Category",
    "Production_Country",
    "Language",
]

total_questions = []

for _, target in df.iterrows():

    remaining = df.copy()
    questions = 0

    for feature in question_order:

        if len(remaining) <= 1:
            break

        value = target[feature]
        remaining = remaining[remaining[feature] == value]
        questions += 1

    total_questions.append(questions)

import numpy as np

print("\nManual System Statistics:")
print("Average Questions:", round(np.mean(total_questions), 2))
print("Minimum Questions:", np.min(total_questions))
print("Maximum Questions:", np.max(total_questions))