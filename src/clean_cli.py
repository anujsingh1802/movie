import pandas as pd
import numpy as np
import os
from utils import DATA_DIR

df = pd.read_csv(os.path.join(DATA_DIR, "movie_cleaned.csv"))

remaining = df.copy()

print("\n🎬 Think of a movie from the dataset.")
print("Answer only yes or no.\n")

def ask(question):
    while True:
        ans = input(question + " (yes/no): ").strip().lower()
        if ans in ["yes", "y"]:
            return True
        elif ans in ["no", "n"]:
            return False
        else:
            print("Please answer yes or no.")

while len(remaining) > 1:

    # Choose best feature automatically (entropy-like)
    best_feature = None
    best_balance = 1.0

    for col in remaining.columns:
        if col == "Title":
            continue

        counts = remaining[col].value_counts()

        if len(counts) <= 1:
            continue

        largest_ratio = counts.max() / len(remaining)

        if largest_ratio < best_balance:
            best_balance = largest_ratio
            best_feature = col

    if best_feature is None:
        break

    # Choose most common value for binary question
    most_common_value = remaining[best_feature].value_counts().idxmax()

    question = f"Is {best_feature} equal to '{most_common_value}'?"

    if ask(question):
        remaining = remaining[remaining[best_feature] == most_common_value]
    else:
        remaining = remaining[remaining[best_feature] != most_common_value]

    print("Remaining movies:", len(remaining))
    print()

if len(remaining) == 1:
    print("🎉 Your movie is:", remaining.iloc[0]["Title"])
else:
    print("Could not uniquely identify.")