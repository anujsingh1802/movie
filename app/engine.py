import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "movie_cleaned.csv")

class MovieEngine:

    def __init__(self):
        self.df = pd.read_csv(DATA_PATH)
        self.reset()

    def reset(self):
        self.remaining = self.df.copy()
        self.questions_asked = 0

    def get_best_question(self):

        best_feature = None
        best_balance = 1.0

        for col in self.remaining.columns:
            if col == "Title":
                continue

            counts = self.remaining[col].value_counts()

            if len(counts) <= 1:
                continue

            ratio = counts.max() / len(self.remaining)

            if ratio < best_balance:
                best_balance = ratio
                best_feature = col

        if best_feature is None:
            return None

        most_common = self.remaining[best_feature].value_counts().idxmax()

        return best_feature, most_common

    def apply_answer(self, feature, value, answer):

        if answer:
            self.remaining = self.remaining[self.remaining[feature] == value]
        else:
            self.remaining = self.remaining[self.remaining[feature] != value]

        self.questions_asked += 1

    def get_result(self):

        if len(self.remaining) == 1:
            return self.remaining.iloc[0]["Title"]

        return None