from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "movie_cleaned.csv")


@dataclass(frozen=True)
class Question:
    feature: str
    value: str
    question_text: str
    yes_count: int
    no_count: int
    information_gain: float


@lru_cache(maxsize=1)
def load_dataset(data_path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if "Title" not in df.columns:
        raise ValueError("Dataset must contain a 'Title' column")

    # Normalize all feature columns to deterministic string values once.
    for col in df.columns:
        if col == "Title":
            continue
        df[col] = df[col].fillna("Unknown").astype(str)

    return df


class MovieEngine:
    """Session-scoped entropy-based movie guessing engine."""

    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df = df if df is not None else load_dataset()
        self.feature_columns = [col for col in self.df.columns if col != "Title"]
        self.reset()

    def reset(self) -> None:
        self.remaining = self.df
        self.questions_asked = 0

    @staticmethod
    def _expected_posterior_entropy(yes_count: int, no_count: int) -> float:
        total = yes_count + no_count
        if total <= 1:
            return 0.0

        entropy = 0.0
        if yes_count:
            p = yes_count / total
            entropy += p * math.log2(yes_count)
        if no_count:
            p = no_count / total
            entropy += p * math.log2(no_count)
        return entropy

    @staticmethod
    def _question_text(feature: str, value: str) -> str:
        readable_feature = feature.replace("_", " ").lower()
        return f"Is your movie {readable_feature} '{value}'?"

    def get_best_question(self) -> Optional[Question]:
        total = len(self.remaining)
        if total <= 1:
            return None

        parent_entropy = math.log2(total)
        best_question: Optional[Question] = None
        best_ig = float("-inf")

        for feature in self.feature_columns:
            counts = self.remaining[feature].value_counts(dropna=False)
            if len(counts) <= 1:
                continue

            for raw_value, yes_count_raw in counts.items():
                yes_count = int(yes_count_raw)
                no_count = total - yes_count
                if yes_count == 0 or no_count == 0:
                    continue

                posterior_entropy = self._expected_posterior_entropy(yes_count, no_count)
                information_gain = parent_entropy - posterior_entropy
                value = str(raw_value)

                # Better split quality first; deterministic fallback on feature/value.
                if (
                    information_gain > best_ig
                    or (
                        math.isclose(information_gain, best_ig)
                        and best_question is not None
                        and (feature, value) < (best_question.feature, best_question.value)
                    )
                ):
                    best_ig = information_gain
                    best_question = Question(
                        feature=feature,
                        value=value,
                        question_text=self._question_text(feature, value),
                        yes_count=yes_count,
                        no_count=no_count,
                        information_gain=information_gain,
                    )

        return best_question

    def apply_answer(self, feature: str, value: str, answer: bool) -> int:
        if feature not in self.feature_columns:
            raise ValueError(f"Unknown feature: {feature}")

        mask = self.remaining[feature] == value
        self.remaining = self.remaining[mask] if answer else self.remaining[~mask]
        self.questions_asked += 1
        return len(self.remaining)

    def get_result(self) -> Optional[str]:
        if len(self.remaining) == 1:
            return str(self.remaining.iloc[0]["Title"])
        return None