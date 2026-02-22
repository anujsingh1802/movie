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
    weighted_child_entropy: float
    information_gain: float
    explanation: str


@lru_cache(maxsize=1)
def load_dataset(data_path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if "Title" not in df.columns:
        raise ValueError("Dataset must contain a 'Title' column")

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
    def _safe_entropy(group_size: int) -> float:
        return math.log2(group_size) if group_size > 0 else 0.0

    @classmethod
    def _weighted_child_entropy(cls, parent_count: int, yes_count: int, no_count: int) -> float:
        if parent_count <= 1:
            return 0.0
        p_yes = yes_count / parent_count
        p_no = no_count / parent_count
        return p_yes * cls._safe_entropy(yes_count) + p_no * cls._safe_entropy(no_count)

    @classmethod
    def information_gain(cls, parent_count: int, yes_count: int, no_count: int) -> float:
        parent_entropy = cls._safe_entropy(parent_count)
        weighted_children = cls._weighted_child_entropy(parent_count, yes_count, no_count)
        return parent_entropy - weighted_children

    @classmethod
    def information_gain_explanation(cls, feature: str, value: str, parent_count: int, yes_count: int, no_count: int) -> str:
        parent_entropy = cls._safe_entropy(parent_count)
        weighted_children = cls._weighted_child_entropy(parent_count, yes_count, no_count)
        gain = parent_entropy - weighted_children
        return (
            f"Question '{feature} == {value}' selected because Gain = Entropy(parent) - "
            f"weighted_entropy(children) = {parent_entropy:.4f} - {weighted_children:.4f} = {gain:.4f} bits."
        )

    @staticmethod
    def _question_text(feature: str, value: str) -> str:
        readable_feature = feature.replace("_", " ").lower()
        return f"Is your movie {readable_feature} '{value}'?"

    def get_best_question(self) -> Optional[Question]:
        parent_count = len(self.remaining)
        if parent_count <= 1:
            return None

        best_question: Optional[Question] = None
        best_ig = float("-inf")

        for feature in self.feature_columns:
            counts = self.remaining[feature].value_counts(dropna=False)
            if len(counts) <= 1:
                continue

            for raw_value, yes_count_raw in counts.items():
                yes_count = int(yes_count_raw)
                no_count = parent_count - yes_count
                if yes_count == 0 or no_count == 0:
                    continue

                gain = self.information_gain(parent_count, yes_count, no_count)
                value = str(raw_value)

                if (
                    gain > best_ig
                    or (
                        math.isclose(gain, best_ig)
                        and best_question is not None
                        and (feature, value) < (best_question.feature, best_question.value)
                    )
                ):
                    weighted_children = self._weighted_child_entropy(parent_count, yes_count, no_count)
                    best_ig = gain
                    best_question = Question(
                        feature=feature,
                        value=value,
                        question_text=self._question_text(feature, value),
                        yes_count=yes_count,
                        no_count=no_count,
                        weighted_child_entropy=weighted_children,
                        information_gain=gain,
                        explanation=self.information_gain_explanation(feature, value, parent_count, yes_count, no_count),
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
