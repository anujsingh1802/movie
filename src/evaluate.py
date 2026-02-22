from utils import load_model
from preprocess import load_and_engineer

model = load_model("movie_model.pkl")
encoders = load_model("feature_encoders.pkl")
target_encoder = load_model("target_encoder.pkl")

X, y_encoded, _, _ = load_and_engineer()

accuracy = model.score(X, y_encoded)

print("\nFull Dataset Accuracy:", round(accuracy * 100, 2), "%")