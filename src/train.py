# src/train.py

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from preprocess import load_and_engineer
from utils import save_model

# --------------------------
# Load Engineered Dataset
# --------------------------

X, y_encoded, encoders, target_encoder = load_and_engineer()

X = np.asarray(X)
y_encoded = np.asarray(y_encoded)

total_movies = len(y_encoded)

print("\nDataset Loaded Successfully.")
print("Total Movies:", total_movies)
print("Total Features:", X.shape[1])

# --------------------------
# Train on FULL Dataset
# --------------------------

model = DecisionTreeClassifier(
    criterion="entropy",     # ID3-style splitting
    splitter="best",         # best split strategy
    max_depth=None,          # allow full memorization
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

model.fit(X, y_encoded)

# --------------------------
# Validate Memorization
# --------------------------

predictions = model.predict(X)
correct_predictions = np.sum(predictions == y_encoded)

accuracy = correct_predictions / total_movies

if accuracy == 1.0:
    print("\n✅ Model perfectly memorized dataset.")
else:
    print("\n⚠ WARNING: Model did NOT memorize perfectly.")

# --------------------------
# Save Model + Encoders
# --------------------------

save_model(model, "movie_model.pkl")
save_model(encoders, "feature_encoders.pkl")
save_model(target_encoder, "target_encoder.pkl")

# --------------------------
# Training Info
# --------------------------

print("\nModel trained successfully on FULL dataset.")
print("Training Accuracy:", round(accuracy * 100, 2), "%")
print("Tree Depth:", model.get_depth())
print("Number of Leaves:", model.get_n_leaves())
print("Total Nodes:", model.tree_.node_count)