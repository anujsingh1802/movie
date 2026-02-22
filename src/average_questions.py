# src/average_questions.py

import numpy as np
from utils import load_model
from preprocess import load_and_engineer

# Load model and data
model = load_model("movie_model.pkl")
X, y_encoded, _, _ = load_and_engineer()

X = np.asarray(X)

# Get decision path matrix
node_indicator = model.decision_path(X)

# Count number of nodes visited per sample
depths = []

for i in range(X.shape[0]):
    node_count = node_indicator.indptr[i+1] - node_indicator.indptr[i]
    depths.append(node_count - 1)  # subtract 1 (root node not a question)

depths = np.array(depths)

print("\nAverage Questions Required:", round(np.mean(depths), 2))
print("Minimum Questions Required:", np.min(depths))
print("Maximum Questions Required:", np.max(depths))
print("Tree Maximum Depth:", model.get_depth())