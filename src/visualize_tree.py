import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from utils import load_model
from preprocess import load_and_engineer

# Load model and encoders
model = load_model("movie_model.pkl")
X, _, encoders, _ = load_and_engineer()

feature_names = X.columns

plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    max_depth=3  # show only first 3 levels (tree is huge)
)

plt.title("Decision Tree (Top 3 Levels)")
plt.show()