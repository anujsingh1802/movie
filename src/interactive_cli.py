import numpy as np
from utils import load_model
from preprocess import load_and_engineer

# Load model + encoders
model = load_model("movie_model.pkl")
encoders = load_model("feature_encoders.pkl")
target_encoder = load_model("target_encoder.pkl")

X, _, _, _ = load_and_engineer()

feature_names = X.columns
tree = model.tree_


def ask_yes_no(question):
    while True:
        ans = input(question + " (yes/no): ").strip().lower()
        if ans in ["yes", "y"]:
            return True
        elif ans in ["no", "n"]:
            return False
        else:
            print("Please answer yes or no.")


def traverse(node_id=0):

    # If leaf node
    if tree.feature[node_id] == -2:
        value = tree.value[node_id]
        class_id = np.argmax(value)
        movie = target_encoder.inverse_transform([class_id])[0]
        print("\n🎉 Your movie is:", movie)
        return

    feature_index = tree.feature[node_id]
    threshold = tree.threshold[node_id]

    feature_name = feature_names[feature_index]
    encoder = encoders[feature_name]

    # For categorical encoded values, threshold splits numeric encoding
    question = f"Is {feature_name} <= {threshold:.1f} ?"

    answer = ask_yes_no(question)

    if answer:
        next_node = tree.children_left[node_id]
    else:
        next_node = tree.children_right[node_id]

    traverse(next_node)


print("\n🎬 Think of a movie from the dataset.")
print("Answer with yes or no.\n")

traverse()