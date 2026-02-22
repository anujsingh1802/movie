from utils import load_model, DATA_DIR
import pandas as pd
import os

model = load_model("movie_model.pkl")
encoders = load_model("feature_encoders.pkl")
target_encoder = load_model("target_encoder.pkl")

df = pd.read_csv(os.path.join(DATA_DIR, "movie_cleaned.csv"))

print("\n🎬 Think of a movie.\n")

# Recreate engineered feature choices
df["Decade"] = (df["Release_Year"] // 10 * 10).astype(str) + "s"
df["Language_Group"] = df["Language"].apply(
    lambda x: "English" if x == "en" else "Non-English"
)
df["Is_Blockbuster"] = df["Budget_Category"].apply(
    lambda x: "Yes" if x == "High" else "No"
)

features = [
    "Genre",
    "Decade",
    "Budget_Category",
    "Rating_Category",
    "Language_Group",
    "Actor_Fame",
    "Director_Fame",
    "Is_Blockbuster",
]

# Since Actor_Fame & Director_Fame were engineered,
# reload full engineered dataset via preprocess
from preprocess import load_and_engineer
X, _, encoders, target_encoder = load_and_engineer()

original_df = pd.read_csv(os.path.join(DATA_DIR, "movie_cleaned.csv"))
engineered_df = pd.DataFrame(X)

user_input = {}

for feature in features:
    values = sorted(encoders[feature].classes_)

    print(f"\nSelect {feature}:")
    for i, v in enumerate(values):
        print(f"{i}: {v}")

    while True:
        try:
            choice = int(input("Enter number: "))
            if 0 <= choice < len(values):
                user_input[feature] = values[choice]
                break
            else:
                print("Invalid number.")
        except:
            print("Enter a valid integer.")

input_df = pd.DataFrame([user_input])

for col in input_df.columns:
    input_df[col] = encoders[col].transform(input_df[col])

prediction = model.predict(input_df)
movie = target_encoder.inverse_transform(prediction)

print("\n🎉 Your movie is:", movie[0])