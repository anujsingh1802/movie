# src/feature_importance.py

import matplotlib.pyplot as plt
import pandas as pd
from utils import load_model
from preprocess import load_and_engineer

# Load model
model = load_model("movie_model.pkl")
X, _, _, _ = load_and_engineer()

feature_names = X.columns
importances = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(importance_df)

plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.show()