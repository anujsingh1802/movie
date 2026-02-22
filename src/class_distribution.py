import pandas as pd
import matplotlib.pyplot as plt
from utils import DATA_DIR
import os

df = pd.read_csv(os.path.join(DATA_DIR, "movie_cleaned.csv"))

genre_counts = df["Genre"].value_counts()

plt.figure(figsize=(10,6))
genre_counts.head(10).plot(kind="bar")
plt.title("Top 10 Genres")
plt.show()