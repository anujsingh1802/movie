# Information-Theoretic Movie Identification System

## New advanced assistant module
A fresh extension is available at `src/advanced_movie_assistant.py` with:

- Natural-language querying (`answer_query`) with optional Transformers QA fallback.
- Voice pipeline (`capture_voice_query`, `listen_and_answer`) via `speech_recognition`.
- Content-based recommendations (`train_content_recommender`, `recommend_movies`) + optional Surprise KNN example.
- Interactive visualizations:
  - Plotly scatter: budget vs rating category
  - PyVis collaboration graph (director-actor network)
- Explainability:
  - LIME-based recommendation explanation
  - Information-gain formula helper (`explain_information_gain`)

Run examples:

```bash
python3 src/advanced_movie_assistant.py
```

Optional dependencies for all features:

```bash
pip install transformers spacy speechrecognition openai-whisper surprise pyvis plotly lime shap
python -m spacy download en_core_web_sm
```