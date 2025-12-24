from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_model(X, y, vectorizer):
    os.makedirs("models", exist_ok=True)

    # IMPORTANT: class_weight balances Positive / Neutral / Negative
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    model.fit(X, y)

    joblib.dump(model, "models/sentiment_model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    return model
