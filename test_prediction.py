import joblib
from src.preprocessing import clean_text

model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Explicit keyword guards (survey-aware)
negative_words = [
    "concern", "delay", "delayed", "late",
    "poor", "bad", "issue", "problem",
    "expensive", "worst", "risk"
]

neutral_words = [
    "more", "better", "required", "need",
    "expected", "expect", "variety",
    "should", "could", "would"
]

print("🔹 Q-Commerce Sentiment Prediction")
print("Type 'exit' to stop")

while True:
    text = input("\nEnter customer opinion: ")
    if text.lower() == "exit":
        break

    clean = clean_text(text)

    # 1️⃣ Negative guard (HIGHEST priority)
    if any(w in clean for w in negative_words):
        print("Predicted Sentiment: Negative")
        continue

    # 2️⃣ Neutral guard
    if any(w in clean for w in neutral_words):
        print("Predicted Sentiment: Neutral")
        continue

    # 3️⃣ ML prediction (mainly Positive)
    vec = vectorizer.transform([clean])
    prediction = model.predict(vec)[0]
    print("Predicted Sentiment:", prediction)
