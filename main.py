from src.data_loader import load_data
from src.preprocessing import clean_text
from src.vectorization import vectorize
from src.model_training import train_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 1. Load LABELED dataset
data = load_data("data/raw/Quick_Commerce_Consumer_Behavior_Labeled.xlsx")

# 2. Use combined opinion text and sentiment label
data = data[["Combined_Opinion", "Sentiment"]].dropna()

# 3. Text preprocessing
data["clean_review"] = data["Combined_Opinion"].apply(clean_text)

# 4. Features and labels
X, vectorizer = vectorize(data["clean_review"])
y = data["Sentiment"]

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Train supervised ML model
model = train_model(X_train, y_train, vectorizer)

# 7. Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

print("\n✅ Model trained successfully using supervised learning")
