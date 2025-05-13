import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load data
with open("annotation.json") as f:
    data = json.load(f)

texts = [item["text"] for item in data if "label" in item]
labels = [item["label"] for item in data if "label" in item]

# Vectorize the HTML blocks
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
y = labels

# Train/Test split (for stats)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = AdaBoostClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Save model + vectorizer
joblib.dump((model, vectorizer), "image_relevance_model.pkl")
print("\n✅ Model saved as image_relevance_model.pkl")
