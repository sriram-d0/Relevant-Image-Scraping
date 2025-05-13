import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re

# Load dataset
with open("annotation.json") as f:
    data = json.load(f)

# Helper to clean and extract better features
def build_features(entry):
    src = entry.get("src", "").lower()
    alt = re.sub(r'[^a-zA-Z0-9 ]', '', entry["text"])  # basic cleanup of HTML
    text = entry.get("text", "").lower()

    parent_tag = "unknown"
    try:
        parent_tag = re.search(r"<(\w+)[^>]*>", text).group(1)
    except:
        pass

    features = f"{alt} {src} {parent_tag}"
    return features

# Clean + extract
X_raw = []
y = []

for entry in data:
    if "label" not in entry:
        continue

    src = entry.get("src", "")

    X_raw.append(build_features(entry))
    y.append(entry["label"])

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X_raw)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = AdaBoostClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Save
joblib.dump((model, vectorizer), "image_relevance_model.pkl")
print("✅ Upgraded model saved as image_relevance_model.pkl")
