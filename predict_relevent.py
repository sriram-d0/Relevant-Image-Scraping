import requests
from bs4 import BeautifulSoup
import joblib
import re

# Load model and vectorizer
model, vectorizer = joblib.load("image_relevance_model.pkl")

# Same as training!
def build_features(text, src):
    src = src.lower()
    alt = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    text_lower = text.lower()

    parent_tag = "unknown"
    try:
        parent_tag = re.search(r"<(\w+)[^>]*>", text_lower).group(1)
    except:
        pass

    return f"{alt} {src} {parent_tag}"

def get_img_blocks(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    blocks = []

    for img in soup.find_all('img'):
        parent = img.find_parent()
        text_block = f"{str(parent)} {str(img)}"
        img_url = img.get("src") or ""
        blocks.append({"text": text_block, "src": img_url})

    return blocks

if __name__ == "__main__":
    url = input("🔗 Enter a news/article page URL: ").strip()
    img_blocks = get_img_blocks(url)

    # 💥 Use the same build_features() as training
    features = [build_features(block["text"], block["src"]) for block in img_blocks]
    X = vectorizer.transform(features)
    preds = model.predict(X)

    print(f"\n🧠 Total Images Found: {len(preds)}")
    print(f"✅ Relevant Images: {sum(preds)}")
    print(f"❌ Irrelevant Images: {len(preds) - sum(preds)}")

