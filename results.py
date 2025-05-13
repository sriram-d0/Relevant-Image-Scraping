import requests
from bs4 import BeautifulSoup
import joblib
import pandas as pd

# Load model and vectorizer
model, vectorizer = joblib.load("image_relevance_model.pkl")

def get_img_blocks(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    blocks = []

    for img in soup.find_all('img'):
        parent = img.find_parent()
        text_block = f"{str(parent)} {str(img)}"
        img_url = img.get("src")
        blocks.append({"text": text_block, "src": img_url})

    return blocks

if __name__ == "__main__":
    url = input("🔗 Enter a news/article page URL: ").strip()
    img_blocks = get_img_blocks(url)

    texts = [block["text"] for block in img_blocks]
    X = vectorizer.transform(texts)
    preds = model.predict(X)

    # Add prediction labels to blocks
    for i in range(len(img_blocks)):
        img_blocks[i]["label"] = int(preds[i])

    # Save to CSV
    df = pd.DataFrame(img_blocks)
    df.to_csv("predicted_images.csv", index=False)

    print(f"\n✅ Exported {len(preds)} predictions to predicted_images.csv")
    print(f"🟢 Relevant Images: {sum(preds)}")
