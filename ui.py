import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import joblib
import re

# Load model and vectorizer
model, vectorizer = joblib.load("model/image_relevance_model.pkl")

st.set_page_config(page_title="🧠 Image Relevance Classifier", layout="wide")
st.title("🧠 Image Relevance Prediction Dashboard")

# Same feature builder as training
def build_features(entry):
    src = entry.get("src", "").lower()
    alt = re.sub(r'[^a-zA-Z0-9 ]', '', entry["text"])
    text = entry.get("text", "").lower()

    parent_tag = "unknown"
    try:
        parent_tag = re.search(r"<(\w+)[^>]*>", text).group(1)
    except:
        pass

    return f"{alt} {src} {parent_tag}"


# -- Input URL and process --
url = st.text_input("🔗 Enter a news/article URL to predict relevant images:")

if url:
   # @st.cache_data(show_spinner=True)
   # @st.cache_data(show_spinner=True)
    @st.cache_data(show_spinner=True)
    def get_predictions(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        blocks = []

        for img in soup.find_all("img"):
            src = img.get("src") or ""
            parent = img.find_parent()
            text_block = f"{str(parent)} {str(img)}"

            entry = {"text": text_block, "src": src}
            feature = build_features(entry)

            blocks.append({
                "text": text_block,
                "src": src,
                "feature": feature
            })


        # Vectorize + Predict
        X = vectorizer.transform([b["feature"] for b in blocks])
        preds = model.predict(X)



        for i in range(len(blocks)):
            blocks[i]["label"] = int(preds[i])

        return pd.DataFrame(blocks)


    df = get_predictions(url)

    # 🔍 Show counts
    st.write("📊 Predicted Label Counts:", df["label"].value_counts().to_dict())

    # Sidebar filter
    label_filter = st.sidebar.radio("Filter Images:", ["All", "Relevant Only", "Irrelevant Only"])
    if label_filter == "Relevant Only":
        df = df[df["label"] == 1]
    elif label_filter == "Irrelevant Only":
        df = df[df["label"] == 0]

    # Show images
    cols = st.columns(3)
    for i, row in df.iterrows():
        with cols[i % 3]:
            
            st.image(row["src"], use_column_width=True, caption=f"Label: {'✅ Relevant' if row['label'] else '❌ Irrelevant'}")
            with st.expander("🔍 Raw HTML"):
                st.code(row["text"][:500] + "...", language="html")
