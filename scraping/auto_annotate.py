import requests
from bs4 import BeautifulSoup
import json
import os

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

def is_duplicate(new_img, existing_imgs):
    return any(new_img['src'] == existing['src'] for existing in existing_imgs)

def auto_label(img):
    src = (img.get('src') or '').lower()
    text = (img.get('text') or '').lower()

    bad_patterns = ["placeholder", "facebook.com", "doubleclick", ".svg", ".ico", "pixel", "ads", "googleads"]
    if any(bad in src for bad in bad_patterns):
        return 0
    if "thumb" in src and "feature" not in src:
        return 0
    if "tracking" in text or '1x1' in text:
        return 0
    if "hero" in src or "main" in text or "live" in src:
        return 1
    if any(domain in src for domain in ["media.cnn.com", "ichef.bbci.co.uk", "ndtvimg.com", "toiimg.com"]):
        return 1

    return 1  # fallback assume relevant

if __name__ == "__main__":
    url = input("Enter a news/article page URL: ")
    new_imgs = get_img_blocks(url)

    if os.path.exists("annotation.json"):
        with open("annotation.json") as f:
            existing_imgs = json.load(f)
    else:
        existing_imgs = []

    added = 0
    for img in new_imgs:
        if not img['src'] or is_duplicate(img, existing_imgs):
            continue
        img["label"] = auto_label(img)
        existing_imgs.append(img)
        added += 1

    with open("annotation.json", "w") as f:
        json.dump(existing_imgs, f, indent=2)

    print(f"\n✅ Added {added} new auto-labeled images (Total: {len(existing_imgs)})")
