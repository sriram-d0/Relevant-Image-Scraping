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

if __name__ == "__main__":
    url = input("Enter a news/article page URL: ")
    new_imgs = get_img_blocks(url)

    # Load existing data if available
    if os.path.exists("images_for_annotation.json"):
        with open("images_for_annotation.json") as f:
            existing_imgs = json.load(f)
    else:
        existing_imgs = []

    # Append only new unique images
    for img in new_imgs:
        if not img['src']:
            continue
        if is_duplicate(img, existing_imgs):
            continue  # default to relevant

        existing_imgs.append(img)

    # Save updated list
    with open("images_for_annotation.json", "w") as f:
        json.dump(existing_imgs, f, indent=2)

    print(f"✅ Added {len(new_imgs)} new images (Total: {len(existing_imgs)})")
