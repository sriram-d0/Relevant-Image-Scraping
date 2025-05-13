import json
import webbrowser
import time

# Load image data
with open("images_for_annotation.json") as f:
    img_data = json.load(f)

print(f"\n🖼️ Found {len(img_data)} images. Let's annotate...\n")

for i, block in enumerate(img_data):
    img_url = block.get('src', '')
    print(f"\n[{i+1}/{len(img_data)}] Image URL: {img_url}")

    # Auto-label known irrelevant placeholders
    if "placeholder" in img_url.lower():
        print("🚫 Skipped (auto-labeled as irrelevant due to 'placeholder' in URL)")
        block["label"] = 0
    else:
        block['label'] = 1

# Save results
with open("annotated_images.json", "w") as f:
    json.dump(img_data, f, indent=2)

print("\n✅ Done! Annotated file saved as annotated_images.json")
