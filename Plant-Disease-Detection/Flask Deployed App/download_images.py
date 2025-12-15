import os
import urllib.request

folder = "static/images/supplements"
os.makedirs(folder, exist_ok=True)

# Use a simple placeholder image generator
images = [
    "fungicide_cotton.jpg", "systemic_fungicide.jpg", "npk_fertilizer.jpg",
    "organic_compost.jpg", "copper_fungicide.jpg", "bactericide.jpg",
    "insecticide.jpg", "mango_fertilizer.jpg", "sulfur_fungicide.jpg",
    "neem_oil.jpg", "copper_spray.jpg", "vegetable_fertilizer.jpg",
    "mancozeb.jpg", "late_blight_fungicide.jpg", "potato_fertilizer.jpg",
    "copper_bactericide.jpg", "chlorothalonil.jpg", "tomato_fungicide.jpg",
    "leaf_mold_fungicide.jpg", "miticide.jpg", "target_spot_fungicide.jpg",
    "whitefly_insecticide.jpg", "tomato_seeds.jpg", "tomato_fertilizer.jpg",
    "rice_fungicide.jpg", "rice_fertilizer.jpg", "rice_insecticide.jpg",
    "tricyclazole.jpg", "wheat_fertilizer.jpg", "wheat_fungicide.jpg",
    "rust_fungicide.jpg"
]

# Download a placeholder image once and copy it
placeholder_url = "https://via.placeholder.com/300x300/28a745/ffffff?text=Supplement"

for img in images:
    path = os.path.join(folder, img)
    try:
        # Create text based on filename
        name = img.replace(".jpg", "").replace("_", "+")
        url = f"https://via.placeholder.com/300x300/28a745/ffffff?text={name}"
        urllib.request.urlretrieve(url, path)
        print(f"✅ {img}")
    except Exception as e:
        print(f"❌ {img}: {e}")

print("\nDone!")
