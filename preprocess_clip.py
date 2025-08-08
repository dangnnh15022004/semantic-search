import os
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from pinecone import Pinecone
from transformers import CLIPProcessor, CLIPModel

# === Pinecone setup ===
PINECONE_API_KEY = "pcsk_4Em9zF_NcfUg4U1n3tGfMVGqnuLv4KudrMs1CeKBnsSRSkQPokMuhsujXuCy8M6aRExKhJ"
PINECONE_ENV = "us-east-1"
INDEX_NAME = "semantic-clip"

pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(INDEX_NAME)

# === Load CSV and filter ===
df = pd.read_csv("styles.csv", on_bad_lines="skip")
columns_needed = ['id', 'gender', 'articleType', 'baseColour', 'season', 'usage', 'productDisplayName']
df = df[columns_needed].dropna().astype(str)

# Generate text description for each row
df["text"] = df.apply(
    lambda row: f"{row['productDisplayName']}. {row['gender']} {row['articleType']} in {row['baseColour']}, {row['usage']} - {row['season']} collection",
    axis=1
)

# === Load CLIP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# === Image loader ===
def load_image(image_path):
    try:
        return Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"⚠️ Error loading image at {image_path}: {e}")
        return None

# === Embedding and upload ===
batch_size = 100
image_folder = "images"
vectors = []

for i in tqdm(range(0, len(df), batch_size), desc="Uploading"):
    batch_df = df.iloc[i:i + batch_size]

    for _, row in batch_df.iterrows():
        product_id = row['id']
        image_path = os.path.join(image_folder, f"{product_id}.jpg")
        image = load_image(image_path)
        if image is None:
            continue  # Skip if image loading fails

        # Process inputs separately
        inputs = clip_processor(text=[row["text"]], images=image, return_tensors="pt", padding=True)
        text_inputs = {
            "input_ids": inputs["input_ids"].to(device),
            "attention_mask": inputs["attention_mask"].to(device)
        }
        image_inputs = {
            "pixel_values": inputs["pixel_values"].to(device)
        }

        with torch.no_grad():
            text_emb = clip_model.get_text_features(**text_inputs)
            image_emb = clip_model.get_image_features(**image_inputs)

        # Normalize
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)

        # Combine equally (default)
        combined_emb = (text_emb + image_emb) / 2
        combined_emb = combined_emb.squeeze(0).cpu().numpy()

        # Metadata for search filtering
        metadata = {
            "id": product_id,
            "text": row["text"],
            "productDisplayName": row["productDisplayName"],
            "gender": row["gender"],
            "category": row["articleType"],
            "colour": row["baseColour"],
            "season": row["season"],
            "usage": row["usage"]
        }

        vectors.append((product_id, combined_emb.tolist(), metadata))

    # Ensure we're not uploading empty batches
    if vectors:
        try:
            index.upsert(vectors=vectors)
            print(f"Successfully uploaded {len(vectors)} vectors.")
        except Exception as e:
            print(f"Error uploading vectors: {e}")
        vectors = []  # Reset after upload
    else:
        print(f"Skipping empty batch at index {i}")

# Final message
print("✅ Done uploading CLIP combined embeddings to Pinecone.")
