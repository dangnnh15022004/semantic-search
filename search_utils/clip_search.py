import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from pinecone import Pinecone
import numpy as np
import os

# === Load CLIP model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# === Pinecone setup ===
PINECONE_API_KEY = "pcsk_4Em9zF_NcfUg4U1n3tGfMVGqnuLv4KudrMs1CeKBnsSRSkQPokMuhsujXuCy8M6aRExKhJ"
PINECONE_ENV = "us-east-1"
INDEX_NAME = "semantic-clip"

pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(INDEX_NAME)

# === Embedding Function ===
def generate_combined_embedding(text=None, image: Image.Image = None):
    """Encode text and/or image, return normalized embedding"""
    if text and image:
        inputs = clip_processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_emb = clip_model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            image_emb = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        return ((text_emb + image_emb) / 2).squeeze(0).cpu().numpy()

    elif text:
        inputs = clip_processor(text=[text], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_emb = clip_model.get_text_features(**inputs)
        return (text_emb / text_emb.norm(dim=-1, keepdim=True)).squeeze(0).cpu().numpy()

    elif image:
        inputs = clip_processor(images=[image], return_tensors="pt").to(device)
        with torch.no_grad():
            image_emb = clip_model.get_image_features(**inputs)
        return (image_emb / image_emb.norm(dim=-1, keepdim=True)).squeeze(0).cpu().numpy()

    else:
        raise ValueError("Either text or image must be provided.")


# === Search Function ===
def clip_search(text_query=None, image_path=None, top_k=5):
    """Perform CLIP multimodal search"""
    if not text_query and not image_path:
        return []

    image = None
    if image_path and os.path.exists(image_path):
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"⚠️ Image load error: {e}")
            image = None

    embedding = generate_combined_embedding(text=text_query, image=image)
    results = index.query(vector=embedding.tolist(), top_k=top_k, include_metadata=True)

    return [
        {
            "productDisplayName": match.metadata.get("productDisplayName", ""),
            "score": match.score,
            "id": match.id
        }
        for match in results.matches
    ]
