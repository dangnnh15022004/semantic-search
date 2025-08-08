from PIL import Image
import os
from .clip_search import generate_combined_embedding
from .keyword_search import keyword_search
from pinecone import Pinecone
from config import PINECONE_API_KEY

# Pinecone setup
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("semantic-clip")

def hybrid_search(query, top_k=5, filters=None, image_path=None):
    filters = filters or {}

    # 1. Prepare image (if exists)
    image = None
    if image_path and os.path.exists(image_path):
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"⚠️ Image load error: {e}")

    # 2. Generate CLIP embedding and query Pinecone
    embedding = generate_combined_embedding(text=query, image=image)
    clip_results = index.query(vector=embedding.tolist(), top_k=top_k * 2, include_metadata=True)

    # 3. Filter CLIP matches
    def match_filter(metadata, filters):
        for key, value in filters.items():
            if value and str(metadata.get(key, "")).lower() != str(value).lower():
                return False
        return True

    filtered_clip = []
    for match in clip_results.matches:
        metadata = match.metadata or {}
        if match_filter(metadata, filters):
            filtered_clip.append({
                "productDisplayName": metadata.get("productDisplayName", ""),
                "gender": metadata.get("gender", ""),
                "articleType": metadata.get("articleType", ""),
                "baseColour": metadata.get("baseColour") or metadata.get("colour", ""),
                "season": metadata.get("season", ""),
                "usage": metadata.get("usage", ""),
                "score": match.score,
                "pinecone_id": match.id,
                "semantic_score": match.score,
                "keyword_score": 0.0
            })

    # 4. Keyword search from PostgreSQL
    keyword_results = keyword_search(query, top_k=top_k * 2, filters=filters)

    # 5. Merge results (by pinecone_id or product name as fallback)
    combined = {}

    for item in filtered_clip:
        key = item.get("pinecone_id") or item["productDisplayName"]
        combined[key] = item

    for item in keyword_results:
        key = item.get("pinecone_id") or item["productDisplayName"]
        if key in combined:
            combined[key]["keyword_score"] = 1.0

            # 🔁 Ensure missing metadata fields from keyword are merged in
            for field in ["gender", "articleType", "baseColour", "season", "usage"]:
                if not combined[key].get(field):
                    combined[key][field] = item.get(field, "")
        else:
            item["semantic_score"] = 0.0
            item["keyword_score"] = 1.0
            item["score"] = 0.3
            combined[key] = item

    # 6. Final score computation
    for item in combined.values():
        item["score"] = 0.7 * item.get("semantic_score", 0.0) + 0.3 * item.get("keyword_score", 0.0)

    # 7. Return top-k sorted by score
    sorted_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return sorted_results[:top_k]
