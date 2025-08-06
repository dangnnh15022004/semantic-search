import torch
from transformers import CLIPModel, CLIPProcessor
from pinecone import Pinecone

# === 1. Pinecone cấu hình ===
api_key = "pcsk_4Em9zF_NcfUg4U1n3tGfMVGqnuLv4KudrMs1CeKBnsSRSkQPokMuhsujXuCy8M6aRExKhJ"
index_name = "semantic-clip"
pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)

# === 2. Load CLIP model (chuẩn 512 chiều) ===
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def get_clip_text_embedding(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = clip_model.get_text_features(**inputs)
    return embeddings[0].cpu().numpy().tolist()

# === 3. Truy vấn từ khóa ===
query = "Chelsea"
query_vector = get_clip_text_embedding(query)

# === 4. Truy vấn Pinecone index ===
response = index.query(
    vector=query_vector,
    top_k=10,
    include_metadata=True
)

# === 5. Hiển thị kết quả ===
print(f"\nTop 10 kết quả gần nhất với từ khóa: '{query}'\n")

for i, match in enumerate(response["matches"], 1):
    print(f"[{i}] ID: {match['id']}")
    print(f"Score: {match['score']:.4f}")
    print("Metadata:", match.get("metadata", {}))
    print("-" * 60)
