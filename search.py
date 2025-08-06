from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os

# === Thông tin Pinecone ===
PINECONE_API_KEY = "pcsk_4Em9zF_NcfUg4U1n3tGfMVGqnuLv4KudrMs1CeKBnsSRSkQPokMuhsujXuCy8M6aRExKhJ"  # ẩn bớt nếu public code
INDEX_NAME = "semantic-fashion"

# === Khởi tạo Pinecone và kết nối index ===
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# === Load model ===
model = SentenceTransformer('all-MiniLM-L6-v2')

# === Nhập truy vấn người dùng ===
query = input("🔎 Nhập mô tả sản phẩm bạn muốn tìm: ")
query_embedding = model.encode([query])[0].tolist()

# === Truy vấn top 5 vector gần nhất ===
results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)

# === Hiển thị kết quả ===
print("\n📌 Kết quả tìm kiếm:")
for i, match in enumerate(results['matches'], 1):
    print(f"\n{i}. ID: {match['id']}")
    print(f"   Tên SP: {match['metadata']['productDisplayName']}")
    print(f"   Mô tả: {match['metadata']['text']}")
    print(f"   Điểm tương đồng: {match['score']:.4f}")
