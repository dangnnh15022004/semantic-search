import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from tqdm import tqdm

# === THÔNG TIN KẾT NỐI PINECONE ===
PINECONE_API_KEY = "pcsk_4Em9zF_NcfUg4U1n3tGfMVGqnuLv4KudrMs1CeKBnsSRSkQPokMuhsujXuCy8M6aRExKhJ"
PINECONE_ENV = "us-east-1"
INDEX_NAME = "semantic-fashion"

# === KHỞI TẠO KẾT NỐI ===
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(INDEX_NAME)

# === LOAD DATASET ===
print("📦 Đang tải dữ liệu styles.csv...")
df = pd.read_csv("styles.csv", on_bad_lines="skip")

# === LỌC & LÀM SẠCH DỮ LIỆU ===
columns_needed = ['id', 'gender', 'articleType', 'baseColour', 'season', 'usage', 'productDisplayName']
df = df[columns_needed].dropna()
df = df.astype(str)  # Tránh lỗi encode do kiểu dữ liệu

# === TẠO TEXT CHO EMBEDDING ===
def build_text(row):
    return f"{row['productDisplayName']}. {row['gender']} {row['articleType']} in {row['baseColour']}, {row['usage']} - {row['season']} collection"

df["text"] = df.apply(build_text, axis=1)

# === LOAD MÔ HÌNH EMBEDDING ===
print("🧠 Đang sinh vector với SentenceTransformer...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)

# === UPLOAD VECTORS VÀO PINECONE ===
print("🚀 Đang upload vector vào Pinecone...")
batch_size = 100
total = len(df)

for i in tqdm(range(0, total, batch_size), desc="Uploading"):
    i_end = min(i + batch_size, total)
    batch = []

    for j in range(i, i_end):
        row = df.iloc[j]
        vector_id = str(row["id"])
        vector = embeddings[j]
        metadata = {
            "text": row["text"],
            "productDisplayName": row["productDisplayName"],
            "gender": row["gender"],
            "category": row["articleType"],
            "colour": row["baseColour"],
            "season": row["season"],
            "usage": row["usage"]
        }
        batch.append((vector_id, vector.tolist(), metadata))

    index.upsert(vectors=batch)

print(f"✅ Hoàn tất! Đã upload {total} sản phẩm vào Pinecone.")
