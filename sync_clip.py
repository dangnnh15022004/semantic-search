import os
import uuid
import pandas as pd
import psycopg2
from pinecone import Pinecone
from config import POSTGRES_URL, PINECONE_API_KEY
from tqdm import tqdm

# === Tên Pinecone index mới (dành cho CLIP search) ===
PINECONE_INDEX = "semantic-clip"

# === Load styles.csv để lấy danh sách ID ===
df = pd.read_csv("styles.csv", on_bad_lines="skip")
df = df[['id']].dropna()
df = df.astype(str)

# === Kết nối Pinecone ===
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# === Kết nối PostgreSQL ===
conn = psycopg2.connect(POSTGRES_URL)
cursor = conn.cursor()

# === Lặp qua danh sách ID và insert metadata vào bảng clip_products ===
print("🚀 Đang đồng bộ dữ liệu từ Pinecone (semantic-clip) vào bảng clip_products...")
inserted = 0
skipped = 0
errors = 0

for vector_id in tqdm(df["id"], desc="🔁 Syncing CLIP", unit="vector"):
    try:
        record = index.fetch(ids=[vector_id]).vectors.get(vector_id)
        if not record:
            skipped += 1
            print(f"⚠️  Không tìm thấy vector {vector_id}")
            continue

        metadata = record.metadata
        cursor.execute("""
            INSERT INTO clip_products (
                id, product_display_name, gender, article_type, base_colour,
                season, usage, pinecone_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """, (
            str(uuid.uuid4()),
            metadata.get("productDisplayName"),
            metadata.get("gender"),
            metadata.get("category") or metadata.get("articleType"),
            metadata.get("colour"),
            metadata.get("season"),
            metadata.get("usage"),
            vector_id
        ))
        inserted += 1
    except Exception as e:
        errors += 1
        print(f"❌ Lỗi khi insert {vector_id}: {e}")

# === Hoàn tất ===
conn.commit()
cursor.close()
conn.close()

print(f"\n✅ Hoàn tất!")
print(f"   ✅ Đã thêm: {inserted}")
print(f"   ⚠️  Bỏ qua (không tìm thấy vector): {skipped}")
print(f"   ❌ Lỗi khi insert: {errors}")
