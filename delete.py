from pinecone import Pinecone

pc = Pinecone(api_key="pcsk_4Em9zF_NcfUg4U1n3tGfMVGqnuLv4KudrMs1CeKBnsSRSkQPokMuhsujXuCy8M6aRExKhJ")
index = pc.Index("semantic-fashion")

# ⚠️ XÓA TOÀN BỘ RECORD
index.delete(delete_all=True)

print("✅ Đã xóa toàn bộ dữ liệu khỏi Pinecone index.")
