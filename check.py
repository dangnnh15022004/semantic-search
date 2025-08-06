<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 96fca1fbc9f84820337754c60d4a9b865a9dc6e9
from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Thay bằng vector_id thực tế bạn muốn kiểm tra
vector_id = "dd11217d-dde0-48fc-9e53-13ae7d7a574c"

response = index.fetch(ids=[vector_id])
vector = response.vectors.get(vector_id)

if vector:
    print("✅ Found in Pinecone:", vector.metadata)
else:
    print("❌ Not found in Pinecone")
<<<<<<< HEAD
=======
check.py
>>>>>>> 55910b0 (Initial commit: Clean Semantic + Hybrid + CLIP search app)
=======
>>>>>>> 96fca1fbc9f84820337754c60d4a9b865a9dc6e9
