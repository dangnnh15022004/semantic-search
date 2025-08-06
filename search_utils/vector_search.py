from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from config import PINECONE_API_KEY, PINECONE_INDEX

model = SentenceTransformer("all-MiniLM-L6-v2")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

def search_by_text(query, top_k=5):
    vector = model.encode([query])[0].tolist()
    result = index.query(vector=vector, top_k=top_k, include_metadata=True)

    matches = result.get("matches", [])
    simplified_results = []

    for match in matches:
        metadata = match.get("metadata", {})
        simplified_results.append({
            "productDisplayName": metadata.get("productDisplayName", "Unknown"),
            "score": match.get("score", 0.0)
        })

    return simplified_results
