from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import uuid

# === Khởi tạo Flask ===
app = Flask(__name__)

# === Khởi tạo mô hình và Pinecone ===
model = SentenceTransformer('all-MiniLM-L6-v2')
pc = Pinecone(api_key="pcsk_4Em9zF_NcfUg4U1n3tGfMVGqnuLv4KudrMs1CeKBnsSRSkQPokMuhsujXuCy8M6aRExKhJ")
index = pc.Index("semantic-fashion")

# === API: Giao diện test đơn giản ===
@app.route("/", methods=["GET"])
def home():
    return """
        <h3>🧠 Semantic Search API</h3>
        <p>POST /add-product → thêm sản phẩm mới</p>
        <p>POST /semantic-search → tìm kiếm theo mô tả</p>
    """

# === API: Nhận sản phẩm mới và vector hóa ===
@app.route("/add-product", methods=["POST"])
def add_product():
    data = request.json
    text = f"{data['productDisplayName']}. {data['gender']} {data['articleType']} in {data['baseColour']}, {data['usage']} - {data['season']} collection"
    embedding = model.encode([text])[0].tolist()
    
    vector_id = str(uuid.uuid4())  # hoặc dùng mã sản phẩm
    metadata = {
        "text": text,
        **{k: data[k] for k in ["productDisplayName", "gender", "articleType", "baseColour", "season", "usage"]}
    }

    index.upsert([(vector_id, embedding, metadata)])
    return jsonify({"message": "✅ Product added", "id": vector_id})


# === API: Semantic Search theo mô tả ===
@app.route("/semantic-search", methods=["POST"])
def semantic_search():
    query = request.json["query"]
    embedding = model.encode([query])[0].tolist()
    
    results = index.query(
        vector=embedding,
        top_k=5,
        include_metadata=True
    )

    matches = [{
        "id": m["id"],
        "score": round(m["score"], 4),
        **m["metadata"]
    } for m in results["matches"]]

    return jsonify({
        "query": query,
        "top_results": matches
    })

# === Chạy app ===
if __name__ == "__main__":
    app.run(debug=True)
