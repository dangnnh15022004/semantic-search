from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import uuid

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')
pc = Pinecone(api_key="pcsk_4Em9zF_NcfUg4U1n3tGfMVGqnuLv4KudrMs1CeKBnsSRSkQPokMuhsujXuCy8M6aRExKhJ")
index = pc.Index("semantic-fashion")

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
    return jsonify({"message": "Product added", "id": vector_id})

if __name__ == "__main__":
    app.run(debug=True)
