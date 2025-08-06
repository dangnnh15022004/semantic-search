from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')
pc = Pinecone(api_key="pcsk_4Em9zF_NcfUg4U1n3tGfMVGqnuLv4KudrMs1CeKBnsSRSkQPokMuhsujXuCy8M6aRExKhJ")
index = pc.Index("semantic-fashion")

@app.route("/search", methods=["POST"])
def search():
    query = request.json["query"]
    embedding = model.encode([query])[0].tolist()
    results = index.query(vector=embedding, top_k=5, include_metadata=True)
    
    output = [{
        "id": match["id"],
        "score": match["score"],
        **match["metadata"]
    } for match in results["matches"]]

    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
