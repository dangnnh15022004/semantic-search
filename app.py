from flask import Flask, request, jsonify, render_template
from uuid import uuid4
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX
from database.db import engine
from sqlalchemy import text
from search_utils.vector_search import search_by_text
from search_utils.keyword_search import keyword_search
from search_utils.hybrid_search import hybrid_search
from search_utils.clip_search import clip_search
import os

app = Flask(__name__)

# Load model và Pinecone index
model = SentenceTransformer("all-MiniLM-L6-v2")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Trang chủ
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", results=None, message=None)

# Thêm sản phẩm mới
@app.route("/add-product", methods=["POST"])
def add_product():
    data = request.form
    try:
        text_data = f"{data['productDisplayName']}. {data['gender']} {data['articleType']} in {data['baseColour']}, {data['usage']} - {data['season']} collection"
        embedding = model.encode([text_data])[0].tolist()
        vector_id = str(uuid4())

        # Lưu lên Pinecone
        index.upsert([(vector_id, embedding, {
            "productDisplayName": data["productDisplayName"],
            "gender": data["gender"],
            "articleType": data["articleType"],
            "baseColour": data["baseColour"],
            "season": data["season"],
            "usage": data["usage"],
            "text": text_data
        })])

        # Lưu vào PostgreSQL
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO products (id, product_display_name, gender, article_type, base_colour, season, usage, pinecone_id)
                VALUES (:id, :name, :gender, :type, :colour, :season, :usage, :pine_id)
            """), {
                "id": vector_id,
                "name": data["productDisplayName"],
                "gender": data["gender"],
                "type": data["articleType"],
                "colour": data["baseColour"],
                "season": data["season"],
                "usage": data["usage"],
                "pine_id": vector_id
            })

        return render_template("index.html", message="✅ Product added successfully!", results=None)

    except Exception as e:
        return render_template("index.html", message=f"❌ Error: {str(e)}", results=None)

# Semantic Search
@app.route("/semantic-search", methods=["POST"])
def semantic_search():
    query = request.form.get("query", "")
    if not query.strip():
        return render_template("index.html", message="⚠️ Please enter a search query.", results=None)
    results = search_by_text(query)
    return render_template("index.html", results=results, message=f"🔍 Semantic results for: '{query}'")

# Hybrid Search
@app.route("/hybrid-search", methods=["POST"])
def hybrid_search_web():
    query = request.form.get("query", "")
    if not query.strip():
        return render_template("index.html", message="⚠️ Please enter a search query.", results=None)
    results = hybrid_search(query)
    return render_template("index.html", hybrid_results=results, hybrid_query=query)

# CLIP Search
@app.route("/clip-search", methods=["POST"])
def clip_search_web():
    text_query = request.form.get("query", "")
    image = request.files.get("image")

    image_path = None
    if image:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        image_path = os.path.join(temp_dir, image.filename)
        image.save(image_path)

    if not text_query and not image_path:
        return render_template("index.html", message="⚠️ Please enter text or upload an image.", results=None)

    results = clip_search(text_query, image_path)
    return render_template("index.html", clip_results=results, clip_query=text_query)

# ---------- API Routes ----------
@app.route("/api/semantic-search", methods=["POST"])
def semantic_search_api():
    query = request.json.get("query", "")
    results = search_by_text(query)
    return jsonify(results)

@app.route("/api/keyword-search", methods=["POST"])
def keyword_search_api():
    query = request.json.get("query", "")
    results = keyword_search(query)
    return jsonify(results)

@app.route("/api/hybrid-search", methods=["POST"])
def hybrid_search_api():
    query = request.json.get("query", "")
    results = hybrid_search(query)
    return jsonify(results)
# ---------------------------------

if __name__ == "__main__":
    app.run(debug=True)
