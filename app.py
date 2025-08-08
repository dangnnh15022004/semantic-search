from flask import Flask, request, jsonify, render_template
from uuid import uuid4
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import numpy as np
from pinecone import Pinecone
from config import PINECONE_API_KEY, PINECONE_INDEX
from database.db import engine
from sqlalchemy import text
from search_utils.vector_search import search_by_text
from search_utils.keyword_search import keyword_search
from search_utils.hybrid_search import hybrid_search
from search_utils.clip_search import clip_search

# === App init ===
app = Flask(__name__)

# === SentenceTransformer for semantic search ===
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

# === CLIP for image/text + Pinecone ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("semantic-clip")

# === CLIP embedding function ===
def generate_combined_embedding(text=None, image: Image.Image = None):
    if text and image:
        inputs = clip_processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_emb = clip_model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            image_emb = clip_model.get_image_features(pixel_values=inputs["pixel_values"])
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        return ((text_emb + image_emb) / 2).squeeze(0).cpu().numpy()
    elif text:
        inputs = clip_processor(text=[text], return_tensors="pt").to(device)
        with torch.no_grad():
            text_emb = clip_model.get_text_features(**inputs)
        return (text_emb / text_emb.norm(dim=-1, keepdim=True)).squeeze(0).cpu().numpy()
    elif image:
        inputs = clip_processor(images=[image], return_tensors="pt").to(device)
        with torch.no_grad():
            image_emb = clip_model.get_image_features(**inputs)
        return (image_emb / image_emb.norm(dim=-1, keepdim=True)).squeeze(0).cpu().numpy()
    else:
        raise ValueError("Either text or image must be provided.")

# === Trang chủ ===
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", results=None, message=None, active_tab="semantic")

# === Add product using CLIP ===
@app.route("/add-product", methods=["POST"])
def add_product():
    data = request.form
    image = request.files.get("image")

    try:
        # Tạo mô tả từ các trường dữ liệu
        text_data = f"{data['productDisplayName']}. {data['gender']} {data['articleType']} in {data['baseColour']}, {data['usage']} - {data['season']} collection"

        # Check thiếu dữ liệu
        if not text_data.strip() or not image:
            return render_template("index.html", message="⚠️ Cần nhập đầy đủ mô tả và ảnh để thêm sản phẩm!", results=None, active_tab="add")

        # Mở ảnh
        pil_image = Image.open(image).convert("RGB")

        # Tạo embedding với CLIP (text + image)
        embedding = generate_combined_embedding(text=text_data, image=pil_image)
        vector_id = str(uuid4())

        # Lưu ảnh vào static/images/
        image_path = os.path.join("static", "images", f"{vector_id}.jpg")
        image.stream.seek(0)
        image.save(image_path)

        # Lưu lên Pinecone
        index.upsert([(vector_id, embedding.tolist(), {
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
                INSERT INTO clip_products (id, product_display_name, gender, article_type, base_colour, season, usage, pinecone_id)
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

        return render_template("index.html", message="✅ Product added successfully!", results=None, active_tab="add")

    except Exception as e:
        return render_template("index.html", message=f"❌ Error: {str(e)}", results=None, active_tab="add")

# === Semantic Search ===
@app.route("/semantic-search", methods=["POST"])
def semantic_search():
    query = request.form.get("query", "")
    if not query.strip():
        return render_template(
            "index.html",
            message="⚠️ Please enter a search query.",
            results=None,
            active_tab="semantic"
        )
    results = search_by_text(query)
    return render_template(
        "index.html",
        results=results,
        message=f"🔍 Semantic results for: '{query}'",
        active_tab="semantic"
    )

# === Hybrid Search ===
@app.route("/hybrid-search", methods=["POST"])
def hybrid_search_web():
    query = request.form.get("query", "").strip()
    image = request.files.get("image")

    if not query and not image:
        return render_template("index.html", message="⚠️ Please enter a query or upload an image.", hybrid_results=None, active_tab="hybrid")

    filters = {
        "baseColour": request.form.get("base_colour", "").strip(),
        "gender": request.form.get("gender", "").strip(),
        "articleType": request.form.get("article_type", "").strip(),
        "season": request.form.get("season", "").strip(),
    }

    image_path = None
    if image:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        image_path = os.path.join(temp_dir, image.filename)
        image.save(image_path)

    results = hybrid_search(query, filters=filters, image_path=image_path)

    return render_template(
        "index.html",
        hybrid_results=results,
        hybrid_query=query,
        filters=filters,
        active_tab="hybrid"
    )

# === CLIP Search ===
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
        return render_template(
            "index.html",
            message="⚠️ Please enter text or upload an image.",
            clip_results=None,
            active_tab="clip"
        )

    results = clip_search(text_query, image_path)
    return render_template(
        "index.html",
        clip_results=results,
        clip_query=text_query,
        active_tab="clip"
    )

# === API Endpoints ===
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
    filters = request.json.get("filters", {})
    results = hybrid_search(query, filters=filters)
    return jsonify(results)

# === Run ===
if __name__ == "__main__":
    app.run(debug=True)
