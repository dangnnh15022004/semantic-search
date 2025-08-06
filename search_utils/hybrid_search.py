from search_utils.vector_search import search_by_text
from search_utils.keyword_search import keyword_search


def hybrid_search(query, top_k=5):
    # Semantic search: dùng embedding để tìm các vector tương tự
    semantic_results = search_by_text(query, top_k=top_k * 2)

    # Keyword search: dùng full-text search trong PostgreSQL
    keyword_results = keyword_search(query, top_k=top_k * 2)

    combined = {}

    # Lưu kết quả semantic search vào dict
    for item in semantic_results:
        name = item["productDisplayName"]
        combined[name] = {
            "productDisplayName": name,
            "semantic_score": item.get("score", 0.0),
            "keyword_score": 0.0
        }

    # Gộp kết quả keyword search
    for item in keyword_results:
        name = item["product_display_name"]
        if name in combined:
            combined[name]["keyword_score"] = 1.0  # Có match keyword
        else:
            combined[name] = {
                "productDisplayName": name,
                "semantic_score": 0.0,
                "keyword_score": 1.0
            }

    # Cộng điểm và thêm trường 'score'
    for item in combined.values():
        item["score"] = 0.7 * item["semantic_score"] + 0.3 * item["keyword_score"]

    # Sắp xếp theo score tổng
    final_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)

