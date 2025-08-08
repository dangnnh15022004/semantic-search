from database.db import engine
from sqlalchemy import text

def keyword_search(query, top_k=5, filters=None):
    filters = filters or {}
    conditions = ["to_tsvector(product_display_name) @@ plainto_tsquery(:q)"]
    params = {"q": query, "k": top_k}

    if filters.get("base_colour"):
        conditions.append("base_colour ILIKE :base_colour")
        params["base_colour"] = filters["base_colour"]
    
    if filters.get("gender"):
        conditions.append("gender = :gender")
        params["gender"] = filters["gender"]

    if filters.get("article_type"):
        conditions.append("article_type = :article_type")
        params["article_type"] = filters["article_type"]

    if filters.get("season"):
        conditions.append("season = :season")
        params["season"] = filters["season"]

    where_clause = " AND ".join(conditions)

    with engine.begin() as conn:
        result = conn.execute(text(f"""
            SELECT *, ts_rank_cd(to_tsvector(product_display_name), plainto_tsquery(:q)) AS rank
            FROM products
            WHERE {where_clause}
            ORDER BY rank DESC
            LIMIT :k
        """), params)

        return [
            {
                "productDisplayName": row._mapping["product_display_name"],
                "gender": row._mapping.get("gender", ""),
                "articleType": row._mapping.get("article_type", ""),
                "baseColour": row._mapping.get("base_colour", ""),
                "season": row._mapping.get("season", ""),
                "usage": row._mapping.get("usage", ""),
                "pinecone_id": row._mapping.get("pinecone_id"),
                "score": 0.0
            }
            for row in result
        ]

