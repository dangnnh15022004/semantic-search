from database.db import engine
from sqlalchemy import text

def keyword_search(query, top_k=5):
    with engine.begin() as conn:
        result = conn.execute(text("""
            SELECT *, ts_rank_cd(to_tsvector(product_display_name), plainto_tsquery(:q)) AS rank
            FROM products
            WHERE to_tsvector(product_display_name) @@ plainto_tsquery(:q)
            ORDER BY rank DESC
            LIMIT :k
        """), {"q": query, "k": top_k})
        
        # Chuyển từng row sang dict đúng cách
        return [dict(row._mapping) for row in result]
