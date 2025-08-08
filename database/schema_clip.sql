CREATE TABLE clip_products (
    id UUID PRIMARY KEY,
    product_display_name TEXT,
    gender TEXT,
    article_type TEXT,
    base_colour TEXT,
    season TEXT,
    usage TEXT,
    pinecone_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
