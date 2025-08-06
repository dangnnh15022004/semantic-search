import requests

# === Gửi sản phẩm mới
add_url = "http://127.0.0.1:5000/add-product"
product = {
    "productDisplayName": "Adidas Training Jacket",
    "gender": "Men",
    "articleType": "Jackets",
    "baseColour": "Black",
    "season": "Winter",
    "usage": "Training"
}

res1 = requests.post(add_url, json=product)
print("✅ Thêm sản phẩm:")
print(res1.json())

# === Tìm kiếm theo mô tả
search_url = "http://127.0.0.1:5000/semantic-search"
query = {"query": "black winter jacket for men"}

res2 = requests.post(search_url, json=query)
print("\n🔎 Kết quả tìm kiếm:")
for i, item in enumerate(res2.json()['top_results'], 1):
    print(f"{i}. {item['productDisplayName']} (score: {item['score']})")
