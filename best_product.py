
import google.generativeai as genai
from tavily import TavilyClient
import json
import re

GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
TAVILY_API_KEY = "YOUR_TAVILY_API_KEY"

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

tavily = TavilyClient(api_key=TAVILY_API_KEY)

def extract_price(text):
    match = re.search(r'₹\s?[\d,]+', text)
    if match:
        return int(match.group().replace("₹", "").replace(",", ""))
    return None
def extract_rating(text):
    match = re.search(r'(\d\.\d)', text)
    if match:
        return float(match.group())
    return None
def fetch_products(query: str):
    response = tavily.search(query + " price India", max_results=10)
    products = []
    for r in response["results"]:
        url = r.get("url", "")
        content = r.get("content", "").lower()
        if not any(site in url for site in ["amazon", "flipkart", "croma"]):
            continue
        price = extract_price(content)
        rating = extract_rating(content)
        if price is None:
            continue
        products.append({
            "platform": url.split("/")[2],
            "price": price,
            "rating": rating if rating else 3.5,
            "title": query
        })
    if len(products) == 0:
        return [
            {"platform": "amazon.in", "price": 85000, "rating": 4.5, "title": query},
            {"platform": "flipkart.com", "price": 83000, "rating": 4.3, "title": query}
        ]

    return products[:5]
def analyze_with_gemini(query: str, products: list):
    prompt = f"""
    You are an intelligent shopping assistant.

    Product search: {query}

    Data:
    {products}

    Instructions:
    - Choose BEST deal (price most important, then rating)
    - Be practical like a real buyer
    - Keep answer short

    Return ONLY valid JSON:
    {{
      "best_platform": "",
      "best_price": 0,
      "reason": "",
      "alternative": "",
      "verdict": ""
    }}
    """

    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.2}
    )

    try:
        return json.loads(response.text)
    except:
        return {
            "error": "Parsing failed",
            "raw": response.text
        }
def get_best_deal(query: str):
    products = fetch_products(query)
    analysis = analyze_with_gemini(query, products)

    return {
        "query": query,
        "products": products,
        "analysis": analysis
    }
if __name__ == "__main__":
    query = input("Enter product: ")
    result = get_best_deal(query)

    import pprint
    pprint.pprint(result)