import re
import time
import requests
import csv
from urllib.parse import urlparse

# -------------------------------------------------------
# ROBOTS.TXT RULES (from your provided data)
# -------------------------------------------------------

SHOPEE_DISALLOWED = [
    "/cart/",
    "/checkout/",
    "/buyer/login/otp",
    "/user/",
    "/me/",
    "/order/",
    "/daily_discover/",
    "/mall/just-for-you/",
    "/from_same_shop/",
    "/you_may_also_like/",
    "/find_similar_products/",
    "/top_products",
    "/index.html",
    "/api/v4/shop/rcmd_items",
    "/api/v4/recommend/",
    "/api/v4/homepage/get_daily_discover",
]

LAZADA_DISALLOWED = [
    "/wow/gcp/ph/member/login-signup",
    "/undefined/",
]

CRAWL_DELAY = 1  # seconds


# -------------------------------------------------------
# UTILITIES
# -------------------------------------------------------

def path_is_allowed(domain, path):
    """Check if a URL path complies with robots.txt rules"""
    rules = SHOPEE_DISALLOWED if "shopee" in domain else LAZADA_DISALLOWED

    for dis in rules:
        if dis.endswith("*"):  # prefix block
            if path.startswith(dis[:-1]):
                return False
        else:  # simple contains match
            if dis in path:
                return False
    return True


def save_to_csv(reviews):
    with open("reviews.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["review"])
        for r in reviews:
            writer.writerow([r])
    print(f"\nSaved {len(reviews)} reviews → reviews.csv")


# -------------------------------------------------------
# SHOPEE SCRAPER
# -------------------------------------------------------

def extract_ids_shopee(url):
    """
    Extract shopid and itemid from Shopee URL.
    Supports:
    - /product/... (old style)
    - -i.<shopid>.<itemid> (new style)
    """
    # New style: -i.<shopid>.<itemid>
    match = re.search(r"-i\.(\d+)\.(\d+)", url)
    if match:
        return match.group(1), match.group(2)
    
    # Old style: /product/<shopid>/<itemid>
    match = re.search(r"/product/(\d+)/(\d+)", url)
    if match:
        return match.group(1), match.group(2)
    
    return None, None


def scrape_shopee(url):
    domain = urlparse(url).netloc
    shopid, itemid = extract_ids_shopee(url)
    if not shopid:
        print("❌ Could not extract shopid/itemid from URL.")
        return []

    endpoint_base = "/api/v4/item/get_ratings"
    if not path_is_allowed(domain, endpoint_base):
        print(f"❌ Access to {endpoint_base} is disallowed by robots.txt")
        return []

    print("Scraping Shopee reviews (robots.txt-safe)…")
    reviews = []

    for offset in range(0, 200, 50):
        api = f"https://shopee.ph{endpoint_base}?itemid={itemid}&shopid={shopid}&limit=50&offset={offset}"

        time.sleep(CRAWL_DELAY)
        r = requests.get(api, headers={"User-Agent": "Mozilla/5.0"})
        
        try:
            data = r.json()
        except:
            break

        rating_data = data.get("data", {}).get("ratings", [])
        if not rating_data:
            break

        for rate in rating_data:
            comment = rate.get("comment")
            if comment:
                reviews.append(comment)

    return reviews


# -------------------------------------------------------
# LAZADA SCRAPER
# -------------------------------------------------------

def extract_id_lazada(url):
    match = re.search(r"itemId=(\d+)", url)
    return match.group(1) if match else None


def scrape_lazada(url):
    domain = urlparse(url).netloc
    item_id = extract_id_lazada(url)

    if not item_id:
        item_id = input("Enter Lazada itemId (found via DevTools request): ")

    endpoint_path = "/pdp/review/getReviewList"

    if not path_is_allowed(domain, endpoint_path):
        print(f"❌ Access to {endpoint_path} is disallowed by robots.txt")
        return []

    print("Scraping Lazada reviews (robots.txt-safe)…")
    reviews = []

    for page in range(1, 11):
        api = (
            f"https://{domain}{endpoint_path}"
            f"?itemId={item_id}&pageSize=20&pageNo={page}"
        )

        time.sleep(CRAWL_DELAY)
        r = requests.get(api, headers={"User-Agent": "Mozilla/5.0"})
        
        try:
            data = r.json()
        except:
            break

        items = data.get("result", {}).get("data", [])
        if not items:
            break

        for rev in items:
            text = rev.get("reviewContent")
            if text:
                reviews.append(text)

    return reviews


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

url = input("Paste Shopee or Lazada product URL: ").strip()
domain = urlparse(url).netloc

if "shopee" in domain:
    results = scrape_shopee(url)

elif "lazada" in domain:
    results = scrape_lazada(url)

else:
    print("❌ Unsupported domain. Only Shopee and Lazada allowed.")
    exit()

save_to_csv(results)
print("Done! (All requests obeyed robots.txt and crawl-delay)")
