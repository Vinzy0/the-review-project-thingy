from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import re

# -------------------------------------------------------------------
# Function: Extract shop_id and item_id from Shopee URL
# -------------------------------------------------------------------
def extract_ids_from_url(url):
    pattern = r"i\.(\d+)\.(\d+)"
    match = re.search(pattern, url)
    if match:
        return match.group(1), match.group(2)
    else:
        raise ValueError("Could not extract shop_id and item_id from the URL.")

# -------------------------------------------------------------------
# Function: Scrape reviews using Selenium
# -------------------------------------------------------------------
def scrape_reviews_selenium(url, limit=50):
    print("🚀 Starting Chrome browser...")
    
    # Setup Chrome options
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless')  # Uncomment to run without showing browser
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    # Initialize driver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    try:
        driver.get(url)
        print("⏳ Loading page...")
        time.sleep(5)  # Wait for page to load
        
        reviews = []
        
        # Scroll down to load reviews
        print("📜 Scrolling to load reviews...")
        for i in range(5):  # Scroll 5 times
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
        
        # Wait for reviews section
        wait = WebDriverWait(driver, 10)
        
        # Click on "View All Reviews" or similar button if it exists
        try:
            view_all_button = driver.find_element(By.XPATH, "//button[contains(text(), 'View All') or contains(text(), 'See All')]")
            view_all_button.click()
            time.sleep(3)
        except:
            print("ℹ️ No 'View All' button found, continuing...")
        
        # Find review elements (adjust selectors based on Shopee's structure)
        review_elements = driver.find_elements(By.CSS_SELECTOR, "div.shopee-product-rating__content")
        
        if not review_elements:
            # Try alternative selector
            review_elements = driver.find_elements(By.XPATH, "//div[contains(@class, 'product-rating')]")
        
        print(f"📝 Found {len(review_elements)} review elements")
        
        for idx, element in enumerate(review_elements[:limit]):
            try:
                # Extract rating (stars)
                rating_element = element.find_elements(By.CSS_SELECTOR, "div.shopee-product-rating__rating")
                rating = len(rating_element[0].find_elements(By.CSS_SELECTOR, "svg")) if rating_element else 0
                
                # Extract review text
                comment_element = element.find_elements(By.CSS_SELECTOR, "div.shopee-product-rating__content, div[class*='comment']")
                comment = comment_element[0].text if comment_element else ""
                
                if comment:  # Only add if there's actual text
                    reviews.append({
                        "product_name": driver.title,
                        "rating": rating,
                        "review": comment,
                        "hair_type": ""
                    })
                    print(f"  ✓ Review {idx+1}: {comment[:50]}...")
                
            except Exception as e:
                print(f"  ⚠️ Error extracting review {idx+1}: {e}")
                continue
        
        return reviews
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []
        
    finally:
        driver.quit()
        print("✅ Browser closed")

# -------------------------------------------------------------------
# Function: Save to CSV
# -------------------------------------------------------------------
def save_to_csv(data, filename="reviews_dataset.csv"):
    if not data:
        print("⚠️ No data to save")
        return
        
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"✅ Saved {len(data)} reviews to {filename}")

# -------------------------------------------------------------------
# MAIN SCRIPT
# -------------------------------------------------------------------
if __name__ == "__main__":
    product_url = input("Enter Shopee product link: ").strip()
    
    try:
        reviews = scrape_reviews_selenium(product_url, limit=50)
        
        if reviews:
            save_to_csv(reviews)
            print(f"\n🎉 Successfully scraped {len(reviews)} reviews!")
        else:
            print("\n⚠️ No reviews found. The page structure might have changed.")
            
    except Exception as e:
        print(f"❌ Error: {e}")