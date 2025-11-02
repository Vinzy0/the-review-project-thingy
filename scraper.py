from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import os

def scrape_reviews_selenium(url, limit=50):
    print("🚀 Starting Chrome browser (stealth mode)...")
    
    options = webdriver.ChromeOptions()
    
    # Make Selenium undetectable
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')
    options.add_argument('--start-maximized')
    
    # Add realistic user agent
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    # Hide webdriver property
    driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
        'source': '''
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            })
        '''
    })
    
    try:
        # Go to Shopee homepage first
        print("🌐 Opening Shopee...")
        driver.get("https://shopee.ph")
        time.sleep(3)
        
        print("\n" + "="*60)
        print("⏸️  PLEASE LOG IN MANUALLY:")
        print("1. Look at the Chrome window that just opened")
        print("2. Click 'Log In' and enter your credentials")
        print("3. Complete any verification if needed")
        print("4. Once you're logged in, come back here")
        print("="*60)
        input("\nPress Enter AFTER you've successfully logged in...")
        
        # Now go to the product page
        print("📦 Loading product page...")
        driver.get(url)
        time.sleep(5)
        
        reviews = []
        
        # Scroll to load reviews
        print("📜 Scrolling to load reviews...")
        for i in range(6):
            driver.execute_script("window.scrollBy(0, 800);")
            time.sleep(1.5)
        
        # Try clicking "View All" or "See More" button
        try:
            print("🔍 Looking for 'View All Reviews' button...")
            buttons = driver.find_elements(By.XPATH, 
                "//button[contains(text(), 'View All') or contains(text(), 'See All') or contains(., 'View More')]")
            
            if buttons:
                driver.execute_script("arguments[0].scrollIntoView(true);", buttons[0])
                time.sleep(1)
                buttons[0].click()
                print("✅ Clicked 'View All Reviews'")
                time.sleep(3)
                
                # Scroll more after clicking
                for i in range(5):
                    driver.execute_script("window.scrollBy(0, 600);")
                    time.sleep(1)
        except Exception as e:
            print(f"ℹ️ Couldn't find 'View All' button: {e}")
        
        # Extract reviews - multiple strategies
        print("🔍 Extracting reviews...")
        
        # Strategy 1: Shopee's main review containers
        review_elements = driver.find_elements(By.CSS_SELECTOR, 
            "div.shopee-product-rating__content, div[class*='product-rating__content']")
        
        if not review_elements:
            # Strategy 2: Look for any div with review-like text
            review_elements = driver.find_elements(By.XPATH, 
                "//div[contains(@class, 'rating') or contains(@class, 'review')]")
        
        if not review_elements:
            # Strategy 3: Broad search
            review_elements = driver.find_elements(By.XPATH, 
                "//div[string-length(text()) > 20]")
        
        print(f"📝 Found {len(review_elements)} potential review elements")
        
        seen_reviews = set()  # Avoid duplicates
        
        for idx, element in enumerate(review_elements[:limit*2]):  # Check more elements
            try:
                text = element.text.strip()
                
                # Filter out non-review text
                if (text and 
                    len(text) > 20 and 
                    len(text) < 1000 and
                    text not in seen_reviews and
                    not text.startswith(('©', 'Shopee', 'Help', 'Cart'))):
                    
                    # Try to find rating stars
                    try:
                        stars = element.find_elements(By.CSS_SELECTOR, 
                            "svg[fill*='ee4d2d'], svg[class*='rating'], .shopee-rating-stars__lit")
                        rating = len(stars) if len(stars) <= 5 else 5
                    except:
                        rating = "N/A"
                    
                    reviews.append({
                        "product_name": driver.title.split('|')[0].strip(),
                        "rating": rating if rating != "N/A" else "N/A",
                        "review": text,
                        "hair_type": ""
                    })
                    
                    seen_reviews.add(text)
                    print(f"  ✓ Review {len(reviews)}: {text[:60]}...")
                    
                    if len(reviews) >= limit:
                        break
                        
            except Exception as e:
                continue
        
        # Debug: If no reviews found, save a screenshot
        if not reviews:
            screenshot_path = "shopee_page_debug.png"
            driver.save_screenshot(screenshot_path)
            print(f"\n⚠️ No reviews found. Saved screenshot to: {screenshot_path}")
            print("Check the screenshot to see what the page looks like.")
        
        return reviews
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return []
        
    finally:
        print("\n💡 Browser will stay open for 10 seconds so you can check...")
        time.sleep(10)
        driver.quit()
        print("✅ Browser closed")

def save_to_csv(data, filename="reviews_dataset.csv"):
    if not data:
        print("⚠️ No data to save")
        return
    
    df = pd.DataFrame(data)
    
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', index=False, header=False, encoding='utf-8-sig')
    else:
        df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    print(f"✅ Saved {len(data)} reviews to {filename}")

if __name__ == "__main__":
    product_url = input("Enter Shopee product link: ").strip()
    
    try:
        reviews = scrape_reviews_selenium(product_url, limit=50)
        
        if reviews:
            save_to_csv(reviews)
            print(f"\n🎉 Successfully scraped {len(reviews)} reviews!")
        else:
            print("\n⚠️ No reviews found")
            print("Possible reasons:")
            print("  - Product has no written reviews (only star ratings)")
            print("  - Reviews section didn't load")
            print("  - Page structure changed")
            
    except Exception as e:
        print(f"❌ Error: {e}")