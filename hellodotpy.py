import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os

# ==================== CONFIGURATION ====================
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
MASTER_CSV_PATH = r"C:\Users\VINZ\Downloads\New folder (2)\master_sentiment_results.csv"

# Alternative path format if you get unicode errors:
# MASTER_CSV_PATH = "C:/Users/VINZ/Downloads/New folder (2)/master_sentiment_results.csv"

# ==================== LOAD MODEL (once) ====================
print("Loading sentiment analysis model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
print("Model loaded successfully!\n")


# ==================== FUNCTION 1: ANALYZE NEW REVIEWS ====================
def analyze_reviews(input_csv_path, shampoo_name, hair_type):
    """
    Analyzes sentiment of reviews from a CSV file.
    
    Parameters:
    - input_csv_path: Path to CSV with reviews (must have 'feedback' column)
    - shampoo_name: Name of the shampoo product
    - hair_type: One of 'straight', 'wavy', or 'curly'
    
    Returns:
    - Average sentiment score
    """
    # Validate hair type
    valid_hair_types = ['straight', 'wavy', 'curly']
    if hair_type.lower() not in valid_hair_types:
        print(f"ERROR: Hair type must be one of {valid_hair_types}")
        return None
    
    # Read the reviews
    df = pd.read_csv(input_csv_path)
    
    if 'feedback' not in df.columns:
        print("ERROR: CSV must have a 'feedback' column with reviews")
        return None
    
    print(f"Analyzing {len(df)} reviews for {shampoo_name} ({hair_type} hair)...")
    
    # Run sentiment analysis
    results = []
    for review in df["feedback"]:
        try:
            analysis = sentiment_analyzer(str(review)[:512])[0]
            results.append(analysis)
        except Exception as e:
            print(f"Error analyzing review, skipping...")
            results.append({"label": "ERROR", "score": 0.0})
    
    # Calculate average score
    valid_scores = [r['score'] for r in results if r['label'] != 'ERROR']
    
    if not valid_scores:
        print("ERROR: No valid reviews analyzed")
        return None
    
    avg_score = sum(valid_scores) / len(valid_scores)
    print(f"Analysis complete! Average sentiment score: {avg_score:.4f}\n")
    
    return avg_score


# ==================== FUNCTION 2: APPEND TO MASTER CSV ====================
def append_to_master(shampoo_name, hair_type, avg_score):
    """
    Appends or updates the master CSV with sentiment results.
    
    Parameters:
    - shampoo_name: Name of the shampoo
    - hair_type: Hair type (straight/wavy/curly)
    - avg_score: Average sentiment score
    """
    # Check if master CSV exists
    if os.path.exists(MASTER_CSV_PATH):
        master_df = pd.read_csv(MASTER_CSV_PATH)
    else:
        # Create new master dataframe
        master_df = pd.DataFrame(columns=['Shampoo Name', 'Hair Type', 'Avg Sentiment Score'])
    
    # Check if this product already exists
    existing = master_df[(master_df['Shampoo Name'] == shampoo_name) & 
                         (master_df['Hair Type'] == hair_type)]
    
    if len(existing) > 0:
        # Update existing entry
        master_df.loc[(master_df['Shampoo Name'] == shampoo_name) & 
                      (master_df['Hair Type'] == hair_type), 'Avg Sentiment Score'] = avg_score
        print(f"Updated existing entry for {shampoo_name} ({hair_type})")
    else:
        # Add new entry
        new_row = pd.DataFrame({
            'Shampoo Name': [shampoo_name],
            'Hair Type': [hair_type],
            'Avg Sentiment Score': [avg_score]
        })
        master_df = pd.concat([master_df, new_row], ignore_index=True)
        print(f"Added new entry for {shampoo_name} ({hair_type})")
    
    # Save master CSV
    master_df.to_csv(MASTER_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"Master CSV updated: {MASTER_CSV_PATH}\n")


# ==================== FUNCTION 3: GET TOP PRODUCTS ====================
def get_top_products(hair_type, top_n=3):
    """
    Retrieves top N products for a specific hair type.
    
    Parameters:
    - hair_type: Hair type to filter by (straight/wavy/curly)
    - top_n: Number of top products to return (default 3)
    
    Returns:
    - DataFrame with top products
    """
    if not os.path.exists(MASTER_CSV_PATH):
        print("ERROR: Master CSV not found. Run analysis first!")
        return None
    
    master_df = pd.read_csv(MASTER_CSV_PATH)
    
    # Filter by hair type
    filtered = master_df[master_df['Hair Type'].str.lower() == hair_type.lower()]
    
    if len(filtered) == 0:
        print(f"No products found for {hair_type} hair type")
        return None
    
    # Sort by sentiment score and get top N
    top_products = filtered.sort_values('Avg Sentiment Score', ascending=False).head(top_n)
    
    print(f"\n{'='*60}")
    print(f"TOP {top_n} SHAMPOOS FOR {hair_type.upper()} HAIR")
    print(f"{'='*60}")
    for idx, row in top_products.iterrows():
        print(f"{row.name + 1}. {row['Shampoo Name']}")
        print(f"   Sentiment Score: {row['Avg Sentiment Score']:.4f}")
        print()
    
    return top_products


# ==================== MAIN WORKFLOW ====================
def process_new_product(input_csv, shampoo_name, hair_type):
    """
    Complete workflow: analyze reviews -> append to master CSV
    
    Parameters:
    - input_csv: Path to CSV with reviews
    - shampoo_name: Name of shampoo product
    - hair_type: Hair type (straight/wavy/curly)
    """
    print(f"\n{'='*60}")
    print(f"PROCESSING: {shampoo_name}")
    print(f"{'='*60}\n")
    
    # Step 1: Analyze reviews
    avg_score = analyze_reviews(input_csv, shampoo_name, hair_type)
    
    if avg_score is None:
        print("Analysis failed. Aborting.")
        return
    
    # Step 2: Append to master CSV
    append_to_master(shampoo_name, hair_type, avg_score)
    
    print(f"{'='*60}")
    print("PROCESS COMPLETE!")
    print(f"{'='*60}\n")


# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    # Example 1: Process a new product
    # Uncomment and modify these lines to run
    
    process_new_product(
         input_csv=r"C:\Users\VINZ\Downloads\New folder (2)\scraped_reviews.csv",
         shampoo_name="Pantene Pro-V Smooth & Sleek",
         hair_type="straight"
    )
    
    # Example 2: Get top products for a hair type
    
    # get_top_products(hair_type="curly", top_n=5)
    
    print("Script loaded! Use the functions:")
    print("1. process_new_product(input_csv, shampoo_name, hair_type)")
    print("2. get_top_products(hair_type, top_n=3)")
    print("\nUncomment the example usage section to test!")