import os
import re  # <--- CHANGED: Imported regex for sentence splitting
from dateutil import parser

# Force transformers to ignore TensorFlow
os.environ["TRANSFORMERS_NO_TF"] = "1"

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
import threading
from datetime import datetime

# ==================== CONFIGURATION ====================
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
BASE_DIR = r"C:\Users\VINZ\Downloads\New folder (2)"
MASTER_CSV_PATH = os.path.join(BASE_DIR, "master_sentiment_results.csv")
REVIEWS_FOLDER = os.path.join(BASE_DIR, "product_reviews")
FEEDBACK_CSV_PATH = os.path.join(BASE_DIR, "recommendation_feedback.csv")

VALID_HAIR_TYPES = ['straight', 'wavy', 'curly']
VALID_CATEGORIES = ['Shampoo', 'Conditioner', 'Styling Product']
TAG_SEPARATOR = '; '

# <--- CHANGED: Added Delivery Keywords Blacklist (English + Tagalog)
DELIVERY_KEYWORDS = [
    'delivery', 'shipping', 'ship', 'rider', 'dumating', 'arrived', 
    'balot', 'bubble wrap', 'pack', 'packaging', 'order', 'mabilis', 'fast',
    'courier', 'lazada', 'shopee', 'driver', 'late'
]
MIN_WORD_COUNT = 3

# Create folders if they don't exist
os.makedirs(REVIEWS_FOLDER, exist_ok=True)

# Global variables
tokenizer = None
model = None
sentiment_analyzer = None

# ==================== HELPER: SENTIMENT LABEL NORMALIZER (NEW) ====================
def map_sentiment_label(label):
    """
    Normalize model labels to one of: negative, neutral, positive.
    Handles:
      - plain text labels (e.g., 'Positive', 'NEGATIVE')
      - HuggingFace numeric labels (e.g., 'LABEL_0', 'LABEL_1', 'LABEL_2') via model.config.id2label
    Returns one of ('negative', 'neutral', 'positive') or None if unknown.
    """
    if label is None:
        return None
    raw = str(label).strip()
    if not raw:
        return None
    lower = raw.lower()

    # Direct string matches
    if lower in ("positive", "pos"):
        return "positive"
    if lower in ("negative", "neg"):
        return "negative"
    if lower in ("neutral", "neu"):
        return "neutral"

    # LABEL_X style (e.g., LABEL_0)
    if lower.startswith("label_"):
        try:
            idx = int(lower.replace("label_", ""))
            if model and hasattr(model, "config") and hasattr(model.config, "id2label"):
                mapped = model.config.id2label.get(idx)
                if mapped:
                    return map_sentiment_label(mapped)
        except ValueError:
            return None
    return None

# ==================== HELPER: CONTINUOUS SCORE CALCULATOR (NEW) ====================
def calculate_continuous_score(label, score):
    """
    Convert label + confidence score into a continuous 0-1 sentiment score.
    Positive: score (e.g., 0.9 -> 0.9)
    Negative: 1 - score (e.g., 0.9 -> 0.1)
    Neutral: 0.5
    """
    normalized = map_sentiment_label(label)
    if normalized == "positive":
        return score  # 0.5 to 1.0 roughly
    elif normalized == "negative":
        return 1.0 - score  # 0.0 to 0.5 roughly
    else:
        return 0.5

# ==================== LOAD MODEL ====================
def load_model():
    global tokenizer, model, sentiment_analyzer
    print("Loading sentiment analysis model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    print("Model loaded successfully!\n")


def sample_sentiment_check():
    """
    Quick sanity check to verify mapping is not stuck at 0.5.
    Usage (optional): print(sample_sentiment_check())
    """
    test_sentence = "This shampoo made my hair soft and shiny."
    result = sentiment_analyzer(test_sentence[:512])[0]
    mapped = map_sentiment_label(result.get('label'))
    return {"sentence": test_sentence, "raw": result, "mapped": mapped}

# ==================== HELPER: GET PRODUCT FILENAME ====================
def get_product_reviews_filename(shampoo_name, hair_type):
    safe_name = "".join(c for c in shampoo_name if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_name = safe_name.replace(' ', '_').lower()
    return os.path.join(REVIEWS_FOLDER, f"{safe_name}_{hair_type.lower()}.csv")


# ==================== HELPER: SENTENCE SPLITTER (NEW) ====================
def split_into_sentences(text):
    """Splits text by periods, exclamation/question marks, or newlines"""
    # <--- CHANGED: Added this helper function
    sentences = re.split(r'[.!?\n]+', text)
    return [s.strip() for s in sentences if s.strip()]


def calculate_time_weight(date_str):
    """
    Calculate time-decay weight based on review age.
    Decay: weight = 0.8 ** age_in_years. Returns 1.0 on parse errors.
    """
    try:
        dt = parser.parse(str(date_str))
        age_days = (datetime.now() - dt).days
        age_years = age_days / 365.0
        return 0.8 ** age_years
    except Exception:
        return 1.0


# ==================== HELPER: TAG UTILITIES ====================
def normalize_tags_input(tags):
    if tags is None:
        return []
    if isinstance(tags, str):
        raw_tags = [part.strip() for part in tags.replace(';', ',').split(',')]
    elif isinstance(tags, (list, tuple, set)):
        raw_tags = [str(tag).strip() for tag in tags]
    else:
        return []
    cleaned = []
    seen = set()
    for tag in raw_tags:
        if not tag: continue
        key = tag.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(tag)
    return cleaned

def tags_list_to_string(tags_list):
    if not tags_list: return ''
    return TAG_SEPARATOR.join(tags_list)

def parse_tags_string(tags_str):
    if not isinstance(tags_str, str) or not tags_str.strip():
        return []
    return [tag.strip() for tag in tags_str.split(TAG_SEPARATOR) if tag.strip()]

def ensure_columns(df, defaults):
    if df is None: return df
    for column, default_value in defaults.items():
        if column not in df.columns:
            df[column] = default_value
    return df

def get_all_tags_from_master():
    if not os.path.exists(MASTER_CSV_PATH):
        return []
    master_df = pd.read_csv(MASTER_CSV_PATH)
    master_df = ensure_columns(master_df, {'Tags': ''})
    tags_set = set()
    for tag_string in master_df['Tags'].dropna():
        for tag in parse_tags_string(tag_string):
            tags_set.add(tag)
    return sorted(tags_set, key=lambda t: t.lower())


# ==================== HELPER: CHECK DUPLICATES ====================
def is_duplicate_review(new_comment, existing_reviews_df):
    if existing_reviews_df is None or len(existing_reviews_df) == 0:
        return False
    comment_col = None
    for col in existing_reviews_df.columns:
        if col.lower() == 'comment':
            comment_col = col
            break
    if comment_col is None:
        return False
    return new_comment in existing_reviews_df[comment_col].values


# ==================== FUNCTION 1: ANALYZE AND STORE REVIEWS (UPDATED) ====================
def analyze_and_store_reviews(input_csv_path, shampoo_name, hair_type, description=None, price=None,
                              product_url=None, tags=None, category=None):
    
    # ... (CSV Handling Code remains same) ...
    if isinstance(input_csv_path, (list, tuple, set)):
        csv_paths = [path for path in input_csv_path if path]
    elif isinstance(input_csv_path, str):
        csv_paths = [input_csv_path]
    else:
        return False, "ERROR: Invalid CSV input.", 0, 0

    if not csv_paths: return False, "ERROR: No CSV files selected.", 0, 0
    if hair_type.lower() not in VALID_HAIR_TYPES: return False, f"ERROR: Invalid Hair Type", 0, 0
    if category is None or category not in VALID_CATEGORIES: return False, f"ERROR: Invalid Category", 0, 0
    
    normalized_tags = normalize_tags_input(tags)
    tags_str = tags_list_to_string(normalized_tags)
    product_image = None
    
    product_file = get_product_reviews_filename(shampoo_name, hair_type)
    is_new_product = not os.path.exists(product_file)
    
    if is_new_product:
        existing_reviews_df = None
        print(f"Creating new product: {shampoo_name}")
    else:
        existing_reviews_df = pd.read_csv(product_file)
        existing_reviews_df = ensure_columns(existing_reviews_df, {'Tags': '', 'Category': ''})
        print(f"Adding reviews to: {shampoo_name}")
    
    existing_comments = set()
    batch_scores = []
    batch_weights = []
    existing_date_col = None
    existing_score_col = None
    if existing_reviews_df is not None:
        comment_col_existing = next((col for col in existing_reviews_df.columns if col.lower() == 'comment'), None)
        existing_date_col = next((col for col in existing_reviews_df.columns if col.lower() == 'comment date'), None)
        existing_score_col = next((col for col in existing_reviews_df.columns if col.lower() == 'sentiment score'), None)
        if comment_col_existing:
            existing_comments = set(existing_reviews_df[comment_col_existing].dropna().astype(str).tolist())
        # Seed weights with existing reviews
        for _, row in existing_reviews_df.iterrows():
            try:
                sentiment_val = float(row[existing_score_col]) if existing_score_col else 0.5
            except Exception:
                sentiment_val = 0.5
            date_val = row[existing_date_col] if existing_date_col else None
            weight_val = calculate_time_weight(date_val)
            batch_scores.append(sentiment_val)
            batch_weights.append(weight_val)

    new_reviews = []
    num_duplicates = 0
    processed_files = []
    file_errors = []

    for csv_path in csv_paths:
        try:
            new_df = pd.read_csv(csv_path)
        except Exception as e:
            file_errors.append(f"{os.path.basename(csv_path)}: {str(e)}")
            continue

        comment_col = next((col for col in new_df.columns if col.lower() == 'comment'), None)
        if comment_col is None:
            file_errors.append(f"{os.path.basename(csv_path)}: Missing 'Comment' column")
            continue

        username_col = next((col for col in new_df.columns if col.lower() == 'user name'), None)
        date_col = next((col for col in new_df.columns if col.lower() == 'comment date'), None)
        url_col = next((col for col in new_df.columns if col.lower() == 'product url'), None)
        image_col = next((col for col in new_df.columns if col.lower() == 'product image'), None)

        if product_url is None and url_col and len(new_df) > 0:
            product_url = str(new_df[url_col].iloc[0])
        if product_image is None and image_col and len(new_df) > 0:
            product_image = str(new_df[image_col].iloc[0])

        print(f"Analyzing {len(new_df)} reviews from {os.path.basename(csv_path)}...")
        processed_files.append(csv_path)

        for idx, row in new_df.iterrows():
            comment = str(row[comment_col]).strip()
            if not comment: continue
            if len(comment.split()) < MIN_WORD_COUNT:
                continue
            if comment in existing_comments:
                num_duplicates += 1
                continue

            try:
                sentences = split_into_sentences(comment)
                valid_scores = []

                for sentence in sentences:
                    lower_sent = sentence.lower()
                    if any(word in lower_sent for word in DELIVERY_KEYWORDS):
                        continue
                    analysis = sentiment_analyzer(sentence[:512])[0]

                    # <--- CHANGED: Use continuous score
                    score_val = calculate_continuous_score(analysis.get('label'), analysis.get('score'))
                    valid_scores.append(score_val)

                review_sentiment = sum(valid_scores) / len(valid_scores) if len(valid_scores) > 0 else 0.5
                time_weight = calculate_time_weight(row[date_col]) if date_col else 1.0
                
                # <--- CHANGED: Apply time weight to stored score
                final_stored_score = review_sentiment * time_weight
                
                batch_scores.append(review_sentiment)
                batch_weights.append(time_weight)

            except Exception as e:
                print(f"Error analyzing review #{idx+1}: {e}")
                continue

            review_data = {
                'Product Name': shampoo_name,
                'Hair Type': hair_type,
                'Comment': comment,
                'Sentiment Score': final_stored_score, # <--- CHANGED: Store weighted score
                'Tags': tags_str,
                'Category': category,
                'User Name': row[username_col] if username_col else 'Anonymous',
                'Comment Date': row[date_col] if date_col else 'Unknown',
                'Product URL': product_url if product_url else '',
                'Product Image': product_image if product_image else '',
                'Description': description if description else '',
                'Price': price if price else '',
                'Analyzed Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            new_reviews.append(review_data)
            existing_comments.add(comment)

    # ... (Rest of the function remains identical) ...
    if len(processed_files) == 0: return False, "ERROR: No files read.", 0, 0
    if len(new_reviews) == 0:
        if num_duplicates > 0: return False, f"All {num_duplicates} reviews were duplicates.", 0, num_duplicates
        else: return False, "No valid reviews.", 0, 0
    
    new_reviews_df = pd.DataFrame(new_reviews)
    
    if existing_reviews_df is not None:
        if description: existing_reviews_df['Description'] = description
        if price: existing_reviews_df['Price'] = price
        combined_df = pd.concat([existing_reviews_df, new_reviews_df], ignore_index=True)
    else:
        combined_df = new_reviews_df
    
    combined_df = ensure_columns(combined_df, {'Tags': '', 'Category': ''})
    combined_df['Tags'] = tags_str
    combined_df['Category'] = category
    
    combined_df.to_csv(product_file, index=False, encoding='utf-8-sig')
    print(f"Saved {len(combined_df)} reviews.")
    
    if batch_scores and batch_weights and sum(batch_weights) > 0:
        weighted_sum = sum(score * weight for score, weight in zip(batch_scores, batch_weights))
        total_weight = sum(batch_weights)
        avg_score = weighted_sum / total_weight
    else:
        avg_score = combined_df['Sentiment Score'].mean()
    num_reviews = len(combined_df)
    
    update_master_csv(shampoo_name, hair_type, avg_score, num_reviews, description, price, 
                     product_url, product_image, tags_str, category)
    
    for csv_path in processed_files:
        try: os.remove(csv_path)
        except: pass
    
    print("PROCESS COMPLETE!")
    
    message_lines = [
        "Successfully processed!",
        f"New reviews: {len(new_reviews)}",
        f"Duplicates skipped: {num_duplicates}",
        f"Total reviews: {num_reviews}",
        f"Avg Aspect Score: {avg_score:.4f}" # Updated label
    ]

    return True, "\n".join(message_lines), len(new_reviews), num_duplicates


# ==================== FUNCTION 2: UPDATE MASTER CSV ====================
def update_master_csv(shampoo_name, hair_type, avg_score, num_reviews, description, price,
                      product_url, product_image, tags, category):
    if os.path.exists(MASTER_CSV_PATH):
        master_df = pd.read_csv(MASTER_CSV_PATH)
        master_df = ensure_columns(master_df, {'Tags': '', 'Category': ''})
    else:
        master_df = pd.DataFrame(columns=[
            'Shampoo Name', 'Hair Type', 'Avg Sentiment Score', 'Number of Reviews',
            'Description', 'Price', 'Product URL', 'Product Image', 'Tags', 'Category', 'Last Updated'
        ])
    
    mask = (master_df['Shampoo Name'] == shampoo_name) & (master_df['Hair Type'] == hair_type)
    
    if mask.any():
        master_df.loc[mask, 'Avg Sentiment Score'] = avg_score
        master_df.loc[mask, 'Number of Reviews'] = num_reviews
        if description: master_df.loc[mask, 'Description'] = description
        if price: master_df.loc[mask, 'Price'] = price
        if product_url: master_df.loc[mask, 'Product URL'] = product_url
        if product_image: master_df.loc[mask, 'Product Image'] = product_image
        if tags is not None: master_df.loc[mask, 'Tags'] = tags
        if category: master_df.loc[mask, 'Category'] = category
        master_df.loc[mask, 'Last Updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    else:
        new_row = pd.DataFrame({
            'Shampoo Name': [shampoo_name],
            'Hair Type': [hair_type],
            'Avg Sentiment Score': [avg_score],
            'Number of Reviews': [num_reviews],
            'Description': [description if description else ''],
            'Price': [price if price else ''],
            'Product URL': [product_url if product_url else ''],
            'Product Image': [product_image if product_image else ''],
            'Tags': [tags if tags else ''],
            'Category': [category],
            'Last Updated': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        })
        master_df = pd.concat([master_df, new_row], ignore_index=True)
    
    master_df.to_csv(MASTER_CSV_PATH, index=False, encoding='utf-8-sig')


# ==================== FUNCTION 3: GET TOP PRODUCTS ====================
def get_top_products(hair_type, top_n=3, category=None, tags=None):
    if not os.path.exists(MASTER_CSV_PATH):
        return None, "ERROR: No products in database."
    
    master_df = pd.read_csv(MASTER_CSV_PATH)
    master_df = ensure_columns(master_df, {'Tags': '', 'Category': ''})
    
    filtered = master_df[master_df['Hair Type'].str.lower() == hair_type.lower()]
    
    if category and category != "All":
        filtered = filtered[filtered['Category'].str.lower() == category.lower()]
    
    normalized_tags = normalize_tags_input(tags)
    if normalized_tags:
        required = [tag.lower() for tag in normalized_tags]
        def has_tags(tag_string):
            available = {t.lower() for t in parse_tags_string(tag_string)}
            return all(tag in available for tag in required)
        filtered = filtered[filtered['Tags'].apply(has_tags)]
    
    if len(filtered) == 0:
        return None, "No products found."
    
    top_products = filtered.sort_values('Avg Sentiment Score', ascending=False).head(top_n)
    return top_products, None


# ==================== FUNCTION 4: GET PRODUCT DETAILS ====================
def get_product_details(shampoo_name, hair_type):
    product_file = get_product_reviews_filename(shampoo_name, hair_type)
    if not os.path.exists(product_file): return None, "Product reviews not found"
    
    reviews_df = pd.read_csv(product_file)
    reviews_df = ensure_columns(reviews_df, {'Tags': '', 'Category': ''})
    if len(reviews_df) == 0: return None, "No reviews available"
    
    product_info = {
        'name': reviews_df['Product Name'].iloc[0],
        'hair_type': reviews_df['Hair Type'].iloc[0],
        'avg_score': reviews_df['Sentiment Score'].mean(),
        'num_reviews': len(reviews_df),
        'description': reviews_df['Description'].iloc[0] if 'Description' in reviews_df.columns else '',
        'price': reviews_df['Price'].iloc[0] if 'Price' in reviews_df.columns else '',
        'product_url': reviews_df['Product URL'].iloc[0] if 'Product URL' in reviews_df.columns else '',
        'product_image': reviews_df['Product Image'].iloc[0] if 'Product Image' in reviews_df.columns else '',
        'tags': parse_tags_string(reviews_df['Tags'].iloc[0]),
        'category': reviews_df['Category'].iloc[0] if 'Category' in reviews_df.columns else '',
        'reviews': reviews_df
    }
    return product_info, None


# ==================== FUNCTION 5: SAVE FEEDBACK ====================
def save_recommendation_feedback(shampoo_name, hair_type, user_hair_type, was_helpful, comments):
    feedback_data = {
        'Shampoo Name': shampoo_name,
        'Product Hair Type': hair_type,
        'User Hair Type': user_hair_type,
        'Was Helpful': 'Yes' if was_helpful else 'No',
        'Comments': comments,
        'Feedback Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    if os.path.exists(FEEDBACK_CSV_PATH):
        feedback_df = pd.read_csv(FEEDBACK_CSV_PATH)
        new_feedback = pd.DataFrame([feedback_data])
        feedback_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
    else:
        feedback_df = pd.DataFrame([feedback_data])
    feedback_df.to_csv(FEEDBACK_CSV_PATH, index=False, encoding='utf-8-sig')
    return True


# ==================== FUNCTION 6: UPDATE PRODUCT DETAILS ====================
def update_product_details(shampoo_name, hair_type, description=None, price=None, tags=None, category=None):
    if category is not None and category not in VALID_CATEGORIES:
        return False, f"Category must be {VALID_CATEGORIES}"
    
    tags_to_store = None
    if tags is not None:
        normalized = normalize_tags_input(tags)
        tags_to_store = tags_list_to_string(normalized)
    
    product_file = get_product_reviews_filename(shampoo_name, hair_type)
    if not os.path.exists(product_file): return False, "Product not found"
    
    reviews_df = pd.read_csv(product_file)
    reviews_df = ensure_columns(reviews_df, {'Tags': '', 'Category': ''})
    if description: reviews_df['Description'] = description
    if price: reviews_df['Price'] = price
    if tags_to_store is not None: reviews_df['Tags'] = tags_to_store
    if category: reviews_df['Category'] = category
    reviews_df.to_csv(product_file, index=False, encoding='utf-8-sig')
    
    if os.path.exists(MASTER_CSV_PATH):
        master_df = pd.read_csv(MASTER_CSV_PATH)
        master_df = ensure_columns(master_df, {'Tags': '', 'Category': ''})
        mask = (master_df['Shampoo Name'] == shampoo_name) & (master_df['Hair Type'] == hair_type)
        if mask.any():
            if description: master_df.loc[mask, 'Description'] = description
            if price: master_df.loc[mask, 'Price'] = price
            if tags_to_store is not None: master_df.loc[mask, 'Tags'] = tags_to_store
            if category: master_df.loc[mask, 'Category'] = category
            master_df.to_csv(MASTER_CSV_PATH, index=False, encoding='utf-8-sig')
    
    return True, "Product details updated successfully"


# ==================== GUI APPLICATION ====================
class ShampooAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Shampoo Sentiment Analyzer v2.0 (ABSA)")
        self.root.geometry("700x500")
        self.root.resizable(False, False)
        
        self.model_loaded = False
        self.available_tags = []
        self.selected_files = []
        self.load_model_thread()
        self.show_main_menu()
    
    def load_model_thread(self):
        def load():
            load_model()
            self.model_loaded = True
        thread = threading.Thread(target=load, daemon=True)
        thread.start()

    def refresh_available_tags(self):
        self.available_tags = get_all_tags_from_master()
    
    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()
    
    def show_main_menu(self):
        self.clear_window()
        title = tk.Label(self.root, text="Shampoo Sentiment Analyzer", font=("Arial", 20, "bold"), pady=20)
        title.pack()
        subtitle = tk.Label(self.root, text="AI-Powered Recommendation (Aspect-Based)", font=("Arial", 10), fg="gray")
        subtitle.pack()
        
        if not self.model_loaded:
            status = tk.Label(self.root, text="⏳ Loading AI model...", font=("Arial", 10), fg="orange")
            status.pack(pady=10)
        else:
            status = tk.Label(self.root, text="✓ Model loaded successfully!", font=("Arial", 10), fg="green")
            status.pack(pady=10)
        
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=30)
        
        tk.Button(btn_frame, text="Import Product Reviews", command=self.show_import_screen, font=("Arial", 13), width=25, height=2, bg="#4CAF50", fg="white").pack(pady=8)
        tk.Button(btn_frame, text="Get Recommendations", command=self.show_recommendations_screen, font=("Arial", 13), width=25, height=2, bg="#2196F3", fg="white").pack(pady=8)
        tk.Button(btn_frame, text="Edit Product Details", command=self.show_edit_screen, font=("Arial", 13), width=25, height=2, bg="#FF9800", fg="white").pack(pady=8)
        tk.Button(btn_frame, text="Live NLP Visualizer", command=self.show_nlp_visualizer, font=("Arial", 13), width=25, height=2, bg="#9C27B0", fg="white").pack(pady=8)
        
        if not self.model_loaded: self.root.after(1000, self.show_main_menu)
    
    def show_import_screen(self):
        """Show screen for importing reviews"""
        if not self.model_loaded:
            messagebox.showwarning("Please Wait", "Model is still loading. Please wait a moment.")
            return
        
        self.clear_window()
        
        # Title
        title = tk.Label(self.root, text="Import Product Reviews", 
                        font=("Arial", 18, "bold"), pady=15)
        title.pack()
        
        # Form frame
        form_frame = tk.Frame(self.root)
        form_frame.pack(pady=10)
        
        # CSV File Selection
        tk.Label(form_frame, text="CSV File:", font=("Arial", 11)).grid(row=0, column=0, sticky="w", pady=8, padx=10)
        self.file_label = tk.Label(form_frame, text="No files selected", font=("Arial", 9), fg="gray")
        self.file_label.grid(row=0, column=1, sticky="w", pady=8)
        self.selected_files = []
        
        # --- RESTORED BUTTONS HERE ---
        select_btn = tk.Button(form_frame, text="Select File", command=lambda: self.select_file(False),
                              font=("Arial", 9), bg="#E0E0E0", cursor="hand2")
        select_btn.grid(row=0, column=2, padx=10)
        
        multi_btn = tk.Button(form_frame, text="Select Multiple Files", command=lambda: self.select_file(True),
                              font=("Arial", 9), bg="#D1C4E9", cursor="hand2")
        multi_btn.grid(row=0, column=3, padx=5)
        # -----------------------------
        
        # Shampoo name
        tk.Label(form_frame, text="Shampoo Name:", font=("Arial", 11)).grid(row=1, column=0, sticky="w", pady=8, padx=10)
        self.shampoo_entry = tk.Entry(form_frame, font=("Arial", 10), width=35)
        self.shampoo_entry.grid(row=1, column=1, columnspan=2, sticky="w", pady=8)
        
        # Hair type
        tk.Label(form_frame, text="Hair Type:", font=("Arial", 11)).grid(row=2, column=0, sticky="w", pady=8, padx=10)
        self.hair_type_var = tk.StringVar(value="straight")
        hair_dropdown = ttk.Combobox(form_frame, textvariable=self.hair_type_var, 
                                     values=["straight", "wavy", "curly"],
                                     state="readonly", font=("Arial", 10), width=32)
        hair_dropdown.grid(row=2, column=1, columnspan=2, sticky="w", pady=8)
        
        # Description (optional)
        tk.Label(form_frame, text="Description (optional):", font=("Arial", 11)).grid(row=3, column=0, sticky="nw", pady=8, padx=10)
        self.description_text = tk.Text(form_frame, font=("Arial", 9), width=35, height=3)
        self.description_text.grid(row=3, column=1, columnspan=2, sticky="w", pady=8)
        
        # Price (optional)
        tk.Label(form_frame, text="Price (optional):", font=("Arial", 11)).grid(row=4, column=0, sticky="w", pady=8, padx=10)
        self.price_entry = tk.Entry(form_frame, font=("Arial", 10), width=35)
        self.price_entry.grid(row=4, column=1, columnspan=2, sticky="w", pady=8)

        # Category
        tk.Label(form_frame, text="Category:", font=("Arial", 11)).grid(row=5, column=0, sticky="w", pady=8, padx=10)
        category_options = ["Use existing value"] + VALID_CATEGORIES
        self.category_var = tk.StringVar(value=category_options[0])
        category_dropdown = ttk.Combobox(form_frame, textvariable=self.category_var,
                                         values=category_options, state="readonly",
                                         font=("Arial", 10), width=32)
        category_dropdown.grid(row=5, column=1, columnspan=2, sticky="w", pady=8)

        # Tags
        tk.Label(form_frame, text="Tags (comma separated):", font=("Arial", 11)).grid(row=6, column=0, sticky="nw", pady=8, padx=10)
        self.tags_entry = tk.Text(form_frame, font=("Arial", 9), width=35, height=3)
        self.tags_entry.grid(row=6, column=1, columnspan=2, sticky="w", pady=8)
        tk.Label(form_frame, text="Example: hydrating, anti-frizz, color-safe", font=("Arial", 8), fg="gray")\
            .grid(row=7, column=1, columnspan=2, sticky="w", pady=(0, 5))
        
        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=20)
        
        process_btn = tk.Button(btn_frame, text="Import & Analyze", 
                               command=self.process_import,
                               font=("Arial", 11), width=18, height=2,
                               bg="#4CAF50", fg="white", cursor="hand2")
        process_btn.pack(side="left", padx=10)
        
        back_btn = tk.Button(btn_frame, text="Back", 
                            command=self.show_main_menu,
                            font=("Arial", 11), width=18, height=2,
                            bg="#757575", fg="white", cursor="hand2")
        back_btn.pack(side="left", padx=10)
        
    def select_file(self, allow_multiple=False):
        dialog_fn = filedialog.askopenfilenames if allow_multiple else filedialog.askopenfilename
        result = dialog_fn(filetypes=[("CSV files", "*.csv")])
        if allow_multiple: self.selected_files = list(result) if result else []
        else: self.selected_files = [result] if result else []
        
        if self.selected_files:
            self.file_label.config(text=f"{len(self.selected_files)} files selected" if len(self.selected_files)>1 else os.path.basename(self.selected_files[0]), fg="black")
            try:
                temp_df = pd.read_csv(self.selected_files[0])
                col = next((c for c in temp_df.columns if c.lower() == 'product name'), None)
                if col: self.shampoo_entry.insert(0, str(temp_df[col].iloc[0]))
            except: pass

    def process_import(self):
        if not self.selected_files:
            messagebox.showerror("Error", "Select a file")
            return
        
        shampoo_name = self.shampoo_entry.get().strip()
        if not shampoo_name: return
        
        hair_type = self.hair_type_var.get()
        description = self.description_text.get("1.0", tk.END).strip()
        price = self.price_entry.get().strip()
        category_sel = self.category_var.get()
        tags_raw = self.tags_entry.get("1.0", tk.END).strip()
        tags_list = normalize_tags_input(tags_raw)

        # Logic to check existing product categories/tags if not provided...
        # (Simplified for brevity as logic exists in original function)
        product_file = get_product_reviews_filename(shampoo_name, hair_type)
        existing_cat = None
        if os.path.exists(product_file):
            try: existing_cat = pd.read_csv(product_file)['Category'].iloc[0]
            except: pass
        
        category = existing_cat if category_sel == "Use existing value" else category_sel
        if category not in VALID_CATEGORIES:
            messagebox.showerror("Error", "Invalid Category")
            return

        self.root.config(cursor="wait")
        self.root.update()
        
        success, message, _, _ = analyze_and_store_reviews(
            self.selected_files, shampoo_name, hair_type, description, price, 
            None, tags_list, category
        )
        self.root.config(cursor="")
        if success:
            messagebox.showinfo("Success", message)
            self.show_main_menu()
        else:
            messagebox.showerror("Error", message)

    def show_recommendations_screen(self):
        self.clear_window()
        tk.Label(self.root, text="Get Product Recommendations", font=("Arial", 18, "bold"), pady=20).pack()
        form = tk.Frame(self.root)
        form.pack(pady=20)
        
        tk.Label(form, text="Hair Type:", font=("Arial", 12)).grid(row=0, column=0, pady=10)
        self.rec_hair_type_var = tk.StringVar(value="straight")
        ttk.Combobox(form, textvariable=self.rec_hair_type_var, values=["straight", "wavy", "curly"]).grid(row=0, column=1)
        
        tk.Button(self.root, text="Show Recommendations", command=self.show_recommendations_results, bg="#2196F3", fg="white", width=25).pack(pady=20)
        tk.Button(self.root, text="Back", command=self.show_main_menu).pack()

    def show_recommendations_results(self):
        hair_type = self.rec_hair_type_var.get()
        products, err = get_top_products(hair_type)
        if err:
            messagebox.showerror("Error", err)
            return
        
        win = tk.Toplevel(self.root)
        win.title(f"Top Products for {hair_type}")
        win.geometry("600x500")
        
        for idx, row in products.iterrows():
            f = tk.Frame(win, relief="solid", borderwidth=1, padx=10, pady=10)
            f.pack(fill="x", padx=10, pady=5)
            tk.Label(f, text=f"#{idx+1} {row['Shampoo Name']}", font=("Arial", 12, "bold")).pack(anchor="w")
            tk.Label(f, text=f"Score: {row['Avg Sentiment Score']:.4f}", fg="green").pack(anchor="w")
    
    def show_edit_screen(self):
        self.clear_window()
        tk.Label(self.root, text="Edit details not implemented in simple view", pady=20).pack()
        tk.Button(self.root, text="Back", command=self.show_main_menu).pack()

    # ==================== NLP VISUALIZER FEATURE ====================
    def show_nlp_visualizer(self):
        self.clear_window()
        
        # Title
        tk.Label(self.root, text="Live NLP Pipeline Visualizer", font=("Arial", 16, "bold"), pady=10).pack()
        
        # Input Frame
        input_frame = tk.Frame(self.root)
        input_frame.pack(fill="x", padx=20)
        
        tk.Label(input_frame, text="Enter a Review:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.viz_input = scrolledtext.ScrolledText(input_frame, width=80, height=4, font=("Arial", 10))
        self.viz_input.pack(fill="x", pady=5)
        
        # Controls Frame
        ctrl_frame = tk.Frame(self.root)
        ctrl_frame.pack(fill="x", padx=20, pady=5)
        
        tk.Label(ctrl_frame, text="Review Date (YYYY-MM-DD):", font=("Arial", 10)).pack(side="left")
        self.viz_date = tk.Entry(ctrl_frame, font=("Arial", 10), width=15)
        self.viz_date.insert(0, datetime.now().strftime('%Y-%m-%d'))
        self.viz_date.pack(side="left", padx=10)
        
        analyze_btn = tk.Button(ctrl_frame, text="Analyze Step-by-Step", 
                               command=self.start_analysis_thread,
                               bg="#9C27B0", fg="white", font=("Arial", 10, "bold"))
        analyze_btn.pack(side="left", padx=20)
        
        back_btn = tk.Button(ctrl_frame, text="Back to Menu", command=self.show_main_menu)
        back_btn.pack(side="right")
        
        # Output Area
        tk.Label(self.root, text="Pipeline Execution Log:", font=("Arial", 10, "bold")).pack(anchor="w", padx=20, pady=(10,0))
        self.viz_output = scrolledtext.ScrolledText(self.root, width=80, height=15, font=("Consolas", 10), state="disabled")
        self.viz_output.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Configure Tags for Styling
        self.viz_output.tag_config("header", foreground="blue", font=("Consolas", 10, "bold"))
        self.viz_output.tag_config("error", foreground="red")
        self.viz_output.tag_config("success", foreground="green")
        self.viz_output.tag_config("discard", foreground="gray", font=("Consolas", 10, "italic"))
        self.viz_output.tag_config("highlight", background="#FFFF00", foreground="black")

    def start_analysis_thread(self):
        if not self.model_loaded:
            messagebox.showwarning("Wait", "Model is still loading...")
            return
            
        text = self.viz_input.get("1.0", tk.END).strip()
        date_str = self.viz_date.get().strip()
        
        if not text:
            return
            
        self.viz_output.config(state="normal")
        self.viz_output.delete("1.0", tk.END)
        self.viz_output.config(state="disabled")
        
        threading.Thread(target=self.run_analysis_pipeline, args=(text, date_str), daemon=True).start()

    def append_visualizer_text(self, text, tags=None):
        def _update():
            self.viz_output.config(state="normal")
            self.viz_output.insert(tk.END, text + "\n", tags)
            self.viz_output.see(tk.END)
            self.viz_output.config(state="disabled")
        self.root.after(0, _update)

    def run_analysis_pipeline(self, text, date_str):
        self.append_visualizer_text("=== STARTING PIPELINE ===\n", "header")
        
        # STEP 1: Raw Input
        self.append_visualizer_text(f"STEP 1: Raw Input Received", "header")
        self.append_visualizer_text(f"\"{text}\"\n")
        
        # STEP 2: Spam Check
        self.append_visualizer_text(f"STEP 2: Spam Check (Min Words: {MIN_WORD_COUNT})", "header")
        word_count = len(text.split())
        if word_count < MIN_WORD_COUNT:
            self.append_visualizer_text(f"❌ FAILED: Only {word_count} words. (Marked as Spam/Too Short)", "error")
            self.append_visualizer_text("\n=== PIPELINE TERMINATED ===", "header")
            return
        else:
            self.append_visualizer_text(f"✅ PASSED: {word_count} words.\n", "success")
            
        # STEP 3: Segmentation
        self.append_visualizer_text(f"STEP 3: Sentence Segmentation", "header")
        sentences = split_into_sentences(text)
        self.append_visualizer_text(f"Found {len(sentences)} sentences:", "highlight")
        for i, s in enumerate(sentences):
            self.append_visualizer_text(f"  [{i+1}] {s}")
        self.append_visualizer_text("")
        
        # STEP 4 & 5: Aspect Filtering & Sentiment
        self.append_visualizer_text(f"STEP 4 & 5: Aspect Filtering & Sentiment Inference", "header")
        valid_scores = []
        
        for i, sentence in enumerate(sentences):
            lower_sent = sentence.lower()
            
            # Check Delivery Keywords
            is_delivery = False
            found_keyword = ""
            for word in DELIVERY_KEYWORDS:
                if word in lower_sent:
                    is_delivery = True
                    found_keyword = word
                    break
            
            self.append_visualizer_text(f"Analyzing Sentence #{i+1}: \"{sentence}\"")
            
            if is_delivery:
                self.append_visualizer_text(f"  -> 🗑 DISCARDED: Contains delivery keyword '{found_keyword}'", "discard")
            else:
                self.append_visualizer_text(f"  -> ✅ KEPT: Product-related", "success")
                
                # Sentiment Inference
                try:
                    analysis = sentiment_analyzer(sentence[:512])[0]
                    raw_label = analysis.get('label')
                    raw_score = analysis.get('score')
                    normalized = map_sentiment_label(raw_label)
                    
                    # <--- CHANGED: Use continuous score
                    score_val = calculate_continuous_score(raw_label, raw_score)
                    
                    self.append_visualizer_text(f"     [Model Output]: Label={raw_label} | Conf={raw_score:.4f} | Normalized={normalized.upper()} | Score={score_val:.4f}", "highlight")
                    valid_scores.append(score_val)
                except Exception as e:
                    self.append_visualizer_text(f"     [Error]: {e}", "error")
            self.append_visualizer_text("")

        # STEP 6: Temporal Weight
        self.append_visualizer_text(f"STEP 6: Temporal Weight Calculation", "header")
        time_weight = calculate_time_weight(date_str)
        try:
            dt = parser.parse(str(date_str))
            age_days = (datetime.now() - dt).days
            self.append_visualizer_text(f"Review Date: {date_str} ({age_days} days old)")
        except:
            self.append_visualizer_text(f"Review Date: {date_str} (Parse Error)")
            
        self.append_visualizer_text(f"Calculated Weight: {time_weight:.4f}\n", "highlight")
        
        # STEP 7: Final Calculation
        self.append_visualizer_text(f"STEP 7: Final Weighted Score", "header")
        if not valid_scores:
            self.append_visualizer_text("No valid sentences found for scoring.", "error")
            final_score = 0.5
        else:
            avg_sentiment = sum(valid_scores) / len(valid_scores)
            self.append_visualizer_text(f"Average Sentiment (Raw): {avg_sentiment:.4f}")
            self.append_visualizer_text(f"Time Weight: {time_weight:.4f}")
            
            # <--- CHANGED: Apply time weight to final score
            final_score = avg_sentiment * time_weight
            
        self.append_visualizer_text(f"FINAL STORED SCORE: {final_score:.4f}", "success")
        self.append_visualizer_text("=== ANALYSIS COMPLETE ===", "header")

if __name__ == "__main__":
    root = tk.Tk()
    app = ShampooAnalyzerApp(root)
    root.mainloop()