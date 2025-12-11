import os

# Force transformers to ignore TensorFlow, which is misconfigured in this environment
# and breaking the import of `pipeline`. We only need the PyTorch backend.
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

# Create folders if they don't exist
os.makedirs(REVIEWS_FOLDER, exist_ok=True)

# Global variables for model (load once)
tokenizer = None
model = None
sentiment_analyzer = None

# ==================== LOAD MODEL ====================
def load_model():
    global tokenizer, model, sentiment_analyzer
    print("Loading sentiment analysis model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    print("Model loaded successfully!\n")


# ==================== HELPER: GET PRODUCT FILENAME ====================
def get_product_reviews_filename(shampoo_name, hair_type):
    """Generate safe filename for product reviews"""
    safe_name = "".join(c for c in shampoo_name if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_name = safe_name.replace(' ', '_').lower()
    return os.path.join(REVIEWS_FOLDER, f"{safe_name}_{hair_type.lower()}.csv")


# ==================== HELPER: TAG UTILITIES ====================
def normalize_tags_input(tags):
    """Convert raw tag input into a list of unique, trimmed tags"""
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
        if not tag:
            continue
        key = tag.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(tag)
    return cleaned


def tags_list_to_string(tags_list):
    """Convert tag list to storage string"""
    if not tags_list:
        return ''
    return TAG_SEPARATOR.join(tags_list)


def parse_tags_string(tags_str):
    """Parse stored tag string into list"""
    if not isinstance(tags_str, str) or not tags_str.strip():
        return []
    return [tag.strip() for tag in tags_str.split(TAG_SEPARATOR) if tag.strip()]


def ensure_columns(df, defaults):
    """Ensure dataframe has required columns with default values"""
    if df is None:
        return df
    for column, default_value in defaults.items():
        if column not in df.columns:
            df[column] = default_value
    return df


def get_all_tags_from_master():
    """Return sorted list of all tags currently stored"""
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
    """Check if comment already exists (exact match)"""
    if existing_reviews_df is None or len(existing_reviews_df) == 0:
        return False
    
    # Find comment column (case-insensitive)
    comment_col = None
    for col in existing_reviews_df.columns:
        if col.lower() == 'comment':
            comment_col = col
            break
    
    if comment_col is None:
        return False
    
    # Check for exact match
    return new_comment in existing_reviews_df[comment_col].values


# ==================== FUNCTION 1: ANALYZE AND STORE REVIEWS ====================
def analyze_and_store_reviews(input_csv_path, shampoo_name, hair_type, description=None, price=None,
                              product_url=None, tags=None, category=None):
    """
    Analyzes sentiment and stores individual reviews + product details.

    `input_csv_path` can be a single CSV path or an iterable of paths for bulk imports.

    Returns:
    - (success, message, num_new_reviews, num_duplicates)
    """
    # Normalize csv path(s)
    if isinstance(input_csv_path, (list, tuple, set)):
        csv_paths = [path for path in input_csv_path if path]
    elif isinstance(input_csv_path, str):
        csv_paths = [input_csv_path]
    else:
        return False, "ERROR: Invalid CSV input. Please select one or more CSV files.", 0, 0

    if not csv_paths:
        return False, "ERROR: No CSV files selected.", 0, 0
    # Validate hair type
    if hair_type.lower() not in VALID_HAIR_TYPES:
        return False, f"ERROR: Hair type must be one of {VALID_HAIR_TYPES}", 0, 0
    
    # Validate category
    if category is None or category not in VALID_CATEGORIES:
        return False, f"ERROR: Category must be one of {VALID_CATEGORIES}", 0, 0
    
    normalized_tags = normalize_tags_input(tags)
    tags_str = tags_list_to_string(normalized_tags)
    
    product_image = None
    
    # Check if product already exists
    product_file = get_product_reviews_filename(shampoo_name, hair_type)
    is_new_product = not os.path.exists(product_file)
    
    if is_new_product:
        existing_reviews_df = None
        print(f"Creating new product: {shampoo_name} ({hair_type} hair)")
    else:
        existing_reviews_df = pd.read_csv(product_file)
        existing_reviews_df = ensure_columns(existing_reviews_df, {'Tags': '', 'Category': ''})
        print(f"Adding reviews to existing product: {shampoo_name} ({hair_type} hair)")
    
    # Prepare duplicate tracking
    existing_comments = set()
    if existing_reviews_df is not None:
        comment_col_existing = next((col for col in existing_reviews_df.columns if col.lower() == 'comment'), None)
        if comment_col_existing:
            existing_comments = set(existing_reviews_df[comment_col_existing].dropna().astype(str).tolist())

    # Analyze sentiment and check duplicates
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

        # Capture metadata once if not provided externally
        if product_url is None and url_col and len(new_df) > 0:
            candidate_url = str(new_df[url_col].iloc[0])
            if candidate_url:
                product_url = candidate_url
        if product_image is None and image_col and len(new_df) > 0:
            candidate_image = str(new_df[image_col].iloc[0])
            if candidate_image:
                product_image = candidate_image

        print(f"Analyzing {len(new_df)} reviews from {os.path.basename(csv_path)}...")
        processed_files.append(csv_path)

        for idx, row in new_df.iterrows():
            comment = str(row[comment_col]).strip()
            
            if not comment:
                continue

            if comment in existing_comments:
                num_duplicates += 1
                continue

            try:
                analysis = sentiment_analyzer(comment[:512])[0]
                sentiment_score = analysis['score']
            except Exception as e:
                print(f"Error analyzing review #{idx+1} in {os.path.basename(csv_path)}, skipping...")
                continue

            review_data = {
                'Product Name': shampoo_name,
                'Hair Type': hair_type,
                'Comment': comment,
                'Sentiment Score': sentiment_score,
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

    if len(processed_files) == 0:
        return False, "ERROR: Unable to read any of the selected CSV files.", 0, 0
    
    if len(new_reviews) == 0:
        if num_duplicates > 0:
            return False, f"All {num_duplicates} reviews were duplicates. No new data added.", 0, num_duplicates
        else:
            return False, "No valid reviews to process.", 0, 0
    
    # Convert to dataframe
    new_reviews_df = pd.DataFrame(new_reviews)
    
    # Merge with existing reviews
    if existing_reviews_df is not None:
        # Update description and price if provided
        if description:
            existing_reviews_df['Description'] = description
        if price:
            existing_reviews_df['Price'] = price
        
        combined_df = pd.concat([existing_reviews_df, new_reviews_df], ignore_index=True)
    else:
        combined_df = new_reviews_df
    
    combined_df = ensure_columns(combined_df, {'Tags': '', 'Category': ''})
    combined_df['Tags'] = tags_str
    combined_df['Category'] = category
    
    # Save reviews to product file
    combined_df.to_csv(product_file, index=False, encoding='utf-8-sig')
    print(f"Saved {len(combined_df)} total reviews to {product_file}")
    
    # Calculate average sentiment
    avg_score = combined_df['Sentiment Score'].mean()
    num_reviews = len(combined_df)
    
    # Update master CSV
    update_master_csv(shampoo_name, hair_type, avg_score, num_reviews, description, price, 
                     product_url, product_image, tags_str, category)
    
    # Delete original CSV(s)
    for csv_path in processed_files:
        try:
            os.remove(csv_path)
            print(f"Deleted original CSV: {csv_path}")
        except Exception as e:
            print(f"Warning: Could not delete file {csv_path}: {str(e)}")
    
    print(f"{'='*60}")
    print("PROCESS COMPLETE!")
    print(f"{'='*60}\n")
    
    message_lines = [
        "Successfully processed!",
        f"CSV files processed: {len(processed_files)}",
        f"New reviews: {len(new_reviews)}",
        f"Duplicates skipped: {num_duplicates}",
        f"Total reviews: {num_reviews}",
        f"Avg Sentiment: {avg_score:.4f}"
    ]

    if file_errors:
        message_lines.append("\nFiles with issues:")
        message_lines.extend(f"- {err}" for err in file_errors)

    return True, "\n".join(message_lines), len(new_reviews), num_duplicates


# ==================== FUNCTION 2: UPDATE MASTER CSV ====================
def update_master_csv(shampoo_name, hair_type, avg_score, num_reviews, description, price,
                      product_url, product_image, tags, category):
    """Update or add entry to master CSV"""
    
    # Check if master CSV exists
    if os.path.exists(MASTER_CSV_PATH):
        master_df = pd.read_csv(MASTER_CSV_PATH)
        master_df = ensure_columns(master_df, {'Tags': '', 'Category': ''})
    else:
        master_df = pd.DataFrame(columns=[
            'Shampoo Name', 'Hair Type', 'Avg Sentiment Score', 'Number of Reviews',
            'Description', 'Price', 'Product URL', 'Product Image', 'Tags', 'Category', 'Last Updated'
        ])
    
    # Check if product exists
    mask = (master_df['Shampoo Name'] == shampoo_name) & (master_df['Hair Type'] == hair_type)
    
    if mask.any():
        # Update existing
        master_df.loc[mask, 'Avg Sentiment Score'] = avg_score
        master_df.loc[mask, 'Number of Reviews'] = num_reviews
        if description:
            master_df.loc[mask, 'Description'] = description
        if price:
            master_df.loc[mask, 'Price'] = price
        if product_url:
            master_df.loc[mask, 'Product URL'] = product_url
        if product_image:
            master_df.loc[mask, 'Product Image'] = product_image
        if tags is not None:
            master_df.loc[mask, 'Tags'] = tags
        if category:
            master_df.loc[mask, 'Category'] = category
        master_df.loc[mask, 'Last Updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Updated master CSV entry for {shampoo_name}")
    else:
        # Add new
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
        print(f"Added new master CSV entry for {shampoo_name}")
    
    # Save
    master_df.to_csv(MASTER_CSV_PATH, index=False, encoding='utf-8-sig')


# ==================== FUNCTION 3: GET TOP PRODUCTS ====================
def get_top_products(hair_type, top_n=3, category=None, tags=None):
    """Retrieve top N products for a specific hair type"""
    if not os.path.exists(MASTER_CSV_PATH):
        return None, "ERROR: No products in database. Import some products first!"
    
    master_df = pd.read_csv(MASTER_CSV_PATH)
    master_df = ensure_columns(master_df, {'Tags': '', 'Category': ''})
    
    # Filter by hair type
    filtered = master_df[master_df['Hair Type'].str.lower() == hair_type.lower()]
    
    # Filter by category if provided (and not "All")
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
        return None, f"No products found for {hair_type} hair type with the selected filters."
    
    # Sort and get top N
    top_products = filtered.sort_values('Avg Sentiment Score', ascending=False).head(top_n)
    
    return top_products, None


# ==================== FUNCTION 4: GET PRODUCT DETAILS ====================
def get_product_details(shampoo_name, hair_type):
    """Get full details for a specific product"""
    product_file = get_product_reviews_filename(shampoo_name, hair_type)
    
    if not os.path.exists(product_file):
        return None, "Product reviews not found"
    
    reviews_df = pd.read_csv(product_file)
    reviews_df = ensure_columns(reviews_df, {'Tags': '', 'Category': ''})
    
    if len(reviews_df) == 0:
        return None, "No reviews available"
    
    # Get product info from first row (all rows have same product info)
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
    """Save user feedback on recommendations"""
    
    feedback_data = {
        'Shampoo Name': shampoo_name,
        'Product Hair Type': hair_type,
        'User Hair Type': user_hair_type,
        'Was Helpful': 'Yes' if was_helpful else 'No',
        'Comments': comments,
        'Feedback Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Check if feedback CSV exists
    if os.path.exists(FEEDBACK_CSV_PATH):
        feedback_df = pd.read_csv(FEEDBACK_CSV_PATH)
        new_feedback = pd.DataFrame([feedback_data])
        feedback_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
    else:
        feedback_df = pd.DataFrame([feedback_data])
    
    feedback_df.to_csv(FEEDBACK_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"Feedback saved for {shampoo_name}")
    return True


# ==================== FUNCTION 6: UPDATE PRODUCT DETAILS ====================
def update_product_details(shampoo_name, hair_type, description=None, price=None, tags=None, category=None):
    """Update description and price for existing product"""
    if category is not None and category not in VALID_CATEGORIES:
        return False, f"Category must be one of {VALID_CATEGORIES}"
    
    tags_to_store = None
    if tags is not None:
        normalized = normalize_tags_input(tags)
        tags_to_store = tags_list_to_string(normalized)
    
    product_file = get_product_reviews_filename(shampoo_name, hair_type)
    
    if not os.path.exists(product_file):
        return False, "Product not found"
    
    # Update reviews file
    reviews_df = pd.read_csv(product_file)
    reviews_df = ensure_columns(reviews_df, {'Tags': '', 'Category': ''})
    if description:
        reviews_df['Description'] = description
    if price:
        reviews_df['Price'] = price
    if tags_to_store is not None:
        reviews_df['Tags'] = tags_to_store
    if category:
        reviews_df['Category'] = category
    reviews_df.to_csv(product_file, index=False, encoding='utf-8-sig')
    
    # Update master CSV
    if os.path.exists(MASTER_CSV_PATH):
        master_df = pd.read_csv(MASTER_CSV_PATH)
        master_df = ensure_columns(master_df, {'Tags': '', 'Category': ''})
        mask = (master_df['Shampoo Name'] == shampoo_name) & (master_df['Hair Type'] == hair_type)
        if mask.any():
            if description:
                master_df.loc[mask, 'Description'] = description
            if price:
                master_df.loc[mask, 'Price'] = price
            if tags_to_store is not None:
                master_df.loc[mask, 'Tags'] = tags_to_store
            if category:
                master_df.loc[mask, 'Category'] = category
            master_df.to_csv(MASTER_CSV_PATH, index=False, encoding='utf-8-sig')
    
    return True, "Product details updated successfully"


# ==================== GUI APPLICATION ====================
class ShampooAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Shampoo Sentiment Analyzer v2.0")
        self.root.geometry("700x500")
        self.root.resizable(False, False)
        
        # Load model in background
        self.model_loaded = False
        self.available_tags = []
        self.selected_files = []
        self.load_model_thread()
        
        # Show initial menu
        self.show_main_menu()
    
    def load_model_thread(self):
        """Load model in background thread"""
        def load():
            load_model()
            self.model_loaded = True
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()

    def refresh_available_tags(self):
        """Refresh cached tag list from master CSV"""
        self.available_tags = get_all_tags_from_master()
    
    def clear_window(self):
        """Clear all widgets from window"""
        for widget in self.root.winfo_children():
            widget.destroy()
    
    def show_main_menu(self):
        """Show main menu"""
        self.clear_window()
        
        # Title
        title = tk.Label(self.root, text="Shampoo Sentiment Analyzer", 
                        font=("Arial", 20, "bold"), pady=20)
        title.pack()
        
        subtitle = tk.Label(self.root, text="AI-Powered Product Recommendation System", 
                           font=("Arial", 10), fg="gray")
        subtitle.pack()
        
        # Model loading status
        if not self.model_loaded:
            status = tk.Label(self.root, text="⏳ Loading AI model...", 
                            font=("Arial", 10), fg="orange")
            status.pack(pady=10)
        else:
            status = tk.Label(self.root, text="✓ Model loaded successfully!", 
                            font=("Arial", 10), fg="green")
            status.pack(pady=10)
        
        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=30)
        
        import_btn = tk.Button(btn_frame, text="Import Product Reviews", 
                               command=self.show_import_screen,
                               font=("Arial", 13), width=25, height=2,
                               bg="#4CAF50", fg="white", cursor="hand2")
        import_btn.pack(pady=8)
        
        top_btn = tk.Button(btn_frame, text="Get Recommendations", 
                           command=self.show_recommendations_screen,
                           font=("Arial", 13), width=25, height=2,
                           bg="#2196F3", fg="white", cursor="hand2")
        top_btn.pack(pady=8)
        
        edit_btn = tk.Button(btn_frame, text="Edit Product Details", 
                            command=self.show_edit_screen,
                            font=("Arial", 13), width=25, height=2,
                            bg="#FF9800", fg="white", cursor="hand2")
        edit_btn.pack(pady=8)
        
        # Check model loading periodically
        if not self.model_loaded:
            self.root.after(1000, self.show_main_menu)
    
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
        
        # CSV File
        tk.Label(form_frame, text="CSV File:", font=("Arial", 11)).grid(row=0, column=0, sticky="w", pady=8, padx=10)
        self.file_label = tk.Label(form_frame, text="No files selected", font=("Arial", 9), fg="gray")
        self.file_label.grid(row=0, column=1, sticky="w", pady=8)
        self.selected_files = []
        
        select_btn = tk.Button(form_frame, text="Select File", command=lambda: self.select_file(False),
                              font=("Arial", 9), bg="#E0E0E0", cursor="hand2")
        select_btn.grid(row=0, column=2, padx=10)
        
        multi_btn = tk.Button(form_frame, text="Select Multiple Files", command=lambda: self.select_file(True),
                              font=("Arial", 9), bg="#D1C4E9", cursor="hand2")
        multi_btn.grid(row=0, column=3, padx=5)
        
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
        """Open file dialog to select one or many CSV files"""
        dialog_fn = filedialog.askopenfilenames if allow_multiple else filedialog.askopenfilename
        result = dialog_fn(
            title="Select CSV file(s) with reviews",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if allow_multiple:
            filenames = list(result) if result else []
            self.selected_files = filenames
        else:
            self.selected_files = [result] if result else []

        self.update_file_label()

        if self.selected_files:
            self.autofill_product_name_from_csv(self.selected_files[0])

    def update_file_label(self):
        """Refresh the label that shows selected file(s)"""
        if not hasattr(self, 'file_label'):
            return

        if not self.selected_files:
            self.file_label.config(text="No files selected", fg="gray")
        elif len(self.selected_files) == 1:
            display_name = os.path.basename(self.selected_files[0])
            self.file_label.config(text=display_name, fg="black")
        else:
            self.file_label.config(text=f"{len(self.selected_files)} files selected", fg="black")

    def autofill_product_name_from_csv(self, filename):
        """Attempt to populate the shampoo name field using the first selected CSV"""
        try:
            temp_df = pd.read_csv(filename)
            product_col = next((col for col in temp_df.columns if col.lower() == 'product name'), None)
            
            if product_col and len(temp_df) > 0:
                product_name = str(temp_df[product_col].iloc[0])
                self.shampoo_entry.delete(0, tk.END)
                self.shampoo_entry.insert(0, product_name)
        except Exception:
            pass
    
    def process_import(self):
        """Process the import"""
        if not self.selected_files:
            messagebox.showerror("Error", "Please select a CSV file")
            return
        
        file_paths = list(self.selected_files)
        shampoo_name = self.shampoo_entry.get().strip()
        if not shampoo_name:
            messagebox.showerror("Error", "Please enter a shampoo name")
            return
        
        hair_type = self.hair_type_var.get()
        description = self.description_text.get("1.0", tk.END).strip()
        price = self.price_entry.get().strip()
        category_selection = self.category_var.get()
        tags_raw = self.tags_entry.get("1.0", tk.END).strip()
        tags_list = normalize_tags_input(tags_raw)

        product_file = get_product_reviews_filename(shampoo_name, hair_type)
        product_exists = os.path.exists(product_file)
        existing_category = None
        existing_tags = []

        if product_exists:
            try:
                existing_df = pd.read_csv(product_file)
                existing_df = ensure_columns(existing_df, {'Category': '', 'Tags': ''})
                if len(existing_df) > 0:
                    existing_category = existing_df['Category'].iloc[0] or None
                    existing_tags = parse_tags_string(existing_df['Tags'].iloc[0])
            except Exception:
                existing_category = None
                existing_tags = []

        if category_selection == "Use existing value":
            category = existing_category
        else:
            category = category_selection

        if category not in VALID_CATEGORIES:
            messagebox.showerror("Error", "Please select a valid category.")
            return

        if tags_raw and not tags_list:
            messagebox.showerror("Error", "Please enter at least one valid tag or leave the field blank.")
            return

        if not tags_list and existing_tags:
            tags_list = existing_tags

        if not product_exists and not tags_list:
            messagebox.showerror("Error", "Please provide at least one tag for new products.")
            return
        
        # Show processing
        self.root.config(cursor="wait")
        self.root.update()
        
        # Process
        csv_input = file_paths if len(file_paths) > 1 else file_paths[0]

        success, message, num_new, num_dupes = analyze_and_store_reviews(
            csv_input, shampoo_name, hair_type, 
            description if description else None,
            price if price else None,
            product_url=None,
            tags=tags_list if tags_list else None,
            category=category
        )
        
        self.root.config(cursor="")
        
        if success:
            messagebox.showinfo("Success", message)
            self.refresh_available_tags()
            self.show_main_menu()
        else:
            messagebox.showerror("Error", message)
    
    def show_recommendations_screen(self):
        """Show recommendations screen"""
        self.clear_window()
        
        # Title
        title = tk.Label(self.root, text="Get Product Recommendations", 
                        font=("Arial", 18, "bold"), pady=20)
        title.pack()
        
        # Form
        form_frame = tk.Frame(self.root)
        form_frame.pack(pady=20)
        
        tk.Label(form_frame, text="Your Hair Type:", font=("Arial", 12)).grid(row=0, column=0, sticky="w", pady=10, padx=10)
        self.rec_hair_type_var = tk.StringVar(value="straight")
        hair_dropdown = ttk.Combobox(form_frame, textvariable=self.rec_hair_type_var, 
                                     values=["straight", "wavy", "curly"],
                                     state="readonly", font=("Arial", 11), width=20)
        hair_dropdown.grid(row=0, column=1, sticky="w", pady=10)
        
        tk.Label(form_frame, text="Number of Results:", font=("Arial", 12)).grid(row=1, column=0, sticky="w", pady=10, padx=10)
        self.top_n_var = tk.StringVar(value="3")
        top_n_dropdown = ttk.Combobox(form_frame, textvariable=self.top_n_var, 
                                      values=["3", "5", "10"],
                                      state="readonly", font=("Arial", 11), width=20)
        top_n_dropdown.grid(row=1, column=1, sticky="w", pady=10)

        tk.Label(form_frame, text="Category Filter:", font=("Arial", 12)).grid(row=2, column=0, sticky="w", pady=10, padx=10)
        self.rec_category_var = tk.StringVar(value="All")
        category_options = ["All"] + VALID_CATEGORIES
        category_dropdown = ttk.Combobox(form_frame, textvariable=self.rec_category_var,
                                         values=category_options, state="readonly",
                                         font=("Arial", 11), width=20)
        category_dropdown.grid(row=2, column=1, sticky="w", pady=10)

        tk.Label(form_frame, text="Filter by Tags:", font=("Arial", 12)).grid(row=3, column=0, sticky="nw", pady=10, padx=10)
        self.refresh_available_tags()
        if self.available_tags:
            self.tags_listbox = tk.Listbox(form_frame, selectmode="multiple", height=min(6, len(self.available_tags)),
                                           exportselection=False, width=30)
            for tag in self.available_tags:
                self.tags_listbox.insert(tk.END, tag)
            self.tags_listbox.grid(row=3, column=1, sticky="w", pady=10)
            tk.Label(form_frame, text="Hold Ctrl or Shift to select multiple tags", font=("Arial", 8), fg="gray")\
                .grid(row=4, column=1, sticky="w")
        else:
            self.tags_listbox = None
            tk.Label(form_frame, text="No tags available yet. Import products to add some!",
                     font=("Arial", 9), fg="gray").grid(row=3, column=1, sticky="w", pady=10)
        
        # Get button
        get_btn = tk.Button(self.root, text="Show Recommendations", 
                           command=self.show_recommendations_results,
                           font=("Arial", 12), width=25, height=2,
                           bg="#2196F3", fg="white", cursor="hand2")
        get_btn.pack(pady=20)
        
        # Back button
        back_btn = tk.Button(self.root, text="Back to Main Menu", 
                            command=self.show_main_menu,
                            font=("Arial", 10), bg="#757575", fg="white", cursor="hand2")
        back_btn.pack()
    
    def show_recommendations_results(self):
        """Display recommendations"""
        hair_type = self.rec_hair_type_var.get()
        top_n = int(self.top_n_var.get())
        category_filter = self.rec_category_var.get() if hasattr(self, 'rec_category_var') else "All"
        selected_tags = []
        if hasattr(self, 'tags_listbox') and self.tags_listbox:
            selected_indices = self.tags_listbox.curselection()
            selected_tags = [self.available_tags[i] for i in selected_indices]
        
        category_arg = category_filter if category_filter != "All" else None
        tags_arg = selected_tags if selected_tags else None
        
        top_products, error = get_top_products(hair_type, top_n, category_arg, tags_arg)
        
        if error:
            messagebox.showerror("Error", error)
            return
        
        # Store for feedback later
        self.current_recommendations = top_products
        self.current_rec_hair_type = hair_type
        
        # Create results window
        results_window = tk.Toplevel(self.root)
        results_window.title(f"Top {top_n} Products for {hair_type.capitalize()} Hair")
        results_window.geometry("600x500")
        
        # Title
        title = tk.Label(results_window, 
                        text=f"Top {top_n} Shampoos for {hair_type.upper()} Hair",
                        font=("Arial", 14, "bold"), pady=10)
        title.pack()
        
        # Scrollable frame
        canvas = tk.Canvas(results_window)
        scrollbar = tk.Scrollbar(results_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Display products
        for idx, row in top_products.iterrows():
            product_frame = tk.Frame(scrollable_frame, relief="solid", borderwidth=1, padx=15, pady=12)
            product_frame.pack(fill="x", padx=20, pady=8)
            
            # Rank and name
            rank_label = tk.Label(product_frame, text=f"#{idx+1}", font=("Arial", 14, "bold"))
            rank_label.pack(anchor="w")
            
            name_label = tk.Label(product_frame, text=row['Shampoo Name'], font=("Arial", 12, "bold"))
            name_label.pack(anchor="w")

            category_value = row['Category'] if 'Category' in row.index else ''
            category_text = category_value if pd.notna(category_value) and category_value else "Not specified"
            category_label = tk.Label(product_frame, text=f"Category: {category_text}", font=("Arial", 9), fg="#555555")
            category_label.pack(anchor="w")
            
            # Score and reviews
            info_frame = tk.Frame(product_frame)
            info_frame.pack(anchor="w", pady=5)
            
            score_label = tk.Label(info_frame, 
                                  text=f"Sentiment Score: {row['Avg Sentiment Score']:.4f}",
                                  font=("Arial", 10), fg="green")
            score_label.pack(side="left", padx=(0, 15))
            
            reviews_label = tk.Label(info_frame, 
                                    text=f"Based on {int(row['Number of Reviews'])} reviews",
                                    font=("Arial", 9), fg="gray")
            reviews_label.pack(side="left")
            
            # View Details button
            view_btn = tk.Button(product_frame, text="View Details & Give Feedback",
                               command=lambda r=row: self.show_product_details(r, hair_type),
                               font=("Arial", 9), bg="#2196F3", fg="white", cursor="hand2")
            view_btn.pack(anchor="w", pady=5)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Close button
        close_btn = tk.Button(results_window, text="Close", 
                             command=results_window.destroy,
                             font=("Arial", 10))
        close_btn.pack(pady=10)
    
    def show_product_details(self, product_row, user_hair_type):
        """Show detailed product info with feedback option"""
        shampoo_name = product_row['Shampoo Name']
        hair_type = product_row['Hair Type']
        
        product_info, error = get_product_details(shampoo_name, hair_type)
        
        if error:
            messagebox.showerror("Error", error)
            return
        
        # Create details window
        details_window = tk.Toplevel(self.root)
        details_window.title(f"{shampoo_name} - Product Details")
        details_window.geometry("650x600")
        
        # Main container with scrollbar
        main_canvas = tk.Canvas(details_window)
        main_scrollbar = tk.Scrollbar(details_window, orient="vertical", command=main_canvas.yview)
        main_frame = tk.Frame(main_canvas)
        
        main_frame.bind("<Configure>", lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))
        main_canvas.create_window((0, 0), window=main_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=main_scrollbar.set)
        
        # Product name
        name_label = tk.Label(main_frame, text=shampoo_name, 
                             font=("Arial", 16, "bold"), pady=10)
        name_label.pack()
        
        # Hair type badge
        badge = tk.Label(main_frame, text=f"For {hair_type.upper()} Hair", 
                        font=("Arial", 10), bg="#2196F3", fg="white", padx=10, pady=5)
        badge.pack()

        category_label = tk.Label(main_frame, text=f"Category: {product_info['category'] or 'Not specified'}",
                                  font=("Arial", 10), fg="#424242")
        category_label.pack(pady=4)

        tags_frame = tk.Frame(main_frame)
        tags_frame.pack(pady=5)
        tk.Label(tags_frame, text="Tags:", font=("Arial", 11, "bold")).pack(anchor="w")
        tags_value = ", ".join(product_info['tags']) if product_info['tags'] else "No tags recorded yet."
        tk.Label(tags_frame, text=tags_value, font=("Arial", 10), wraplength=550, justify="left").pack(anchor="w")
        
        # Sentiment score
        score_frame = tk.Frame(main_frame, bg="#E8F5E9", pady=10)
        score_frame.pack(fill="x", padx=20, pady=10)
        
        score_label = tk.Label(score_frame, 
                              text=f"Average Sentiment Score: {product_info['avg_score']:.4f}",
                              font=("Arial", 12, "bold"), bg="#E8F5E9", fg="green")
        score_label.pack()
        
        reviews_count = tk.Label(score_frame,
                                text=f"Based on {product_info['num_reviews']} customer reviews",
                                font=("Arial", 9), bg="#E8F5E9", fg="gray")
        reviews_count.pack()
        
        # Price
        if product_info['price']:
            price_label = tk.Label(main_frame, text=f"Price: {product_info['price']}",
                                  font=("Arial", 11), fg="#FF6B6B")
            price_label.pack(pady=5)
        
        # Description
        if product_info['description']:
            desc_frame = tk.Frame(main_frame)
            desc_frame.pack(fill="x", padx=20, pady=10)
            
            desc_title = tk.Label(desc_frame, text="Description:", 
                                 font=("Arial", 11, "bold"))
            desc_title.pack(anchor="w")
            
            desc_text = tk.Label(desc_frame, text=product_info['description'],
                                font=("Arial", 10), wraplength=550, justify="left")
            desc_text.pack(anchor="w", pady=5)
        
        # Product URL
        if product_info['product_url']:
            url_btn = tk.Button(main_frame, text="View Product Online",
                              command=lambda: self.open_url(product_info['product_url']),
                              font=("Arial", 9), bg="#4CAF50", fg="white", cursor="hand2")
            url_btn.pack(pady=5)
        
        # Sample reviews
        sample_frame = tk.Frame(main_frame)
        sample_frame.pack(fill="x", padx=20, pady=15)
        
        sample_title = tk.Label(sample_frame, text="Sample Reviews:", 
                               font=("Arial", 11, "bold"))
        sample_title.pack(anchor="w")
        
        # Show 3 random reviews
        sample_reviews = product_info['reviews'].sample(min(3, len(product_info['reviews'])))
        
        for _, review in sample_reviews.iterrows():
            review_frame = tk.Frame(sample_frame, relief="groove", borderwidth=1, bg="#F5F5F5")
            review_frame.pack(fill="x", pady=5)
            
            review_text = tk.Label(review_frame, text=review['Comment'][:150] + "...",
                                  font=("Arial", 9), wraplength=500, justify="left",
                                  bg="#F5F5F5", padx=10, pady=8)
            review_text.pack(anchor="w")
            
            review_meta = tk.Label(review_frame, 
                                  text=f"- {review['User Name']} | Score: {review['Sentiment Score']:.3f}",
                                  font=("Arial", 8), fg="gray", bg="#F5F5F5", padx=10, pady=2)
            review_meta.pack(anchor="w")
        
        # Feedback section
        feedback_frame = tk.Frame(main_frame, bg="#FFF3E0", pady=15)
        feedback_frame.pack(fill="x", padx=20, pady=20)
        
        feedback_title = tk.Label(feedback_frame, text="Was this recommendation helpful?",
                                 font=("Arial", 12, "bold"), bg="#FFF3E0")
        feedback_title.pack(pady=5)
        
        feedback_desc = tk.Label(feedback_frame, 
                                text="Help us improve our recommendations by sharing your experience!",
                                font=("Arial", 9), bg="#FFF3E0", fg="gray")
        feedback_desc.pack()
        
        # Feedback buttons
        btn_frame = tk.Frame(feedback_frame, bg="#FFF3E0")
        btn_frame.pack(pady=10)
        
        yes_btn = tk.Button(btn_frame, text="👍 Yes, it helped!",
                          command=lambda: self.save_feedback(shampoo_name, hair_type, 
                                                            user_hair_type, True, details_window),
                          font=("Arial", 10), bg="#4CAF50", fg="white", 
                          width=15, cursor="hand2")
        yes_btn.pack(side="left", padx=10)
        
        no_btn = tk.Button(btn_frame, text="👎 No, not really",
                         command=lambda: self.show_feedback_form(shampoo_name, hair_type,
                                                                user_hair_type, False, details_window),
                         font=("Arial", 10), bg="#F44336", fg="white", 
                         width=15, cursor="hand2")
        no_btn.pack(side="left", padx=10)
        
        # Pack canvas
        main_canvas.pack(side="left", fill="both", expand=True)
        main_scrollbar.pack(side="right", fill="y")
        
        # Close button
        close_btn = tk.Button(details_window, text="Close", 
                             command=details_window.destroy,
                             font=("Arial", 10))
        close_btn.pack(side="bottom", pady=10)
    
    def open_url(self, url):
        """Open URL in browser"""
        import webbrowser
        webbrowser.open(url)
    
    def save_feedback(self, shampoo_name, hair_type, user_hair_type, was_helpful, window):
        """Save simple feedback"""
        save_recommendation_feedback(shampoo_name, hair_type, user_hair_type, was_helpful, "")
        messagebox.showinfo("Thank You!", "Your feedback has been recorded. Thank you!")
        window.destroy()
    
    def show_feedback_form(self, shampoo_name, hair_type, user_hair_type, was_helpful, parent_window):
        """Show form for detailed feedback"""
        feedback_window = tk.Toplevel(parent_window)
        feedback_window.title("Tell us more")
        feedback_window.geometry("400x300")
        
        title = tk.Label(feedback_window, text="Please tell us why:", 
                        font=("Arial", 12, "bold"), pady=10)
        title.pack()
        
        # Comments box
        comments_label = tk.Label(feedback_window, text="Your comments (optional):",
                                 font=("Arial", 10))
        comments_label.pack(pady=5)
        
        comments_text = scrolledtext.ScrolledText(feedback_window, font=("Arial", 10), 
                                                  width=45, height=8)
        comments_text.pack(padx=20, pady=10)
        
        # Submit button
        def submit_feedback():
            comments = comments_text.get("1.0", tk.END).strip()
            save_recommendation_feedback(shampoo_name, hair_type, user_hair_type, 
                                       was_helpful, comments)
            messagebox.showinfo("Thank You!", "Your feedback has been recorded. Thank you!")
            feedback_window.destroy()
            parent_window.destroy()
        
        submit_btn = tk.Button(feedback_window, text="Submit Feedback",
                             command=submit_feedback,
                             font=("Arial", 10), bg="#2196F3", fg="white", 
                             width=20, cursor="hand2")
        submit_btn.pack(pady=10)
    
    def show_edit_screen(self):
        """Show screen for editing product details"""
        self.clear_window()
        
        # Title
        title = tk.Label(self.root, text="Edit Product Details", 
                        font=("Arial", 18, "bold"), pady=15)
        title.pack()
        
        # Check if products exist
        if not os.path.exists(MASTER_CSV_PATH):
            msg = tk.Label(self.root, text="No products in database yet.\nImport some products first!",
                          font=("Arial", 12), fg="gray")
            msg.pack(pady=30)
            
            back_btn = tk.Button(self.root, text="Back to Main Menu",
                               command=self.show_main_menu,
                               font=("Arial", 10), bg="#757575", fg="white", cursor="hand2")
            back_btn.pack()
            return
        
        # Load products
        master_df = pd.read_csv(MASTER_CSV_PATH)
        product_list = [f"{row['Shampoo Name']} ({row['Hair Type']})" 
                       for _, row in master_df.iterrows()]
        
        # Form
        form_frame = tk.Frame(self.root)
        form_frame.pack(pady=20)
        
        tk.Label(form_frame, text="Select Product:", font=("Arial", 11)).grid(row=0, column=0, sticky="w", pady=10, padx=10)
        self.edit_product_var = tk.StringVar()
        product_dropdown = ttk.Combobox(form_frame, textvariable=self.edit_product_var,
                                       values=product_list, state="readonly", 
                                       font=("Arial", 10), width=35)
        product_dropdown.grid(row=0, column=1, sticky="w", pady=10)
        
        tk.Label(form_frame, text="New Description:", font=("Arial", 11)).grid(row=1, column=0, sticky="nw", pady=10, padx=10)
        self.edit_desc_text = tk.Text(form_frame, font=("Arial", 9), width=35, height=4)
        self.edit_desc_text.grid(row=1, column=1, sticky="w", pady=10)
        
        tk.Label(form_frame, text="New Price:", font=("Arial", 11)).grid(row=2, column=0, sticky="w", pady=10, padx=10)
        self.edit_price_entry = tk.Entry(form_frame, font=("Arial", 10), width=35)
        self.edit_price_entry.grid(row=2, column=1, sticky="w", pady=10)

        tk.Label(form_frame, text="Category:", font=("Arial", 11)).grid(row=3, column=0, sticky="w", pady=10, padx=10)
        self.edit_category_var = tk.StringVar(value="No change")
        edit_category_dropdown = ttk.Combobox(form_frame, textvariable=self.edit_category_var,
                                              values=["No change"] + VALID_CATEGORIES,
                                              state="readonly", font=("Arial", 10), width=35)
        edit_category_dropdown.grid(row=3, column=1, sticky="w", pady=10)

        tk.Label(form_frame, text="Tags (comma separated):", font=("Arial", 11)).grid(row=4, column=0, sticky="w", pady=10, padx=10)
        self.edit_tags_entry = tk.Entry(form_frame, font=("Arial", 10), width=35)
        self.edit_tags_entry.grid(row=4, column=1, sticky="w", pady=10)

        self.clear_tags_var = tk.BooleanVar(value=False)
        clear_tags_cb = tk.Checkbutton(form_frame, text="Clear existing tags", variable=self.clear_tags_var,
                                       font=("Arial", 9))
        clear_tags_cb.grid(row=5, column=1, sticky="w")
        
        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=20)
        
        update_btn = tk.Button(btn_frame, text="Update Details",
                             command=self.update_details,
                             font=("Arial", 11), width=18, height=2,
                             bg="#FF9800", fg="white", cursor="hand2")
        update_btn.pack(side="left", padx=10)
        
        back_btn = tk.Button(btn_frame, text="Back",
                           command=self.show_main_menu,
                           font=("Arial", 11), width=18, height=2,
                           bg="#757575", fg="white", cursor="hand2")
        back_btn.pack(side="left", padx=10)
    
    def update_details(self):
        """Update product details"""
        selected = self.edit_product_var.get()
        if not selected:
            messagebox.showerror("Error", "Please select a product")
            return
        
        # Parse product name and hair type
        # Format: "Product Name (hair_type)"
        shampoo_name = selected.rsplit(' (', 1)[0]
        hair_type = selected.rsplit('(', 1)[1].rstrip(')')
        
        description = self.edit_desc_text.get("1.0", tk.END).strip()
        price = self.edit_price_entry.get().strip()
        category_selection = self.edit_category_var.get() if hasattr(self, 'edit_category_var') else "No change"
        tags_value = self.edit_tags_entry.get().strip() if hasattr(self, 'edit_tags_entry') else ""
        tags_to_update = None
        
        if tags_value:
            tags_to_update = normalize_tags_input(tags_value)
            if not tags_to_update:
                messagebox.showerror("Error", "Please enter at least one valid tag or leave the field empty.")
                return
        elif self.clear_tags_var.get():
            tags_to_update = []
        
        category_value = None
        if category_selection and category_selection != "No change":
            category_value = category_selection
        
        if not description and not price and tags_to_update is None and category_value is None:
            messagebox.showwarning("Warning", "Please update at least one field before saving.")
            return
        
        success, message = update_product_details(shampoo_name, hair_type,
                                                 description if description else None,
                                                 price if price else None,
                                                 tags=tags_to_update,
                                                 category=category_value)
        
        if success:
            messagebox.showinfo("Success", message)
            self.refresh_available_tags()
            self.show_main_menu()
        else:
            messagebox.showerror("Error", message)


# ==================== RUN APPLICATION ====================
if __name__ == "__main__":
    root = tk.Tk()
    app = ShampooAnalyzerApp(root)
    root.mainloop()