import os
from datetime import datetime

import pandas as pd


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


# ==================== FUNCTION: GET TOP PRODUCTS ====================
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


# ==================== FUNCTION: GET PRODUCT DETAILS ====================
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


# ==================== FUNCTION: SAVE FEEDBACK ====================
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


# ==================== FUNCTION: UPDATE PRODUCT DETAILS ====================
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


# ==================== FUNCTION: RECOMMEND PRODUCTS ====================
def recommend_products(hair_type, top_n=3):
    """Get top sentiment-based recommendations for a given hair type."""
    hair_type_lower = hair_type.lower()
    top_products_df, error = get_top_products(hair_type_lower, top_n=top_n)

    if error or top_products_df is None or len(top_products_df) == 0:
        return []

    recommendations = []
    for idx, row in top_products_df.iterrows():
        product_name = str(row['Shampoo Name']).strip() if pd.notna(row.get('Shampoo Name')) else ''
        if not product_name:
            print(f"Warning: Skipping product with empty name at index {idx}")
            continue

        tags_str = row.get('Tags', '') if pd.notna(row.get('Tags', '')) else ''
        tags_list = parse_tags_string(tags_str)

        product = {
            'name': product_name,
            'hair_type': str(row['Hair Type']).strip() if pd.notna(row.get('Hair Type')) else hair_type_lower,
            'avg_sentiment_score': float(row['Avg Sentiment Score']) if pd.notna(row.get('Avg Sentiment Score')) else 0.0,
            'num_reviews': int(row['Number of Reviews']) if pd.notna(row.get('Number of Reviews')) else 0,
            'description': str(row['Description']).strip() if pd.notna(row.get('Description')) else '',
            'price': str(row['Price']).strip() if pd.notna(row.get('Price')) else '',
            'product_url': str(row['Product URL']).strip() if pd.notna(row.get('Product URL')) else '',
            'product_image': str(row['Product Image']).strip() if pd.notna(row.get('Product Image')) else '',
            'category': str(row['Category']).strip() if pd.notna(row.get('Category')) else '',
            'tags': tags_list
        }
        recommendations.append(product)

    return recommendations


# ==================== FUNCTION: GENERATE EXPLANATION ====================
def generate_explanation(hair_type, confidence, recommendations):
    """Generate a natural language explanation for the prediction and results."""
    explanation_parts = [
        f"Based on the image analysis, your hair type is predicted to be **{hair_type}** "
        f"with a confidence of {confidence:.1f}%."
    ]

    if not recommendations or len(recommendations) == 0:
        explanation_parts.append(
            "\nUnfortunately, we don't have any product recommendations available for this hair type at the moment. "
            "Please check back later as we continue to add more products to our database."
        )
    else:
        explanation_parts.append(
            f"\nHere are our top {len(recommendations)} recommended products for {hair_type.lower()} hair, "
            "ranked by customer sentiment analysis:"
        )

        for idx, product in enumerate(recommendations, 1):
            product_name = product.get('name', '').strip()
            if not product_name:
                print(f"Warning: Product at index {idx} has no name, skipping...")
                continue

            product_line = f"{idx}. **{product_name}**"
            if product.get('category'):
                product_line += f" ({product['category']})"
            explanation_parts.append(product_line)

            details = []
            if product.get('avg_sentiment_score', 0) > 0:
                details.append(f"   - Sentiment Score: {product['avg_sentiment_score']:.3f}")
            if product.get('num_reviews', 0) > 0:
                review_text = f"{product['num_reviews']} customer review{'s' if product['num_reviews'] != 1 else ''}"
                details.append(f"   - Based on {review_text}")
            if product.get('price'):
                details.append(f"   - Price: {product['price']}")
            if product.get('tags'):
                tags_str = ", ".join(product['tags'])
                details.append(f"   - Tags: {tags_str}")
            if details:
                explanation_parts.extend(details)
            explanation_parts.append("")

        explanation_parts.append(
            "\nThese recommendations are based on analyzing customer reviews and sentiment scores. "
            "Products with higher sentiment scores indicate more positive customer experiences."
        )

    return "\n".join(explanation_parts)


# ==================== API: RECOMMENDATIONS ====================
def recommend_products(hair_type, top_n=3):
    """
    Get top product recommendations for a given hair type.
    """
    hair_type_lower = hair_type.lower()

    top_products_df, error = get_top_products(hair_type_lower, top_n=top_n)

    if error or top_products_df is None or len(top_products_df) == 0:
        return []

    recommendations = []
    for idx, row in top_products_df.iterrows():
        product_name = str(row['Shampoo Name']).strip() if pd.notna(row.get('Shampoo Name')) else ''

        if not product_name:
            print(f"Warning: Skipping product with empty name at index {idx}")
            continue

        tags_str = row.get('Tags', '') if pd.notna(row.get('Tags', '')) else ''
        tags_list = parse_tags_string(tags_str)

        product = {
            'name': product_name,
            'hair_type': str(row['Hair Type']).strip() if pd.notna(row.get('Hair Type')) else hair_type_lower,
            'avg_sentiment_score': float(row['Avg Sentiment Score']) if pd.notna(row.get('Avg Sentiment Score')) else 0.0,
            'num_reviews': int(row['Number of Reviews']) if pd.notna(row.get('Number of Reviews')) else 0,
            'description': str(row['Description']).strip() if pd.notna(row.get('Description')) else '',
            'price': str(row['Price']).strip() if pd.notna(row.get('Price')) else '',
            'product_url': str(row['Product URL']).strip() if pd.notna(row.get('Product URL')) else '',
            'product_image': str(row['Product Image']).strip() if pd.notna(row.get('Product Image')) else '',
            'category': str(row['Category']).strip() if pd.notna(row.get('Category')) else '',
            'tags': tags_list
        }
        recommendations.append(product)

    return recommendations


def generate_explanation(hair_type, confidence, recommendations):
    """
    Generate a natural language explanation of the prediction and recommendations.
    """
    explanation_parts = [
        f"Based on the image analysis, your hair type is predicted to be **{hair_type}** "
        f"with a confidence of {confidence:.1f}%."
    ]

    if not recommendations or len(recommendations) == 0:
        explanation_parts.append(
            "\nUnfortunately, we don't have any product recommendations available for this hair type at the moment. "
            "Please check back later as we continue to add more products to our database."
        )
    else:
        explanation_parts.append(
            f"\nHere are our top {len(recommendations)} recommended products for {hair_type.lower()} hair, "
            "ranked by customer sentiment analysis:"
        )

        for idx, product in enumerate(recommendations, 1):
            product_name = product.get('name', '').strip()
            if not product_name:
                print(f"Warning: Product at index {idx} has no name, skipping...")
                continue

            product_line = f"{idx}. **{product_name}**"
            if product.get('category'):
                product_line += f" ({product['category']})"
            explanation_parts.append(product_line)

            details = []

            if product.get('avg_sentiment_score', 0) > 0:
                details.append(f"   - Sentiment Score: {product['avg_sentiment_score']:.3f}")

            if product.get('num_reviews', 0) > 0:
                review_text = f"{product['num_reviews']} customer review{'s' if product['num_reviews'] != 1 else ''}"
                details.append(f"   - Based on {review_text}")

            if product.get('price'):
                details.append(f"   - Price: {product['price']}")

            if product.get('tags'):
                tags_str = ", ".join(product['tags'])
                details.append(f"   - Tags: {tags_str}")

            if details:
                explanation_parts.extend(details)

            explanation_parts.append("")

        explanation_parts.append(
            "\nThese recommendations are based on analyzing customer reviews and sentiment scores. "
            "Products with higher sentiment scores indicate more positive customer experiences."
        )

    return "\n".join(explanation_parts)

