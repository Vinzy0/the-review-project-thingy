"""
API Helper Functions for Product Recommendations
This module provides functions for the Flask API to get product recommendations
and generate explanations based on hair type predictions.
"""

import pandas as pd
from tooscarysoheresanewfile import get_top_products, parse_tags_string


def recommend_products(hair_type, top_n=3):
    """
    Get top product recommendations for a given hair type.
    
    Args:
        hair_type (str): The predicted hair type (e.g., "Straight", "Wavy", "Curly")
        top_n (int): Number of top products to return (default: 3)
    
    Returns:
        list: A list of dictionaries containing product information, or empty list if no products found
    """
    # Convert hair type to lowercase to match database format
    hair_type_lower = hair_type.lower()
    
    # Get top products from the database
    top_products_df, error = get_top_products(hair_type_lower, top_n=top_n)
    
    # If error or no products found, return empty list
    if error or top_products_df is None or len(top_products_df) == 0:
        return []
    
    # Convert DataFrame to list of dictionaries
    recommendations = []
    for idx, row in top_products_df.iterrows():
        # Get product name - this is critical, must not be empty
        product_name = str(row['Shampoo Name']).strip() if pd.notna(row.get('Shampoo Name')) else ''
        
        # Skip products with empty names (data quality issue)
        if not product_name:
            print(f"Warning: Skipping product with empty name at index {idx}")
            continue
        
        # Parse tags from string to list
        tags_str = row.get('Tags', '') if pd.notna(row.get('Tags', '')) else ''
        tags_list = parse_tags_string(tags_str)
        
        # Build product dictionary
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
    
    Args:
        hair_type (str): The predicted hair type (e.g., "Straight", "Wavy", "Curly")
        confidence (float): The confidence score (0-100)
        recommendations (list): List of recommended products from recommend_products()
    
    Returns:
        str: A human-readable explanation string
    """
    # Start with prediction explanation
    explanation_parts = [
        f"Based on the image analysis, your hair type is predicted to be **{hair_type}** "
        f"with a confidence of {confidence:.1f}%."
    ]
    
    # Add recommendations section
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
            # Validate product has a name
            product_name = product.get('name', '').strip()
            if not product_name:
                print(f"Warning: Product at index {idx} has no name, skipping...")
                continue
            
            # Start with product name and category
            product_line = f"{idx}. **{product_name}**"
            if product.get('category'):
                product_line += f" ({product['category']})"
            explanation_parts.append(product_line)
            
            # Add details on separate lines with proper indentation
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
            
            # Add all details
            if details:
                explanation_parts.extend(details)
            
            # Add blank line between products for readability
            explanation_parts.append("")
        
        explanation_parts.append(
            "\nThese recommendations are based on analyzing customer reviews and sentiment scores. "
            "Products with higher sentiment scores indicate more positive customer experiences."
        )
    
    return "\n".join(explanation_parts)