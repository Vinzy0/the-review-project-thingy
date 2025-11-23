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


# ==================== TEST FUNCTIONS ====================
def test_recommend_products():
    """Test the recommend_products function with various hair types"""
    print("=" * 70)
    print("TESTING: recommend_products()")
    print("=" * 70)
    
    test_cases = [
        ("Straight", 3),
        ("Wavy", 3),
        ("Curly", 3),
        ("straight", 5),  # Test lowercase
        ("STRAIGHT", 2),  # Test uppercase
        ("invalid_type", 3),  # Test invalid hair type
    ]
    
    for hair_type, top_n in test_cases:
        print(f"\n--- Testing: hair_type='{hair_type}', top_n={top_n} ---")
        try:
            recommendations = recommend_products(hair_type, top_n)
            
            if not recommendations:
                print(f"  ✓ No products found (expected for invalid types or empty database)")
            else:
                print(f"  ✓ Found {len(recommendations)} product(s):")
                for idx, product in enumerate(recommendations, 1):
                    print(f"    {idx}. {product.get('name', 'N/A')}")
                    print(f"       - Sentiment Score: {product.get('avg_sentiment_score', 0):.4f}")
                    print(f"       - Reviews: {product.get('num_reviews', 0)}")
                    print(f"       - Category: {product.get('category', 'N/A')}")
                    if product.get('tags'):
                        print(f"       - Tags: {', '.join(product['tags'])}")
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70 + "\n")


def test_generate_explanation():
    """Test the generate_explanation function with various inputs"""
    print("=" * 70)
    print("TESTING: generate_explanation()")
    print("=" * 70)
    
    # Test case 1: With recommendations
    print("\n--- Test Case 1: With recommendations ---")
    hair_type = "Straight"
    confidence = 85.5
    recommendations = recommend_products(hair_type, top_n=3)
    
    if recommendations:
        # Validate recommendations have names
        for rec in recommendations:
            if not rec.get('name') or not rec['name'].strip():
                print(f"✗ FAIL: Found recommendation with empty name: {rec}")
                return False
        
        explanation = generate_explanation(hair_type, confidence, recommendations)
        
        # Check for truncated output
        if explanation.strip().startswith("1. **") and len(explanation) < 100:
            print(f"✗ FAIL: Explanation appears truncated: '{explanation[:50]}...'")
            return False
        
        # Check that explanation contains product names
        has_product_info = False
        for rec in recommendations:
            if rec.get('name') and rec['name'] in explanation:
                has_product_info = True
                break
        
        if not has_product_info and recommendations:
            print(f"✗ FAIL: Explanation doesn't contain product names")
            print(f"Recommendations: {[r.get('name') for r in recommendations]}")
            print(f"Explanation preview: {explanation[:200]}")
            return False
        
        print("Explanation:")
        print(explanation)
        print(f"\n✓ Explanation length: {len(explanation)} characters")
    else:
        print("No recommendations available to test with. Testing with empty list...")
        explanation = generate_explanation(hair_type, confidence, [])
        print("Explanation:")
        print(explanation)
    
    # Test case 2: Empty recommendations
    print("\n--- Test Case 2: Empty recommendations ---")
    explanation = generate_explanation("Wavy", 72.3, [])
    print("Explanation:")
    print(explanation)
    
    # Test case 3: Mock recommendations
    print("\n--- Test Case 3: Mock recommendations ---")
    mock_recommendations = [
        {
            'name': 'Test Product 1',
            'category': 'Shampoo',
            'avg_sentiment_score': 0.95,
            'num_reviews': 150,
            'price': '$12.99',
            'tags': ['hydrating', 'anti-frizz']
        },
        {
            'name': 'Test Product 2',
            'category': 'Conditioner',
            'avg_sentiment_score': 0.88,
            'num_reviews': 89,
            'price': '$10.50',
            'tags': ['repair', 'smoothing']
        }
    ]
    explanation = generate_explanation("Curly", 91.2, mock_recommendations)
    
    # Validate mock recommendations output
    if explanation.strip().startswith("1. **") and len(explanation) < 100:
        print(f"✗ FAIL: Mock explanation appears truncated: '{explanation[:50]}...'")
        return False
    
    if "Test Product 1" not in explanation or "Test Product 2" not in explanation:
        print(f"✗ FAIL: Mock explanation doesn't contain product names")
        print(f"Explanation: {explanation[:300]}")
        return False
    
    print("Explanation:")
    print(explanation)
    print("✓ Mock recommendations test passed")
    
    # Test case 4: Edge cases
    print("\n--- Test Case 4: Edge cases ---")
    test_cases = [
        ("Straight", 0.0, []),  # Zero confidence
        ("Wavy", 100.0, []),    # Max confidence
        ("Curly", 50.5, None),   # None recommendations
    ]
    
    for hair_type, conf, recs in test_cases:
        try:
            explanation = generate_explanation(hair_type, conf, recs)
            print(f"\nHair Type: {hair_type}, Confidence: {conf}%, Recommendations: {recs}")
            print(f"Explanation length: {len(explanation)} characters")
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
    
    print("\n" + "=" * 70 + "\n")


def test_integration():
    """Test the full integration: recommend_products + generate_explanation"""
    print("=" * 70)
    print("TESTING: Full Integration (recommend_products + generate_explanation)")
    print("=" * 70)
    
    hair_types = ["Straight", "Wavy", "Curly"]
    
    for hair_type in hair_types:
        print(f"\n--- Testing full flow for {hair_type} hair ---")
        try:
            # Get recommendations
            recommendations = recommend_products(hair_type, top_n=3)
            
            # Generate explanation
            confidence = 85.5  # Mock confidence
            explanation = generate_explanation(hair_type, confidence, recommendations)
            
            print(f"✓ Successfully processed {hair_type} hair type")
            print(f"  - Found {len(recommendations)} recommendation(s)")
            print(f"  - Explanation generated ({len(explanation)} characters)")
            
            if recommendations:
                print(f"\n  Explanation preview:")
                # Show first 600 characters to include at least one complete product
                preview = explanation[:600] + "..." if len(explanation) > 600 else explanation
                # Indent each line
                indented_preview = '\n'.join("  " + line for line in preview.split('\n'))
                print(indented_preview)
                if len(explanation) > 600:
                    print(f"  ... (showing first 600 of {len(explanation)} characters)")
        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70 + "\n")


def run_all_tests():
    """Run all test functions"""
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS.PY TEST SUITE")
    print("=" * 70 + "\n")
    
    test_results = []
    
    try:
        test_recommend_products()
        result = test_generate_explanation()
        if result is False:
            test_results.append("FAILED")
        test_integration()
        
        print("=" * 70)
        if "FAILED" in test_results:
            print("⚠ SOME TESTS FAILED - Check output above")
        else:
            print("✓ ALL TESTS COMPLETED!")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"\n✗ CRITICAL ERROR during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return "FAILED" not in test_results


def diagnose_dataset():
    """Diagnose the dataset to check for data quality issues"""
    print("\n" + "=" * 70)
    print("DATASET DIAGNOSTIC")
    print("=" * 70 + "\n")
    
    try:
        import os
        from tooscarysoheresanewfile import MASTER_CSV_PATH
        
        if not os.path.exists(MASTER_CSV_PATH):
            print(f"✗ ERROR: Master CSV file not found at: {MASTER_CSV_PATH}")
            return False
        
        df = pd.read_csv(MASTER_CSV_PATH)
        print(f"✓ Found master CSV with {len(df)} products")
        print(f"\nColumns: {list(df.columns)}")
        
        # Check for empty product names
        if 'Shampoo Name' in df.columns:
            empty_names = df[df['Shampoo Name'].isna() | (df['Shampoo Name'].astype(str).str.strip() == '')]
            if len(empty_names) > 0:
                print(f"\n⚠ WARNING: Found {len(empty_names)} products with empty names:")
                print(empty_names[['Shampoo Name', 'Hair Type']].head())
            else:
                print(f"\n✓ All products have names")
        
        # Check hair types
        if 'Hair Type' in df.columns:
            print(f"\nHair types distribution:")
            print(df['Hair Type'].value_counts())
        
        # Sample products
        print(f"\nSample products (first 3):")
        for idx, row in df.head(3).iterrows():
            name = str(row.get('Shampoo Name', 'N/A'))
            hair_type = str(row.get('Hair Type', 'N/A'))
            score = row.get('Avg Sentiment Score', 'N/A')
            print(f"  - {name} ({hair_type}) - Score: {score}")
        
        # Test actual recommendation retrieval
        print(f"\n--- Testing actual recommendation retrieval ---")
        for hair_type in ['straight', 'wavy', 'curly']:
            recs = recommend_products(hair_type, top_n=3)
            print(f"{hair_type.capitalize()}: {len(recs)} recommendations")
            for rec in recs:
                name = rec.get('name', 'NO NAME')
                if not name or name.strip() == '':
                    print(f"  ✗ FAIL: Empty product name found!")
                else:
                    print(f"  ✓ {name}")
        
        print("\n" + "=" * 70 + "\n")
        return True
        
    except Exception as e:
        print(f"✗ ERROR during diagnosis: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run diagnostics first
    diagnose_dataset()
    
    # Then run tests
    run_all_tests()