# =====================================================================
# SENTIMENT ANALYSIS USING HUGGING FACE (MULTILINGUAL SUPPORT)
# Works with English, Tagalog, and other languages.
# =====================================================================

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# =====================================================================
# 1️⃣ LOAD YOUR DATASET
# Replace the path below with your actual CSV file path.
# The CSV should contain at least one column named "feedback"
# =====================================================================
df = pd.read_csv(r"C:\Users\VINZ\Downloads\New folder (2)\processed_shampoo_data.csv")

# =====================================================================
# 2️⃣ LOAD THE MULTILINGUAL SENTIMENT MODEL
# The CardiffNLP XLM-RoBERTa model supports many languages including Tagalog.
# Using use_fast=False fixes the "SentencePiece" error you got earlier.
# =====================================================================
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"

# Load model and tokenizer (slow version for compatibility)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# =====================================================================
# 3️⃣ RUN SENTIMENT ANALYSIS ON EACH REVIEW
# This applies the model to each feedback text in your dataset.
# =====================================================================
results = []
for review in df["feedback"]:
    try:
        # Analyze the sentiment
        analysis = sentiment_analyzer(review[:512])[0]  # [:512] to avoid overly long text errors
        results.append(analysis)
    except Exception as e:
        print(f"Error analyzing review: {review}")
        print(e)
        results.append({"label": "ERROR", "score": 0.0})

# Convert the results into a DataFrame
results_df = pd.DataFrame(results)

# =====================================================================
# 4️⃣ COMBINE RESULTS WITH ORIGINAL DATA
# This attaches the sentiment label and score to the original dataset.
# =====================================================================
df_with_sentiment = pd.concat([df, results_df], axis=1)

# =====================================================================
# 5️⃣ SAVE THE RESULTS TO A NEW FILE
# You can keep adding more reviews later — just append to this file.
# =====================================================================
output_path = r"C:\Users\VINZ\Downloads\New folder (2)\sentiment_results.csv"
df_with_sentiment.to_csv(output_path, index=False, encoding="utf-8-sig")

print("✅ Sentiment analysis complete!")
print(f"Results saved to: {output_path}")

# =====================================================================
# 6️⃣ (OPTIONAL) DISPLAY AVERAGE SENTIMENT PER PRODUCT
# This lets you see which product has the most positive feedback.
# =====================================================================
avg_sentiment = df_with_sentiment.groupby("Product Name")["score"].mean().sort_values(ascending=False)
print("\n⭐ Average Sentiment Score per Product:")
print(avg_sentiment)