# sentiment_analysis using cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

df = pd.read_csv(r"C:\Users\VINZ\Downloads\New folder (2)\processed_shampoo_data.csv") # dataset goes here

model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual" # use this model for multilingual sentiment analysis

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False) # load the tokenizer 
model = AutoModelForSequenceClassification.from_pretrained(model_name) # load the model

# Create the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


# THE ACTUAL SENTIMENT ANALYSIS PART
#
# step 1: Loop through each review in the dataset
# step 2: Analyze the sentiment using the pipeline
# step 3: Store the results (label and score) in a list


results = []
for review in df["feedback"]: # step 1: loops through each review (column must be named "feedback")
    try:
        # Analyze the sentiment (step 2)
        analysis = sentiment_analyzer(review[:512])[0]  # [:512] to avoid overly long text errors
        results.append(analysis)
    except Exception as e: # error handling (just in case)
        print(f"Error analyzing review: {review}")
        print(e)
        results.append({"label": "ERROR", "score": 0.0})

# Convert the results into a DataFrame
results_df = pd.DataFrame(results)

# =====================================================================
# 
# This attaches the sentiment label and score to the original dataset.
#
df_with_sentiment = pd.concat([df, results_df], axis=1)

# =====================================================================

# SAVE THE RESULTS TO A NEW CSV FILE (so we don't have to run the analysis repeatedly)
output_path = r"C:\Users\VINZ\Downloads\New folder (2)\sentiment_results.csv"
df_with_sentiment.to_csv(output_path, index=False, encoding="utf-8-sig")

print("=============== Analysis done ================")
print(f"Results saved to: {output_path}")

# =====================================================================
# Display the sentiment score averages per product
# This lets you see which product has the most positive feedback.
# =====================================================================
avg_sentiment = df_with_sentiment.groupby("Product Name")["score"].mean().sort_values(ascending=False)
print("\n Average Sentiment Score per Product:")
print(avg_sentiment)