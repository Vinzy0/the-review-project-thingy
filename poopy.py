import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# load the dataset (small dataset for testing) (hardcoded path for now)
df = pd.read_csv(r"C:\Users\VINZ\Downloads\New folder (2)\processed_shampoo_data.csv")

# initialize nltk VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# just a check to see if the sentiment analyzer is working
test_scores = sia.polarity_scores('This shampoo is the best for curly hair!')
print("VADER Test Scores:", test_scores)
print("-" * 30)

# --------------------------------------------------------------------------------------

# apply the sentiment analyzer to the feedback column
sentiment_scores = df['feedback'].apply(sia.polarity_scores)

# turn sentiment dictionaries into columns
scores_df = pd.json_normalize(sentiment_scores)
df_with_scores = pd.concat([df, scores_df], axis=1)

# groups the reviews and their sentiment score by 'Product Name' and calculate the mean sentiment scores
avg_sentiment = df_with_scores.groupby('Product Name')[['neg', 'neu', 'pos', 'compound']].mean()

# Convert to Dictionary (Cleaner Format)
res = avg_sentiment.apply(lambda x: x.to_dict(), axis=1).to_dict()

# Print the results in a clean format
print("Aggregated Sentiment Scores (res):")
for name, scores in res.items():
    print(f"Product: {name}")
    print(f"  Neg: {scores['neg']:.4f} | Neu: {scores['neu']:.4f} | Pos: {scores['pos']:.4f} | Compound: {scores['compound']:.4f}")
print("-" * 30)