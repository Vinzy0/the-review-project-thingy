import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()

test_scores = sia.polarity_scores('This shampoo is the best for curly hair!')
print(test_scores)
df = pd.read_csv(r"C:\Users\VINZ\Downloads\New folder (2)\processed_shampoo_data.csv")
print(df.head())

# Run the polarity score on the entire dataset
res = {}
for i, row in df.iterrows(): 
    Name = row['Product Name'] 
    feedback = row['feedback'] 
    res[Name] = sia.polarity_scores(feedback)

print(res)