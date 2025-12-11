# Hairoscope: Advanced Shampoo Sentiment Analyzer (ABSA Edition)

This application is a specialized NLP tool designed to analyze consumer sentiment for hair care products. Unlike standard sentiment analyzers, it employs a **Multi-Factor Filtering Pipeline** to ensure data hygiene and relevance, specifically tailored for the noisy environment of e-commerce reviews (e.g., Shopee/Lazada).

## 🚀 Key Features

### 1. Heuristic Aspect-Based Sentiment Analysis (ABSA)
The system addresses the "Delivery Bias" common in e-commerce reviews. It utilizes a **Sentence-Level Segmentation Pipeline** to:
* **Split** reviews into individual sentences.
* **Filter** out non-product feedback using a keyword blacklist (e.g., "fast delivery", "mabilis dumating", "rider").
* **Score** only the sentences related to product efficacy (scent, texture, results).

### 2. Heuristic Spam Filter (Low-Information Filter)
To combat "coin-farming" reviews, the system automatically discards inputs with low semantic value. Any review containing fewer than **3 tokens** (e.g., "ok", "good item", "nice") is rejected before analysis.

### 3. Time-Decay Weighting
Recognizing that product formulas change over time, the system implements a temporal decay algorithm. Recent reviews carry significantly more weight than older ones, ensuring recommendations reflect the current state of the product.
* **Formula:** `Weight = 0.8 ^ (Age in Years)`

### 4. Live NLP Pipeline Visualizer
A real-time demonstration tool included in the GUI allows users to type a review and watch the pipeline filter, segment, and score it step-by-step. This provides transparency into the "Black Box" of the AI model.

## 🛠️ Installation & Setup

1.  **Install Python:** Ensure you have Python 3.8 or higher installed.
2.  **Install Dependencies:**
    Run the following command in your terminal:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: This installs `pandas`, `transformers`, `torch`, and `python-dateutil`)*.

3.  **Run the Application:**
    ```bash
    python Caffine.py
    ```

## 🧠 Model Architecture

* **Base Model:** `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`
* **Why this model?** It is pre-trained on social media data, making it robust against informal grammar, slang, emojis, and code-switching (Taglish), which are prevalent in Filipino consumer reviews.

## 📂 Project Structure

* `Caffine.py`: The main application source code containing the GUI and NLP logic.
* `product_reviews/`: Directory where individual processed CSV files are stored.
* `master_sentiment_results.csv`: The master database linking all products, scores, and metadata.
* `recommendation_feedback.csv`: Logs user feedback on recommendations for future validation.

## 🖥️ How to Use

1.  **Import Reviews:** Click "Import Product Reviews" and select a CSV file containing raw Shopee/Lazada reviews.
2.  **Analyze:** The system will process the file, applying the Spam, Aspect, and Time filters automatically.
3.  **Get Recommendations:** Go to the "Get Recommendations" screen to see the top-ranked products for your specific hair type.
4.  **Visualize:** Use the "Live NLP Visualizer" to test the logic with your own text (Perfect for demonstrations!).

---
**Created for Thesis Defense: "Hairoscope: A Hybrid Recommendation System using CNN-Based Hair Classification and Aspect-Filtered Sentiment Analysis"**
