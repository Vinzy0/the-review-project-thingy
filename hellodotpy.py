import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading

# ==================== CONFIGURATION ====================
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
MASTER_CSV_PATH = r"C:\Users\VINZ\Downloads\New folder (2)\master_sentiment_results.csv"

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


# ==================== FUNCTION 1: ANALYZE NEW REVIEWS ====================
def analyze_reviews(input_csv_path, shampoo_name, hair_type):
    """
    Analyzes sentiment of reviews from a CSV file.
    
    Parameters:
    - input_csv_path: Path to CSV with reviews (must have 'comment' column)
    - shampoo_name: Name of the shampoo product
    - hair_type: One of 'straight', 'wavy', or 'curly'
    
    Returns:
    - Average sentiment score
    """
    # Validate hair type
    valid_hair_types = ['straight', 'wavy', 'curly']
    if hair_type.lower() not in valid_hair_types:
        return None, f"ERROR: Hair type must be one of {valid_hair_types}"
    
    # Read the reviews
    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        return None, f"ERROR reading CSV: {str(e)}"
    
    # Check for comment column (case-insensitive)
    comment_col = None
    for col in df.columns:
        if col.lower() == 'comment':
            comment_col = col
            break
    
    if comment_col is None:
        return None, "ERROR: CSV must have a 'comment' or 'Comment' column with reviews"
    
    print(f"Analyzing {len(df)} reviews for {shampoo_name} ({hair_type} hair)...")
    
    # Run sentiment analysis
    results = []
    for review in df[comment_col]:
        try:
            analysis = sentiment_analyzer(str(review)[:512])[0]
            results.append(analysis)
        except Exception as e:
            print(f"Error analyzing review, skipping...")
            results.append({"label": "ERROR", "score": 0.0})
    
    # Calculate average score
    valid_scores = [r['score'] for r in results if r['label'] != 'ERROR']
    
    if not valid_scores:
        return None, "ERROR: No valid reviews analyzed"
    
    avg_score = sum(valid_scores) / len(valid_scores)
    print(f"Analysis complete! Average sentiment score: {avg_score:.4f}\n")
    
    return avg_score, None


# ==================== FUNCTION 2: APPEND TO MASTER CSV ====================
def append_to_master(shampoo_name, hair_type, avg_score):
    """
    Appends or updates the master CSV with sentiment results.
    """
    # Check if master CSV exists
    if os.path.exists(MASTER_CSV_PATH):
        master_df = pd.read_csv(MASTER_CSV_PATH)
    else:
        # Create new master dataframe
        master_df = pd.DataFrame(columns=['Shampoo Name', 'Hair Type', 'Avg Sentiment Score'])
    
    # Check if this product already exists
    existing = master_df[(master_df['Shampoo Name'] == shampoo_name) & 
                         (master_df['Hair Type'] == hair_type)]
    
    if len(existing) > 0:
        # Update existing entry
        master_df.loc[(master_df['Shampoo Name'] == shampoo_name) & 
                      (master_df['Hair Type'] == hair_type), 'Avg Sentiment Score'] = avg_score
        print(f"Updated existing entry for {shampoo_name} ({hair_type})")
    else:
        # Add new entry
        new_row = pd.DataFrame({
            'Shampoo Name': [shampoo_name],
            'Hair Type': [hair_type],
            'Avg Sentiment Score': [avg_score]
        })
        master_df = pd.concat([master_df, new_row], ignore_index=True)
        print(f"Added new entry for {shampoo_name} ({hair_type})")
    
    # Save master CSV
    master_df.to_csv(MASTER_CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"Master CSV updated: {MASTER_CSV_PATH}\n")


# ==================== FUNCTION 3: GET TOP PRODUCTS ====================
def get_top_products(hair_type, top_n=3):
    """
    Retrieves top N products for a specific hair type.
    """
    if not os.path.exists(MASTER_CSV_PATH):
        return None, "ERROR: Master CSV not found. Run analysis first!"
    
    master_df = pd.read_csv(MASTER_CSV_PATH)
    
    # Filter by hair type
    filtered = master_df[master_df['Hair Type'].str.lower() == hair_type.lower()]
    
    if len(filtered) == 0:
        return None, f"No products found for {hair_type} hair type"
    
    # Sort by sentiment score and get top N
    top_products = filtered.sort_values('Avg Sentiment Score', ascending=False).head(top_n)
    
    return top_products, None


# ==================== MAIN WORKFLOW ====================
def process_new_product(input_csv, shampoo_name, hair_type):
    """
    Complete workflow: analyze reviews -> append to master CSV -> delete file
    """
    print(f"\n{'='*60}")
    print(f"PROCESSING: {shampoo_name}")
    print(f"{'='*60}\n")
    
    # Step 1: Analyze reviews
    avg_score, error = analyze_reviews(input_csv, shampoo_name, hair_type)
    
    if error:
        return False, error
    
    # Step 2: Append to master CSV
    append_to_master(shampoo_name, hair_type, avg_score)
    
    # Step 3: Delete the CSV file
    try:
        os.remove(input_csv)
        print(f"Deleted file: {input_csv}")
    except Exception as e:
        print(f"Warning: Could not delete file: {str(e)}")
    
    print(f"{'='*60}")
    print("PROCESS COMPLETE!")
    print(f"{'='*60}\n")
    
    return True, f"Successfully processed {shampoo_name}!\nAvg Sentiment Score: {avg_score:.4f}"


# ==================== GUI APPLICATION ====================
class ShampooAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Shampoo Sentiment Analyzer")
        self.root.geometry("600x400")
        self.root.resizable(False, False)
        
        # Load model in background
        self.model_loaded = False
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
    
    def clear_window(self):
        """Clear all widgets from window"""
        for widget in self.root.winfo_children():
            widget.destroy()
    
    def show_main_menu(self):
        """Show main menu with two options"""
        self.clear_window()
        
        # Title
        title = tk.Label(self.root, text="Shampoo Sentiment Analyzer", 
                        font=("Arial", 20, "bold"), pady=30)
        title.pack()
        
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
        btn_frame.pack(pady=40)
        
        process_btn = tk.Button(btn_frame, text="Process New Product", 
                               command=self.show_process_screen,
                               font=("Arial", 14), width=20, height=2,
                               bg="#4CAF50", fg="white", cursor="hand2")
        process_btn.pack(pady=10)
        
        top_btn = tk.Button(btn_frame, text="Get Top Products", 
                           command=self.show_top_products_screen,
                           font=("Arial", 14), width=20, height=2,
                           bg="#2196F3", fg="white", cursor="hand2")
        top_btn.pack(pady=10)
        
        # Check model loading periodically
        if not self.model_loaded:
            self.root.after(1000, self.show_main_menu)
    
    def show_process_screen(self):
        """Show screen for processing new product"""
        if not self.model_loaded:
            messagebox.showwarning("Please Wait", "Model is still loading. Please wait a moment.")
            return
        
        self.clear_window()
        
        # Title
        title = tk.Label(self.root, text="Process New Product", 
                        font=("Arial", 18, "bold"), pady=20)
        title.pack()
        
        # Form frame
        form_frame = tk.Frame(self.root)
        form_frame.pack(pady=20)
        
        # CSV File selector
        tk.Label(form_frame, text="CSV File:", font=("Arial", 12)).grid(row=0, column=0, sticky="w", pady=10, padx=10)
        self.file_label = tk.Label(form_frame, text="No file selected", font=("Arial", 10), fg="gray")
        self.file_label.grid(row=0, column=1, sticky="w", pady=10)
        self.selected_file = None
        
        select_btn = tk.Button(form_frame, text="Select File", command=self.select_file,
                              font=("Arial", 10), bg="#E0E0E0", cursor="hand2")
        select_btn.grid(row=0, column=2, padx=10)
        
        # Shampoo name
        tk.Label(form_frame, text="Shampoo Name:", font=("Arial", 12)).grid(row=1, column=0, sticky="w", pady=10, padx=10)
        self.shampoo_entry = tk.Entry(form_frame, font=("Arial", 11), width=30)
        self.shampoo_entry.grid(row=1, column=1, columnspan=2, sticky="w", pady=10)
        
        # Hair type dropdown
        tk.Label(form_frame, text="Hair Type:", font=("Arial", 12)).grid(row=2, column=0, sticky="w", pady=10, padx=10)
        self.hair_type_var = tk.StringVar(value="straight")
        hair_dropdown = ttk.Combobox(form_frame, textvariable=self.hair_type_var, 
                                     values=["straight", "wavy", "curly"],
                                     state="readonly", font=("Arial", 11), width=28)
        hair_dropdown.grid(row=2, column=1, columnspan=2, sticky="w", pady=10)
        
        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=30)
        
        process_btn = tk.Button(btn_frame, text="Analyze & Process", 
                               command=self.process_product,
                               font=("Arial", 12), width=18, height=2,
                               bg="#4CAF50", fg="white", cursor="hand2")
        process_btn.pack(side="left", padx=10)
        
        back_btn = tk.Button(btn_frame, text="Back", 
                            command=self.show_main_menu,
                            font=("Arial", 12), width=18, height=2,
                            bg="#757575", fg="white", cursor="hand2")
        back_btn.pack(side="left", padx=10)
    
    def select_file(self):
        """Open file dialog to select CSV"""
        filename = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filename:
            self.selected_file = filename
            # Show just the filename, not full path
            display_name = os.path.basename(filename)
            self.file_label.config(text=display_name, fg="black")
            
            # Try to auto-fill product name from CSV
            try:
                temp_df = pd.read_csv(filename)
                # Look for Product Name column (case-insensitive)
                product_col = None
                for col in temp_df.columns:
                    if col.lower() == 'product name':
                        product_col = col
                        break
                
                if product_col and len(temp_df) > 0:
                    # Get the first product name (assuming all rows have same product)
                    product_name = str(temp_df[product_col].iloc[0])
                    self.shampoo_entry.delete(0, tk.END)
                    self.shampoo_entry.insert(0, product_name)
            except Exception as e:
                # If we can't read it, just leave the textbox empty
                print(f"Could not auto-fill product name: {e}")
                pass
    
    def process_product(self):
        """Process the product analysis"""
        # Validate inputs
        if not self.selected_file:
            messagebox.showerror("Error", "Please select a CSV file")
            return
        
        shampoo_name = self.shampoo_entry.get().strip()
        if not shampoo_name:
            messagebox.showerror("Error", "Please enter a shampoo name")
            return
        
        hair_type = self.hair_type_var.get()
        
        # Disable button during processing
        self.root.config(cursor="wait")
        self.root.update()
        
        # Process
        success, message = process_new_product(self.selected_file, shampoo_name, hair_type)
        
        self.root.config(cursor="")
        
        if success:
            messagebox.showinfo("Success", message)
            self.show_main_menu()
        else:
            messagebox.showerror("Error", message)
    
    def show_top_products_screen(self):
        """Show screen for getting top products"""
        self.clear_window()
        
        # Title
        title = tk.Label(self.root, text="Get Top Products", 
                        font=("Arial", 18, "bold"), pady=20)
        title.pack()
        
        # Form frame
        form_frame = tk.Frame(self.root)
        form_frame.pack(pady=20)
        
        # Hair type dropdown
        tk.Label(form_frame, text="Hair Type:", font=("Arial", 12)).grid(row=0, column=0, sticky="w", pady=10, padx=10)
        self.top_hair_type_var = tk.StringVar(value="straight")
        hair_dropdown = ttk.Combobox(form_frame, textvariable=self.top_hair_type_var, 
                                     values=["straight", "wavy", "curly"],
                                     state="readonly", font=("Arial", 11), width=20)
        hair_dropdown.grid(row=0, column=1, sticky="w", pady=10)
        
        # Number of results
        tk.Label(form_frame, text="Top N:", font=("Arial", 12)).grid(row=1, column=0, sticky="w", pady=10, padx=10)
        self.top_n_var = tk.StringVar(value="3")
        top_n_dropdown = ttk.Combobox(form_frame, textvariable=self.top_n_var, 
                                      values=["3", "5", "10"],
                                      state="readonly", font=("Arial", 11), width=20)
        top_n_dropdown.grid(row=1, column=1, sticky="w", pady=10)
        
        # Get results button
        get_btn = tk.Button(self.root, text="Get Top Products", 
                           command=self.show_top_results,
                           font=("Arial", 12), width=20, height=2,
                           bg="#2196F3", fg="white", cursor="hand2")
        get_btn.pack(pady=20)
        
        # Back button
        back_btn = tk.Button(self.root, text="Back to Main Menu", 
                            command=self.show_main_menu,
                            font=("Arial", 10), bg="#757575", fg="white", cursor="hand2")
        back_btn.pack()
    
    def show_top_results(self):
        """Display top products results"""
        hair_type = self.top_hair_type_var.get()
        top_n = int(self.top_n_var.get())
        
        top_products, error = get_top_products(hair_type, top_n)
        
        if error:
            messagebox.showerror("Error", error)
            return
        
        # Create results window
        results_window = tk.Toplevel(self.root)
        results_window.title(f"Top {top_n} Products for {hair_type.capitalize()} Hair")
        results_window.geometry("500x400")
        
        # Title
        title = tk.Label(results_window, 
                        text=f"Top {top_n} Shampoos for {hair_type.upper()} Hair",
                        font=("Arial", 14, "bold"), pady=10)
        title.pack()
        
        # Results frame with scrollbar
        canvas = tk.Canvas(results_window)
        scrollbar = tk.Scrollbar(results_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Display products
        for idx, row in top_products.iterrows():
            product_frame = tk.Frame(scrollable_frame, relief="solid", borderwidth=1, padx=10, pady=10)
            product_frame.pack(fill="x", padx=20, pady=5)
            
            rank_label = tk.Label(product_frame, text=f"#{idx+1}", font=("Arial", 12, "bold"))
            rank_label.pack(anchor="w")
            
            name_label = tk.Label(product_frame, text=row['Shampoo Name'], font=("Arial", 11))
            name_label.pack(anchor="w")
            
            score_label = tk.Label(product_frame, 
                                  text=f"Sentiment Score: {row['Avg Sentiment Score']:.4f}",
                                  font=("Arial", 10), fg="green")
            score_label.pack(anchor="w")
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Close button
        close_btn = tk.Button(results_window, text="Close", 
                             command=results_window.destroy,
                             font=("Arial", 10))
        close_btn.pack(pady=10)


# ==================== RUN APPLICATION ====================
if __name__ == "__main__":
    root = tk.Tk()
    app = ShampooAnalyzerApp(root)
    root.mainloop()