import os
import pandas as pd

BASE_DIR = r"C:\Users\VINZ\Downloads\New folder (2)"
MASTER_CSV_PATH = os.path.join(BASE_DIR, "master_sentiment_results.csv")
OUTPUT_TXT_PATH = os.path.join(BASE_DIR, "shampoo_hairtype_category.txt")


def export_shampoo_hairtype_category(
    csv_path: str = MASTER_CSV_PATH,
    output_path: str = OUTPUT_TXT_PATH,
) -> str:
    """
    Read the master sentiment CSV and write a text file that lists each shampoo
    along with its hair type and category.

    Returns the path to the generated text file.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Ensure the required columns exist before exporting.
    required_columns = ["Shampoo Name", "Hair Type", "Category"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    lines = []
    for _, row in df.iterrows():
        shampoo = str(row["Shampoo Name"]).strip()
        hair_type = str(row["Hair Type"]).strip()
        category = str(row["Category"]).strip() if "Category" in row else ""
        lines.append(f"{shampoo} | {hair_type} | {category}")

    with open(output_path, "w", encoding="utf-8") as txt_file:
        txt_file.write("\n".join(lines))

    return output_path


if __name__ == "__main__":
    path = export_shampoo_hairtype_category()
    print(f"Exported shampoo summary to: {path}")

