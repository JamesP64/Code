import pandas as pd
from datasets import Dataset
import os

def loadMcCance():
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, "data", "McCance_Widdowsons_Composition_of_Foods_Integrated_Dataset_2021.xlsx")

    # Load the sheets with relevant data
    target_sheets = ["1.2 Factors", "1.3 Proximates", "1.4 Inorganics", "1.5 Vitamins", "1.6 Vitamin Fractions"]
    all_sheets = pd.read_excel(path, sheet_name=target_sheets)

    clean_rows = []

    for sheet_name, df in all_sheets.items():

        # Strip column names so its consitent
        df = df.rename(columns=lambda x: x.strip())

        # Basic ID Columns
        base_cols = []
        for col in ["Food Name", "Food Code"]:
            if col in df.columns:
                base_cols.append(col)

        #   Only keep the numeric columns
        nutrient_cols = df.select_dtypes(include=["number"]).columns.tolist()
        keep_cols = base_cols + nutrient_cols

        df_clean = df[keep_cols].fillna("")

        # Convert rows to text 
        for _, row in df_clean.iterrows():
            lines = []
            for col, val in row.items():
                if val == "" or pd.isna(val):
                    continue
                lines.append(f"{col}: {val}")
            text = "\n".join(lines)
            clean_rows.append({"text": text})

    # Build dataset
    dataset = Dataset.from_list(clean_rows)
    return dataset





