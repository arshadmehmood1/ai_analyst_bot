import logging
import os
from typing import Optional
import pandas as pd

logger = logging.getLogger(__name__)

# --- Standardized Column Names (used by analyzer.py and assistant.py) ---
COMPLAINT_ID_COL = "Complaint ID"
COMPLAINT_TEXT_COL = "Complaint Text"
CATEGORY_COL = "Category"
BRANCH_CODE_COL = "Branch Code"

REQUIRED_COLUMNS = [COMPLAINT_ID_COL, COMPLAINT_TEXT_COL, CATEGORY_COL, BRANCH_CODE_COL]

# --- Mapping from Raw CSV to Standardized Names ---
COLUMN_MAPPING = {
    # Actual CSV Column Name: Standardized Name
    'complaint_no': COMPLAINT_ID_COL,
    'complaint_summary': COMPLAINT_TEXT_COL,
    'complaint_categories': CATEGORY_COL,
    # Assuming 'concerned_department' serves as the branch identifier
    'concerned_department': BRANCH_CODE_COL, 
}

def load_complaint_data(file_path: str = "data/complaints.csv") -> Optional[pd.DataFrame]:
    """
    Loads, cleans, and standardizes the complaint data from a CSV file.
    """
    if not os.path.exists(file_path):
        logger.error(f"Data file not found at: {file_path}")
        return None
        
    logger.info(f"Loading data from: {file_path}")
    
    try:
        # Load the raw data
        df = pd.read_csv(file_path)
        
        # 1. Rename columns to standardized names
        # Check for presence of required raw columns before renaming
        raw_cols_to_rename = {k: v for k, v in COLUMN_MAPPING.items() if k in df.columns}
        
        if not raw_cols_to_rename:
            logger.warning("No columns from the mapping were found in the CSV.")

        df.rename(columns=raw_cols_to_rename, inplace=True)
        logger.info(f"Renamed columns: {list(raw_cols_to_rename.keys())}")


        # 2. Check if all required standardized columns are present
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns in CSV after renaming: {missing_cols}")
            logger.error(f"Available columns now: {list(df.columns)}")
            return None

        # 3. Basic cleaning (e.g., handling missing text, converting types)
        # Ensure ID and Text columns are strings and fill NaNs in text/category columns
        df[COMPLAINT_ID_COL] = df[COMPLAINT_ID_COL].astype(str)
        
        # FIX: Replaced df[col].fillna(val, inplace=True) with direct assignment
        df[COMPLAINT_TEXT_COL] = df[COMPLAINT_TEXT_COL].fillna('')
        df[CATEGORY_COL] = df[CATEGORY_COL].fillna('Uncategorized')
        
        logger.info(f"Successfully loaded {len(df)} records. Data is ready.")
        return df

    except Exception as e:
        logger.error(f"An error occurred during data loading: {e}", exc_info=True)
        return None