import pandas as pd
from typing import List

# Columns we want to combine for training
IMPORTANT_FIELDS = [
    "complaint_no",
    "ticket_number",
    "complaint_categories",
    "additional_comments",
    "close_feedback",
    "feedback_summary",
    "complaint_summary",
    "capa_summary",
]

def clean_text(value):
    """Safely convert NaN or None to empty string."""
    if pd.isna(value) or value is None:
        return ""
    return str(value).strip()

def row_to_document(row) -> str:
    """
    Convert a single row into a structured text document
    that will be used for embeddings.
    """

    text = f"""
Complaint Number: {clean_text(row.get("complaint_no"))}
Ticket Number: {clean_text(row.get("ticket_number"))}
Category: {clean_text(row.get("complaint_categories"))}

Comments:
{clean_text(row.get("additional_comments"))}

Close Feedback:
{clean_text(row.get("close_feedback"))}

Feedback Summary:
{clean_text(row.get("feedback_summary"))}

Complaint Summary:
{clean_text(row.get("complaint_summary"))}

CAPA Summary:
{clean_text(row.get("capa_summary"))}
"""

    return text.strip()


def preprocess_dataframe(df: pd.DataFrame) -> List[str]:
    """
    Convert the entire DataFrame into a list of cleaned documents.
    """
    documents = []

    for _, row in df.iterrows():
        doc = row_to_document(row)
        documents.append(doc)

    return documents