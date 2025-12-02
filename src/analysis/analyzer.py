"""
Enhanced ComplaintAnalyzer with correct column names for your database
"""

import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
import pandas as pd

from src.ingestion.load_data import load_complaint_data

logger = logging.getLogger(__name__)


class ComplaintAnalyzer:
    """
    Enhanced Complaint Analyzer with comprehensive analysis capabilities
    """

    def __init__(self):
        logger.info("Initializing Enhanced Complaint Analyzer...")
        self.df: Optional[pd.DataFrame] = load_complaint_data()
        
        if self.df is None or self.df.empty:
            logger.warning("Analyzer initialized, but no data was loaded.")
            self.id_lookup = {}
            self.branch_lookup = {}
        else:
            # Create lookup dictionaries - use your actual column names
            id_col = self._find_column(['complaint_no', 'Complaint ID', 'complaint_id'])
            if id_col:
                self.id_lookup = self.df.set_index(id_col).to_dict('index')
            else:
                self.id_lookup = {}
            
            branch_col = self._find_column(['Branch Code', 'branch_code', 'branch'])
            if branch_col and branch_col in self.df.columns:
                self.branch_lookup = self.df.groupby(branch_col).apply(
                    lambda x: x.to_dict('records')
                ).to_dict()
            else:
                self.branch_lookup = {}
            
            logger.info(f"Analyzer ready with {len(self.df)} records.")

    def _find_column(self, possible_names: List[str]) -> Optional[str]:
        """Find a column from a list of possible names"""
        for col in possible_names:
            if col in self.df.columns:
                return col
        return None

    def get_summary_stats(self) -> Dict[str, Any]:
        """Overall analysis summary"""
        if self.df is None or self.df.empty:
            return {
                "total_complaints": 0,
                "categories_breakdown": {},
                "status": "No data loaded."
            }
        
        total_complaints = len(self.df)
        category_col = self._find_column(['complaint_categories', 'Category', 'category'])
        category_counts = {}
        
        if category_col and category_col in self.df.columns:
            category_counts = self.df[category_col].value_counts().to_dict()
        
        logger.info(f"Generated summary stats for {total_complaints} complaints.")
        
        return {
            "total_complaints": total_complaints,
            "categories_breakdown": category_counts,
            "status": "OK"
        }

    def get_complaint_by_id(self, complaint_id: str) -> Optional[Dict[str, Any]]:
        """Fetch single complaint by ID"""
        if self.df is None or self.df.empty:
            return None
        
        id_col = self._find_column(['complaint_no', 'Complaint ID', 'complaint_id'])
        if not id_col:
            return None
        
        complaint_id = str(complaint_id)
        record = self.df[self.df[id_col].astype(str) == complaint_id]
        
        if not record.empty:
            logger.info(f"Found complaint ID: {complaint_id}")
            return record.iloc[0].to_dict()
        
        logger.warning(f"Complaint ID not found: {complaint_id}")
        return None

    def get_complaints_by_branch(self, branch_code: str) -> List[Dict[str, Any]]:
        """Fetch all complaints for a branch"""
        if self.df is None or self.df.empty:
            return []
        
        branch_col = self._find_column(['Branch Code', 'branch_code', 'branch'])
        if not branch_col or branch_col not in self.df.columns:
            return []
        
        branch_code = str(branch_code).upper()
        records = self.df[self.df[branch_col].astype(str).str.upper() == branch_code].to_dict('records')
        
        logger.info(f"Found {len(records)} complaints for branch: {branch_code}")
        return records

    def get_complaints_by_status(self, status_value: str) -> List[Dict[str, Any]]:
        """Fetch complaints by status"""
        if self.df is None or self.df.empty:
            return []
        
        status_col = self._find_column(['status', 'Status'])
        if not status_col or status_col not in self.df.columns:
            return []
        
        filtered_df = self.df[self.df[status_col].astype(str).str.lower() == status_value.lower()]
        records = filtered_df.to_dict('records')
        
        logger.info(f"Found {len(records)} complaints with status: {status_value}")
        return records

    def filter_complaints(
        self, 
        category: Optional[str] = None, 
        columns: Optional[Set[str]] = None,
        phone: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Advanced filtering by category, phone, status with column projection"""
        if self.df is None or self.df.empty:
            return []

        filtered_df = self.df.copy()

        # Filter by category
        if category:
            cat_col = self._find_column(['complaint_categories', 'Category', 'category'])
            if cat_col and cat_col in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df[cat_col].astype(str).str.lower() == category.lower()
                ]
                logger.info(f"Filtered to {len(filtered_df)} records for category: {category}")

        # Filter by phone
        if phone:
            phone_col = self._find_column(['mobile_number', 'Phone', 'phone', 'phone_number'])
            if phone_col and phone_col in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df[phone_col].astype(str).str.contains(phone, na=False)
                ]
                logger.info(f"Filtered to {len(filtered_df)} records for phone: {phone}")

        # Filter by status
        if status:
            status_col = self._find_column(['status', 'Status'])
            if status_col and status_col in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df[status_col].astype(str).str.lower() == status.lower()
                ]
                logger.info(f"Filtered to {len(filtered_df)} records for status: {status}")

        # Column projection
        if columns:
            valid_columns = [col for col in columns if col in filtered_df.columns]
            if valid_columns:
                filtered_df = filtered_df[valid_columns]
                logger.info(f"Projecting columns: {valid_columns}")
            
        records = filtered_df.to_dict('records')
        return records

    def get_complaints_by_date_range(
        self, 
        start_date: datetime, 
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Fetch complaints within a date range"""
        if self.df is None or self.df.empty:
            logger.warning("DataFrame is empty")
            return []

        # Try to find date column
        date_col = self._find_column(['date_entry', 'date_of_issue', 'Entry Date', 'Date', 'entry_date', 'Date Entry'])
        if not date_col:
            logger.warning(f"Date column not found. Available columns: {list(self.df.columns)}")
            return []

        if date_col not in self.df.columns:
            logger.warning(f"Column {date_col} not in dataframe")
            return []

        try:
            df_copy = self.df.copy()
            
            # Convert to datetime with multiple format attempts
            df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce', infer_datetime_format=True)
            
            # Debug logging
            logger.info(f"Using date column: {date_col}")
            logger.info(f"Sample dates: {df_copy[date_col].head()}")
            logger.info(f"Date range filter: {start_date} to {end_date}")
            
            # Filter by date range
            mask = (df_copy[date_col] >= start_date) & (df_copy[date_col] <= end_date)
            filtered = df_copy[mask]
            
            logger.info(f"Found {len(filtered)} complaints between {start_date} and {end_date}")
            records = filtered.to_dict('records')
            return records
            
        except Exception as e:
            logger.error(f"Error filtering by date range: {e}")
            return []

    def get_complaints_by_assignee(self, assignee_name: str) -> List[Dict[str, Any]]:
        """Fetch complaints assigned to a specific person"""
        if self.df is None or self.df.empty:
            return []

        assignee_col = self._find_column(['assigned_officer', 'Assignee', 'assignee', 'person_issue'])
        if not assignee_col or assignee_col not in self.df.columns:
            logger.warning("Assignee column not found")
            return []

        filtered = self.df[
            self.df[assignee_col].astype(str).str.lower().str.contains(
                assignee_name.lower(), na=False
            )
        ]
        
        records = filtered.to_dict('records')
        logger.info(f"Found {len(records)} complaints for assignee: {assignee_name}")
        return records

    def calculate_resolution_time(self, row: pd.Series) -> Optional[float]:
        """Calculate resolution time in days for a complaint"""
        try:
            # Try to find start date
            start_col = self._find_column(['date_of_issue', 'date_entry', 'Entry Date', 'Date Entry'])
            # Try to find end date
            end_col = self._find_column(['completed_date', 'closed_date', 'rca_date', 'capa_date', 'Resolved Date'])
            
            if not start_col or not end_col:
                return None
            
            start_val = row.get(start_col)
            end_val = row.get(end_col)
            
            if pd.isna(start_val) or pd.isna(end_val):
                return None
                
            entry = pd.to_datetime(start_val)
            resolved = pd.to_datetime(end_val)
            
            if pd.isna(entry) or pd.isna(resolved):
                return None
            
            return (resolved - entry).days
        except Exception as e:
            logger.debug(f"Error calculating resolution time: {e}")
            return None

    def get_overtime_complaints(self, sla_days: int = 30) -> List[Dict[str, Any]]:
        """Get all complaints that exceeded SLA"""
        if self.df is None or self.df.empty:
            return []

        overtime_complaints = []
        
        for _, row in self.df.iterrows():
            resolution_time = self.calculate_resolution_time(row)
            
            if resolution_time is not None and resolution_time > sla_days:
                complaint = row.to_dict()
                complaint['resolution_time_days'] = resolution_time
                complaint['sla_violation_days'] = resolution_time - sla_days
                overtime_complaints.append(complaint)
        
        logger.info(f"Found {len(overtime_complaints)} overtime complaints (SLA: {sla_days} days)")
        return overtime_complaints

    def get_resolution_time_statistics(self) -> Dict[str, Any]:
        """Calculate resolution time statistics"""
        if self.df is None or self.df.empty:
            return {
                "average_resolution_time": "N/A",
                "min_resolution_time": "N/A",
                "max_resolution_time": "N/A",
                "median_resolution_time": "N/A",
                "total_resolved": 0
            }

        resolution_times = []
        
        for _, row in self.df.iterrows():
            rt = self.calculate_resolution_time(row)
            if rt is not None and rt > 0:
                resolution_times.append(rt)
        
        if not resolution_times:
            return {
                "average_resolution_time": "N/A",
                "min_resolution_time": "N/A",
                "max_resolution_time": "N/A",
                "median_resolution_time": "N/A",
                "total_resolved": 0
            }

        sorted_times = sorted(resolution_times)
        return {
            "average_resolution_time": f"{sum(resolution_times) / len(resolution_times):.1f} days",
            "min_resolution_time": f"{min(resolution_times):.1f} days",
            "max_resolution_time": f"{max(resolution_times):.1f} days",
            "median_resolution_time": f"{sorted_times[len(sorted_times)//2]:.1f} days",
            "total_resolved": len(resolution_times)
        }

    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics including counts, averages, breakdowns"""
        if self.df is None or self.df.empty:
            return {
                "total_complaints": 0,
                "open_complaints": 0,
                "overtime_complaints": 0,
                "average_resolution_time": "N/A",
                "min_resolution_time": "N/A",
                "max_resolution_time": "N/A",
                "median_resolution_time": "N/A",
                "by_category": {},
                "by_status": {}
            }

        total = len(self.df)
        
        # Status breakdown
        status_col = self._find_column(['status', 'Status'])
        by_status = {}
        open_count = 0
        
        if status_col and status_col in self.df.columns:
            by_status = self.df[status_col].value_counts().to_dict()
            open_count = by_status.get("Open", 0) + by_status.get("open", 0) + \
                        by_status.get("Pending", 0) + by_status.get("pending", 0)
        
        # Category breakdown
        cat_col = self._find_column(['complaint_categories', 'Category', 'category'])
        by_category = {}
        
        if cat_col and cat_col in self.df.columns:
            by_category = self.df[cat_col].value_counts().to_dict()
        
        # Resolution time statistics
        resolution_times = []
        for _, row in self.df.iterrows():
            rt = self.calculate_resolution_time(row)
            if rt is not None and rt > 0:
                resolution_times.append(rt)
        
        avg_res = "N/A"
        min_res = "N/A"
        max_res = "N/A"
        med_res = "N/A"
        
        if resolution_times:
            sorted_times = sorted(resolution_times)
            avg_res = f"{sum(resolution_times) / len(resolution_times):.1f} days"
            min_res = f"{min(resolution_times):.1f} days"
            max_res = f"{max(resolution_times):.1f} days"
            med_res = f"{sorted_times[len(sorted_times)//2]:.1f} days"
        
        # Overtime count
        overtime_count = len(self.get_overtime_complaints())
        
        return {
            "total_complaints": total,
            "open_complaints": open_count,
            "overtime_complaints": overtime_count,
            "average_resolution_time": avg_res,
            "min_resolution_time": min_res,
            "max_resolution_time": max_res,
            "median_resolution_time": med_res,
            "by_category": by_category,
            "by_status": by_status
        }


# Singleton Pattern
_analyzer_instance: Optional[ComplaintAnalyzer] = None

def get_analyzer() -> ComplaintAnalyzer:
    """Returns a cached singleton instance of the ComplaintAnalyzer."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = ComplaintAnalyzer()
    return _analyzer_instance