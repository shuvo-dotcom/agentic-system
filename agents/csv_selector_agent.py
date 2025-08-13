"""
CSV Selector Agent - Allows users to choose specific CSV files for analysis.
"""
import os
import pandas as pd
from typing import Any, Dict, List, Optional
from datetime import datetime
import json

from core.simple_base_agent import SimpleBaseAgent


class CSVSelectorAgent(SimpleBaseAgent):
    """
    Agent responsible for presenting available CSV files and allowing user selection.
    Provides file information and metadata to help users make informed choices.
    """
    
    def __init__(self, data_directory: str = "data/csv"):
        super().__init__(
            name="CSVSelectorAgent",
            description="Presents available CSV files and handles user selection for analysis."
        )
        self.data_directory = data_directory
        
    def get_available_csv_files(self) -> List[Dict[str, Any]]:
        """
        Scan the data directory and return information about available CSV files.
        
        Returns:
            List of dictionaries containing file information
        """
        csv_files = []
        
        if not os.path.exists(self.data_directory):
            return csv_files
            
        for filename in os.listdir(self.data_directory):
            if filename.lower().endswith('.csv'):
                file_path = os.path.join(self.data_directory, filename)
                try:
                    # Get basic file info
                    file_stats = os.stat(file_path)
                    file_size = file_stats.st_size
                    modified_time = datetime.fromtimestamp(file_stats.st_mtime)
                    
                    # Try to read CSV header and basic info
                    df = pd.read_csv(file_path, nrows=5)  # Just read first 5 rows for preview
                    
                    file_info = {
                        "filename": filename,
                        "file_path": file_path,
                        "size_bytes": file_size,
                        "size_mb": round(file_size / (1024 * 1024), 2),
                        "modified_date": modified_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "columns": list(df.columns),
                        "num_columns": len(df.columns),
                        "preview_rows": df.head(3).to_dict("records"),
                        "estimated_total_rows": self._estimate_row_count(file_path),
                        "contains_dates": self._check_date_columns(df),
                        "numerical_columns": self._get_numerical_columns(df)
                    }
                    csv_files.append(file_info)
                    
                except Exception as e:
                    # If we can't read the file, still include basic info
                    csv_files.append({
                        "filename": filename,
                        "file_path": file_path,
                        "size_bytes": file_size,
                        "size_mb": round(file_size / (1024 * 1024), 2),
                        "modified_date": modified_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "error": f"Could not read file: {str(e)}"
                    })
        
        return csv_files
    
    def _estimate_row_count(self, file_path: str) -> int:
        """Estimate total number of rows in CSV file."""
        try:
            with open(file_path, 'r') as f:
                # Count lines (rough estimate)
                line_count = sum(1 for line in f)
                return line_count - 1  # Subtract header
        except:
            return 0
    
    def _check_date_columns(self, df: pd.DataFrame) -> List[str]:
        """Check which columns might contain dates."""
        date_columns = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'period', 'month', 'year']):
                date_columns.append(col)
            else:
                # Try to parse a few values to see if they're dates
                try:
                    sample_values = df[col].dropna().head(3)
                    for val in sample_values:
                        pd.to_datetime(str(val))
                    date_columns.append(col)
                    break
                except:
                    continue
        return date_columns
    
    def _get_numerical_columns(self, df: pd.DataFrame) -> List[str]:
        """Get columns that contain numerical data."""
        numerical_cols = []
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                numerical_cols.append(col)
        return numerical_cols
    
    def present_file_options(self) -> str:
        """
        Create a formatted presentation of available CSV files for user selection.
        
        Returns:
            Formatted string showing available files
        """
        csv_files = self.get_available_csv_files()
        
        if not csv_files:
            return "No CSV files found in the data directory."
        
        presentation = "📊 Available CSV Files for Analysis:\n\n"
        
        for i, file_info in enumerate(csv_files, 1):
            presentation += f"{i}. **{file_info['filename']}**\n"
            presentation += f"   📁 Size: {file_info['size_mb']} MB\n"
            presentation += f"   📅 Modified: {file_info['modified_date']}\n"
            
            if 'error' in file_info:
                presentation += f"   ❌ Error: {file_info['error']}\n"
            else:
                presentation += f"   📊 Columns ({file_info['num_columns']}): {', '.join(file_info['columns'][:5])}"
                if len(file_info['columns']) > 5:
                    presentation += f" ... and {len(file_info['columns']) - 5} more"
                presentation += "\n"
                
                if file_info['contains_dates']:
                    presentation += f"   📅 Date columns: {', '.join(file_info['contains_dates'])}\n"
                
                if file_info['numerical_columns']:
                    presentation += f"   🔢 Numerical columns: {', '.join(file_info['numerical_columns'][:3])}"
                    if len(file_info['numerical_columns']) > 3:
                        presentation += f" ... and {len(file_info['numerical_columns']) - 3} more"
                    presentation += "\n"
                
                presentation += f"   📈 Estimated rows: ~{file_info['estimated_total_rows']:,}\n"
            
            presentation += "\n"
        
        presentation += "Please enter the number (1-{}) of the CSV file you'd like to use for analysis: ".format(len(csv_files))
        return presentation
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process CSV file selection request.
        
        Args:
            input_data: Dictionary containing selection request
            
        Returns:
            Dictionary with file information or selection results
        """
        try:
            action = input_data.get("action", "list")
            
            if action == "list":
                csv_files = self.get_available_csv_files()
                return self.create_success_response({
                    "available_files": csv_files,
                    "presentation": self.present_file_options()
                })
            
            elif action == "select":
                selection = input_data.get("selection")
                if selection is None:
                    return self.create_error_response("No selection provided")
                
                csv_files = self.get_available_csv_files()
                
                try:
                    # Handle both numeric and filename selection
                    if isinstance(selection, int) or (isinstance(selection, str) and selection.isdigit()):
                        index = int(selection) - 1
                        if 0 <= index < len(csv_files):
                            selected_file = csv_files[index]
                        else:
                            return self.create_error_response(f"Invalid selection. Please choose between 1 and {len(csv_files)}")
                    else:
                        # Try to find by filename
                        selected_file = None
                        for file_info in csv_files:
                            if file_info['filename'].lower() == selection.lower():
                                selected_file = file_info
                                break
                        
                        if selected_file is None:
                            return self.create_error_response(f"File '{selection}' not found")
                    
                    return self.create_success_response({
                        "selected_file": selected_file,
                        "message": f"Selected: {selected_file['filename']}"
                    })
                    
                except (ValueError, IndexError):
                    return self.create_error_response("Invalid selection format")
            
            else:
                return self.create_error_response(f"Unknown action: {action}")
                
        except Exception as e:
            self.logger.error(f"Error in CSV selection: {str(e)}")
            return self.create_error_response(f"CSV selection failed: {str(e)}")
