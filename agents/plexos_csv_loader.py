"""
Plexos CSV Loader Agent - Handles structured data ingestion from Plexos software.
"""
import pandas as pd
import os
from typing import Any, Dict, List, Optional
from datetime import datetime
import json

from core.simple_base_agent import SimpleBaseAgent


class PlexosCSVLoader(SimpleBaseAgent):
    """
    Agent responsible for querying Plexos database and ingesting CSV data.
    Plexos is an energy market simulation software that stores data in SQLite databases.
    """
    
    def __init__(self):
        # Define tools for Plexos data operations

        
        super().__init__(
            name="PlexosCSVLoader",
            description="Loads and processes Plexos energy modeling CSV files."
        )
        



    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Plexos data ingestion request.
        
        Args:
            input_data: Dictionary containing request parameters
            
        Returns:
            Dictionary with processed data and provenance logs
        """
        try:
            self.log_activity("Starting Plexos data ingestion", input_data)
            
            file_path = input_data.get("file_path", "/home/ubuntu/agentic_system/temp_plexos_data/systemgenerators_nuclear_generation.csv")
            data_type = input_data.get("data_type", "")

            if not file_path or not os.path.exists(file_path):
                return self.create_error_response(f"CSV file not found at {file_path}")

            df = pd.read_csv(file_path)

            processed_data = {
                "file_path": file_path,
                "data_type": data_type,
                "rows": len(df),
                "columns": list(df.columns),
                "preview": df.head().to_dict("records")
            }

            return self.create_success_response(processed_data, {"provenance": "Plexos CSV file loaded"})
            
        except Exception as e:
            self.logger.error(f"Error in Plexos data ingestion: {str(e)}")
            return self.create_error_response(f"Plexos ingestion failed: {str(e)}")


