"""
Plexus CSV Loader Agent - Handles structured data ingestion from Plexus API.
"""
import pandas as pd
import requests
from typing import Any, Dict, List, Optional
from datetime import datetime
import io

from core.base_agent import BaseAgent
from config.settings import PLEXUS_API_URL, PLEXUS_API_KEY, TIMEOUT


class PlexusCSVLoader(BaseAgent):
    """
    Agent responsible for querying Plexus API and ingesting CSV data.
    Normalizes data into clean tables and logs provenance information.
    """
    
    def __init__(self):
        super().__init__(
            name="PlexusCSVLoader",
            description="Queries Plexus API, ingests and normalizes scenario CSVs with provenance logging."
        )
        self.api_url = PLEXUS_API_URL
        self.api_key = PLEXUS_API_KEY
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Plexus API data ingestion request.
        
        Args:
            input_data: Dictionary containing 'csv_files' or 'query_params'
            
        Returns:
            Dictionary with clean tables and provenance logs
        """
        try:
            if not self.api_url:
                return self.create_error_response("Plexus API URL not configured")
            
            csv_files = input_data.get("csv_files", [])
            query_params = input_data.get("query_params", {})
            
            if not csv_files and not query_params:
                return self.create_error_response("Either csv_files or query_params must be provided")
            
            self.log_activity("Starting Plexus data ingestion", {
                "csv_files": csv_files,
                "query_params": query_params
            })
            
            results = []
            provenance_logs = []
            
            # Process specific CSV files if provided
            if csv_files:
                for csv_file in csv_files:
                    result = await self._ingest_csv_file(csv_file)
                    if result["success"]:
                        results.append(result["data"])
                        provenance_logs.extend(result["metadata"]["provenance"])
                    else:
                        self.logger.warning(f"Failed to ingest {csv_file}: {result['error']['message']}")
            
            # Process query-based data retrieval
            if query_params:
                result = await self._query_plexus_data(query_params)
                if result["success"]:
                    results.extend(result["data"])
                    provenance_logs.extend(result["metadata"]["provenance"])
                else:
                    self.logger.warning(f"Query failed: {result['error']['message']}")
            
            if not results:
                return self.create_error_response("No data successfully ingested")
            
            # Combine and normalize all data
            normalized_data = self._normalize_data(results)
            
            self.log_activity("Plexus data ingestion completed", {
                "tables_processed": len(results),
                "total_rows": sum(len(df) for df in normalized_data)
            })
            
            return self.create_success_response(
                normalized_data,
                {
                    "provenance": provenance_logs,
                    "processing_stats": {
                        "tables_processed": len(results),
                        "total_rows": sum(len(df) for df in normalized_data),
                        "processing_time": datetime.now().isoformat()
                    }
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in Plexus data ingestion: {str(e)}")
            return self.create_error_response(f"Plexus ingestion failed: {str(e)}")
    
    async def _ingest_csv_file(self, csv_file: str) -> Dict[str, Any]:
        """
        Ingest a specific CSV file from Plexus API.
        """
        try:
            url = f"{self.api_url}/csv/{csv_file}"
            
            self.log_activity(f"Fetching CSV file: {csv_file}")
            
            response = self.session.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            
            # Parse CSV data
            csv_data = io.StringIO(response.text)
            df = pd.read_csv(csv_data)
            
            # Create provenance log
            provenance = {
                "source": "plexus_api",
                "file": csv_file,
                "url": url,
                "timestamp": datetime.now().isoformat(),
                "rows": len(df),
                "columns": list(df.columns),
                "data_types": df.dtypes.to_dict()
            }
            
            return self.create_success_response(
                df,
                {"provenance": [provenance]}
            )
            
        except requests.RequestException as e:
            return self.create_error_response(f"API request failed for {csv_file}: {str(e)}")
        except pd.errors.ParserError as e:
            return self.create_error_response(f"CSV parsing failed for {csv_file}: {str(e)}")
        except Exception as e:
            return self.create_error_response(f"Unexpected error for {csv_file}: {str(e)}")
    
    async def _query_plexus_data(self, query_params: Dict) -> Dict[str, Any]:
        """
        Query Plexus API with specific parameters.
        """
        try:
            url = f"{self.api_url}/query"
            
            self.log_activity("Querying Plexus API", {"params": query_params})
            
            response = self.session.get(url, params=query_params, timeout=TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            results = []
            provenance_logs = []
            
            # Process each dataset in the response
            for dataset_name, dataset_data in data.items():
                if isinstance(dataset_data, list):
                    df = pd.DataFrame(dataset_data)
                elif isinstance(dataset_data, dict) and 'data' in dataset_data:
                    df = pd.DataFrame(dataset_data['data'])
                else:
                    continue
                
                results.append(df)
                
                # Create provenance log
                provenance = {
                    "source": "plexus_api",
                    "dataset": dataset_name,
                    "query_params": query_params,
                    "url": url,
                    "timestamp": datetime.now().isoformat(),
                    "rows": len(df),
                    "columns": list(df.columns)
                }
                provenance_logs.append(provenance)
            
            return self.create_success_response(
                results,
                {"provenance": provenance_logs}
            )
            
        except requests.RequestException as e:
            return self.create_error_response(f"API query failed: {str(e)}")
        except Exception as e:
            return self.create_error_response(f"Query processing failed: {str(e)}")
    
    def _normalize_data(self, dataframes: List[pd.DataFrame]) -> List[pd.DataFrame]:
        """
        Normalize and clean the ingested data.
        """
        normalized = []
        
        for df in dataframes:
            # Create a copy to avoid modifying original
            clean_df = df.copy()
            
            # Basic cleaning operations
            # Remove completely empty rows and columns
            clean_df = clean_df.dropna(how='all').dropna(axis=1, how='all')
            
            # Standardize column names (lowercase, replace spaces with underscores)
            clean_df.columns = [
                col.lower().replace(' ', '_').replace('-', '_') 
                for col in clean_df.columns
            ]
            
            # Convert numeric columns
            for col in clean_df.columns:
                if clean_df[col].dtype == 'object':
                    # Try to convert to numeric
                    numeric_series = pd.to_numeric(clean_df[col], errors='coerce')
                    if not numeric_series.isna().all():
                        clean_df[col] = numeric_series
            
            # Handle datetime columns
            for col in clean_df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        clean_df[col] = pd.to_datetime(clean_df[col], errors='coerce')
                    except:
                        pass
            
            normalized.append(clean_df)
        
        return normalized
    
    def get_available_datasets(self) -> Dict[str, Any]:
        """
        Get list of available datasets from Plexus API.
        """
        try:
            if not self.api_url:
                return {"error": "API URL not configured"}
            
            url = f"{self.api_url}/datasets"
            response = self.session.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Error fetching available datasets: {str(e)}")
            return {"error": str(e)}

