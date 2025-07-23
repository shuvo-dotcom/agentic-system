"""
Time Detection Module

Dynamically detects time periods and forecasting needs in natural language queries.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import calendar
import json
import os

class TimeGranularity(Enum):
    """Time granularity for forecasting."""
    YEARS = "years"
    MONTHS = "months"
    DAYS = "days"
    HOURS = "hours"

@dataclass
class TimePeriod:
    """Represents a time period for forecasting."""
    value: int
    granularity: TimeGranularity
    is_forecast_requested: bool = False
    confidence: float = 0.0

@dataclass
class ForecastingRequest:
    """Represents a forecasting request with user preferences."""
    time_period: TimePeriod
    parameters: Dict[str, float]
    calculation_type: str
    user_preferences: Dict[str, str] = None

class TimeDetector:
    """Dynamically detects time periods and forecasting needs in queries."""
    
    def __init__(self):
        # Time period patterns
        self.time_patterns = {
            TimeGranularity.YEARS: [
                r'(\d+)\s*years?',
                r'next\s+(\d+)\s*years?',
                r'for\s+(\d+)\s*years?',
                r'over\s+(\d+)\s*years?',
                r'(\d+)\s*yr',
                r'(\d+)\s*y'
            ],
            TimeGranularity.MONTHS: [
                r'(\d+)\s*months?',
                r'next\s+(\d+)\s*months?',
                r'for\s+(\d+)\s*months?',
                r'over\s+(\d+)\s*months?',
                r'(\d+)\s*mo'
            ],
            TimeGranularity.DAYS: [
                r'(\d+)\s*days?',
                r'next\s+(\d+)\s*days?',
                r'for\s+(\d+)\s*days?',
                r'over\s+(\d+)\s*days?'
            ]
        }
        
        # Forecasting indicators
        self.forecast_indicators = [
            r'forecast',
            r'projection',
            r'trend',
            r'over time',
            r'variation',
            r'change',
            r'evolution',
            r'progression',
            r'future',
            r'upcoming',
            r'next',
            r'following'
        ]
        
        # Calculation types
        self.calculation_types = [
            'lcoe', 'levelized cost of energy',
            'npv', 'net present value',
            'irr', 'internal rate of return',
            'payback', 'payback period',
            'capacity factor', 'cf',
            'roi', 'return on investment',
            'cost', 'price', 'revenue'
        ]

        # Load date range patterns dynamically from config
        config_path = os.path.join(os.path.dirname(__file__), 'date_range_patterns.json')
        self.date_range_patterns = self._load_date_range_patterns(config_path)

    def _load_date_range_patterns(self, config_path):
        if not os.path.exists(config_path):
            # Default patterns if config missing
            return [
                (r"monthly from ([a-zA-Z]+) (\d{4}) to ([a-zA-Z]+) (\d{4})", TimeGranularity.MONTHS),
                (r"annually from (\d{4}) to (\d{4})", TimeGranularity.YEARS),
                (r"from (\d{4}) to (\d{4})", TimeGranularity.YEARS),
                (r"monthly from (\d{4}) to (\d{4})", TimeGranularity.MONTHS),
                (r"for each month from ([a-zA-Z]+) (\d{4}) to ([a-zA-Z]+) (\d{4})", TimeGranularity.MONTHS),
                (r"for each year from (\d{4}) to (\d{4})", TimeGranularity.YEARS)
            ]
        with open(config_path, 'r') as f:
            patterns = json.load(f)
        result = []
        for entry in patterns:
            granularity = getattr(TimeGranularity, entry['granularity'].upper())
            result.append((entry['pattern'], granularity))
        return result

    def detect_time_period(self, query: str) -> Optional[TimePeriod]:
        """
        Detect time period in the query.
        """
        # Post-process: strip quotes, extract quoted string, remove common intros
        query_clean = query.strip().strip('"').strip("'")
        # If the query contains a quoted string, use that for pattern matching
        import re
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', query_clean)
        if quoted:
            # quoted is a list of tuples, pick the first non-empty group
            for tup in quoted:
                for s in tup:
                    if s:
                        query_clean = s
                        break
        # Remove common intros
        for intro in [
            "can you ", "analyze ", "please ", "that sounds like ", "that revised prompt looks good to me! ",
            "revised prompt: ", "strict: ", "what is ", "what are ", "could you ", "i think ", "let's ", "let us ", "i'd like to ", "i want to ", "i am interested in ", "i would like to ", "do you know ", "tell me ", "give me ", "find ", "show me "
        ]:
            if query_clean.lower().startswith(intro):
                query_clean = query_clean[len(intro):]
                break
        query_lower = query_clean.lower()
        
        # Check for explicit time periods
        for granularity, patterns in self.time_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, query_lower)
                if match:
                    value = int(match.group(1))
                    confidence = self._calculate_confidence(query_lower, pattern)
                    return TimePeriod(
                        value=value,
                        granularity=granularity,
                        is_forecast_requested=True,
                        confidence=confidence
                    )
        
        # Check for forecasting indicators without explicit time
        if self._has_forecasting_indicators(query_lower):
            return TimePeriod(
                value=0,  # Will be asked from user
                granularity=TimeGranularity.YEARS,
                is_forecast_requested=True,
                confidence=0.7
            )
        
        # Check for explicit date range patterns
        for pattern, granularity in self.date_range_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if granularity == TimeGranularity.MONTHS:
                    # monthly from January 2020 to December 2022
                    if len(match.groups()) == 4:
                        start_month, start_year, end_month, end_year = match.groups()
                        try:
                            start_month_num = list(calendar.month_name).index(start_month.capitalize())
                            end_month_num = list(calendar.month_name).index(end_month.capitalize())
                            start_year = int(start_year)
                            end_year = int(end_year)
                            months = (end_year - start_year) * 12 + (end_month_num - start_month_num) + 1
                            return TimePeriod(
                                value=months,
                                granularity=granularity,
                                is_forecast_requested=True,
                                confidence=0.95
                            )
                        except Exception:
                            continue
                    # monthly from 2020 to 2022
                    elif len(match.groups()) == 2:
                        start_year, end_year = map(int, match.groups())
                        months = (end_year - start_year + 1) * 12
                        return TimePeriod(
                            value=months,
                            granularity=granularity,
                            is_forecast_requested=True,
                            confidence=0.95
                        )
                elif granularity == TimeGranularity.YEARS:
                    # annually from 2015 to 2020 or from 2015 to 2020
                    start_year, end_year = map(int, match.groups()[-2:])
                    years = end_year - start_year + 1
                    return TimePeriod(
                        value=years,
                        granularity=granularity,
                        is_forecast_requested=True,
                        confidence=0.95
            )
        
        return None

    def _has_forecasting_indicators(self, query: str) -> bool:
        """Check if query contains forecasting indicators."""
        for indicator in self.forecast_indicators:
            if indicator in query:
                return True
        return False

    def _calculate_confidence(self, query: str, pattern: str) -> float:
        """Calculate confidence score for time period detection."""
        # Higher confidence for explicit time periods
        if 'next' in pattern or 'for' in pattern or 'over' in pattern:
            return 0.95
        elif re.search(pattern, query):
            return 0.85
        return 0.7

    def extract_calculation_type(self, query: str) -> str:
        """
        Extract the type of calculation from the query.
        
        Args:
            query: User query
            
        Returns:
            Calculation type
        """
        query_lower = query.lower()
        
        for calc_type in self.calculation_types:
            if calc_type in query_lower:
                return calc_type
        
        return "calculation"  # Default

    def needs_user_input(self, time_period: TimePeriod) -> bool:
        """
        Check if user input is needed for forecasting preferences.
        
        Args:
            time_period: Detected time period
            
        Returns:
            True if user input is needed
        """
        return time_period.value == 0 or time_period.confidence < 0.8

    def generate_forecasting_questions(self, 
                                     time_period: TimePeriod,
                                     calculation_type: str,
                                     detected_parameters: Dict[str, float]) -> List[str]:
        """
        Generate questions to ask the user for forecasting preferences.
        
        Args:
            time_period: Detected time period
            calculation_type: Type of calculation
            detected_parameters: Parameters found in query
            
        Returns:
            List of questions to ask user
        """
        questions = []
        
        # Ask for time period if not specified
        if not time_period or time_period.value == 0:
            questions.append("How many years would you like to forecast? (e.g., 5, 10, 20)")
        
        # Ask for time granularity
        questions.append("What time granularity would you prefer? (years/months/days)")
        
        # Ask about parameter variations
        if detected_parameters:
            questions.append("Would you like me to model parameter variations over time? (yes/no)")
            questions.append("Are there any specific parameter trends you'd like to consider? (e.g., 'capital costs decreasing', 'energy prices increasing')")
        
        # Ask about calculation preferences
        questions.append("Do you want to see year-by-year results or summary statistics?")
        
        return questions

    def parse_user_preferences(self, 
                              user_responses: Dict[str, str],
                              original_query: str) -> ForecastingRequest:
        """
        Parse user responses into a forecasting request.
        
        Args:
            user_responses: User's responses to questions
            original_query: Original user query
            
        Returns:
            ForecastingRequest object
        """
        # Extract time period
        time_value = int(user_responses.get('time_period', '10'))
        granularity_str = user_responses.get('granularity', 'years').lower()
        
        granularity_map = {
            'years': TimeGranularity.YEARS,
            'months': TimeGranularity.MONTHS,
            'days': TimeGranularity.DAYS
        }
        
        granularity = granularity_map.get(granularity_str, TimeGranularity.YEARS)
        
        time_period = TimePeriod(
            value=time_value,
            granularity=granularity,
            is_forecast_requested=True,
            confidence=0.95
        )
        
        # Extract calculation type
        calculation_type = self.extract_calculation_type(original_query)
        
        # Extract parameters (will be done by parameter extractor)
        parameters = {}
        
        # Extract user preferences
        user_preferences = {
            'model_variations': user_responses.get('model_variations', 'yes').lower() == 'yes',
            'parameter_trends': user_responses.get('parameter_trends', ''),
            'output_format': user_responses.get('output_format', 'year_by_year')
        }
        
        return ForecastingRequest(
            time_period=time_period,
            parameters=parameters,
            calculation_type=calculation_type,
            user_preferences=user_preferences
        ) 