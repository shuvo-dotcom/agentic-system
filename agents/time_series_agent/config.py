"""
Time Series Agent Configuration

Configuration settings for the Time Series Analysis Agent.
"""

from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class TimeSeriesConfig:
    """Configuration for the Time Series Agent."""
    
    # Default variation rates
    DEFAULT_VARIATION_RATES = {
        'capital_cost': -0.05,      # 5% annual decrease (learning curve)
        'om_cost': 0.02,            # 2% annual increase (inflation)
        'energy_production': -0.005, # 0.5% annual degradation
        'discount_rate': 0.0,       # Constant
        'fuel_cost': 0.03,          # 3% annual increase
        'electricity_price': 0.025, # 2.5% annual increase
        'maintenance_cost': 0.015,  # 1.5% annual increase
        'insurance_cost': 0.02,     # 2% annual increase
        'tax_rate': 0.0,           # Constant
        'depreciation_rate': 0.0,   # Constant
    }
    
    # Maximum values for validation
    MAX_VARIATION_RATE = 0.5        # 50% annual change
    MAX_PARAMETER_VALUE = 1e9       # $1 billion
    MIN_PARAMETER_VALUE = -1e6      # -$1 million (for some parameters)
    
    # Time period limits
    MAX_YEARS = 50
    MAX_MONTHS = 600  # 50 years * 12 months
    MAX_QUARTERS = 200  # 50 years * 4 quarters
    
    # Execution settings
    MAX_EXECUTION_TIME = 300  # 5 minutes
    MAX_SCENARIOS = 100       # Maximum scenarios to generate
    PARALLEL_EXECUTION = True  # Enable parallel scenario execution
    
    # Output settings
    DEFAULT_OUTPUT_FORMAT = 'json'
    SUPPORTED_OUTPUT_FORMATS = ['json', 'summary', 'csv']
    
    # Logging settings
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Validation settings
    ENABLE_VALIDATION = True
    STRICT_VALIDATION = False  # If True, fails on warnings
    
    # Integration settings
    INTEGRATE_WITH_MAIN_ORCHESTRATOR = True
    USE_EXISTING_PARAMETER_EXTRACTION = True
    
    @classmethod
    def get_variation_rate(cls, parameter_name: str) -> float:
        """Get default variation rate for a parameter."""
        return cls.DEFAULT_VARIATION_RATES.get(parameter_name, 0.0)
    
    @classmethod
    def validate_variation_rate(cls, rate: float) -> bool:
        """Validate if a variation rate is within acceptable limits."""
        return abs(rate) <= cls.MAX_VARIATION_RATE
    
    @classmethod
    def validate_parameter_value(cls, value: float, parameter_name: str = None) -> bool:
        """Validate if a parameter value is within acceptable limits."""
        if value < cls.MIN_PARAMETER_VALUE:
            return False
        if value > cls.MAX_PARAMETER_VALUE:
            return False
        return True
    
    @classmethod
    def validate_time_period(cls, value: int, unit: str) -> bool:
        """Validate if a time period is within acceptable limits."""
        if unit == 'years':
            return 1 <= value <= cls.MAX_YEARS
        elif unit == 'months':
            return 1 <= value <= cls.MAX_MONTHS
        elif unit == 'quarters':
            return 1 <= value <= cls.MAX_QUARTERS
        else:
            return True  # Allow other units
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """Get configuration as a dictionary."""
        return {
            'default_variation_rates': cls.DEFAULT_VARIATION_RATES,
            'max_variation_rate': cls.MAX_VARIATION_RATE,
            'max_parameter_value': cls.MAX_PARAMETER_VALUE,
            'min_parameter_value': cls.MIN_PARAMETER_VALUE,
            'max_years': cls.MAX_YEARS,
            'max_months': cls.MAX_MONTHS,
            'max_quarters': cls.MAX_QUARTERS,
            'max_execution_time': cls.MAX_EXECUTION_TIME,
            'max_scenarios': cls.MAX_SCENARIOS,
            'parallel_execution': cls.PARALLEL_EXECUTION,
            'default_output_format': cls.DEFAULT_OUTPUT_FORMAT,
            'supported_output_formats': cls.SUPPORTED_OUTPUT_FORMATS,
            'log_level': cls.LOG_LEVEL,
            'enable_validation': cls.ENABLE_VALIDATION,
            'strict_validation': cls.STRICT_VALIDATION,
            'integrate_with_main_orchestrator': cls.INTEGRATE_WITH_MAIN_ORCHESTRATOR,
            'use_existing_parameter_extraction': cls.USE_EXISTING_PARAMETER_EXTRACTION
        } 