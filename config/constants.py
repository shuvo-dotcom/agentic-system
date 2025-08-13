"""
Constants Module for the Agentic System
Centralized configuration for all numerical constants and defaults
"""

class TimeConstants:
    """Time-related constants"""
    HOURS_PER_YEAR = 8760
    HOURS_PER_DAY = 24
    DAYS_PER_YEAR = 365
    MONTHS_PER_YEAR = 12
    HOURS_PER_MONTH_AVG = HOURS_PER_YEAR / MONTHS_PER_YEAR  # ~730

class EnergyDefaults:
    """Default values for energy calculations"""
    DEFAULT_CAPACITY_MW = 100.0
    DEFAULT_EFFICIENCY = 0.85
    DEFAULT_LOAD_FACTOR = 0.8
    DEFAULT_CAPACITY_FACTOR = 0.35
    DEFAULT_DEGRADATION_RATE = 0.005  # 0.5% per year

class FinancialDefaults:
    """Default values for financial calculations"""
    DEFAULT_DISCOUNT_RATE = 0.08  # 8%
    DEFAULT_CAPEX_PER_KW = 2000.0  # $/kW
    DEFAULT_OPEX_PER_KW_YEAR = 50.0  # $/kW/year
    DEFAULT_FUEL_COST_MMBTU = 3.0  # $/MMBtu
    DEFAULT_MAINTENANCE_COST = 10.0  # $/kWh

class LLMDefaults:
    """Default values for LLM operations"""
    DEFAULT_MAX_TOKENS = 1000
    DEFAULT_TEMPERATURE = 0.1
    MAX_RETRIES = 3
    TIMEOUT_SECONDS = 60

class SystemDefaults:
    """System-wide default values"""
    DEFAULT_ANALYSIS_YEARS = 20
    DEFAULT_LOG_TRUNCATE_LENGTH = 500
    DEFAULT_QUERY_PREVIEW_LENGTH = 100

def get_time_constant(constant_name: str) -> float:
    """Get time constant dynamically"""
    return getattr(TimeConstants, constant_name.upper(), 1.0)

def get_energy_default(param_name: str, param_info: dict = None) -> float:
    """Get energy default based on parameter name and context"""
    param_lower = param_name.lower()
    
    # Check if we have specific context from parameter info
    if param_info:
        unit = param_info.get('unit', '').lower()
        if 'mw' in unit:
            return EnergyDefaults.DEFAULT_CAPACITY_MW
        elif 'factor' in param_lower or 'efficiency' in param_lower:
            return EnergyDefaults.DEFAULT_EFFICIENCY
    
    # Fallback to name-based matching
    if 'capacity' in param_lower:
        return EnergyDefaults.DEFAULT_CAPACITY_MW
    elif 'efficiency' in param_lower or 'factor' in param_lower:
        return EnergyDefaults.DEFAULT_EFFICIENCY
    elif 'hours' in param_lower and 'year' in param_lower:
        return TimeConstants.HOURS_PER_YEAR
    elif 'energy' in param_lower:
        return EnergyDefaults.DEFAULT_CAPACITY_MW * EnergyDefaults.DEFAULT_LOAD_FACTOR * TimeConstants.HOURS_PER_YEAR
    else:
        return 1.0

def get_financial_default(param_name: str, param_info: dict = None) -> float:
    """Get financial default based on parameter name and context"""
    param_lower = param_name.lower()
    
    if 'cost' in param_lower:
        if 'fuel' in param_lower:
            return FinancialDefaults.DEFAULT_FUEL_COST_MMBTU
        elif 'maintenance' in param_lower:
            return FinancialDefaults.DEFAULT_MAINTENANCE_COST
        elif 'capex' in param_lower or 'capital' in param_lower:
            return FinancialDefaults.DEFAULT_CAPEX_PER_KW
        elif 'opex' in param_lower or 'operating' in param_lower:
            return FinancialDefaults.DEFAULT_OPEX_PER_KW_YEAR
        else:
            return FinancialDefaults.DEFAULT_CAPEX_PER_KW
    elif 'rate' in param_lower and 'discount' in param_lower:
        return FinancialDefaults.DEFAULT_DISCOUNT_RATE
    else:
        return 1000.0

def get_smart_default(param_name: str, param_info: dict = None) -> float:
    """Get intelligent default value based on parameter name and context"""
    param_lower = param_name.lower()
    
    # Energy-related parameters
    if any(word in param_lower for word in ['capacity', 'power', 'energy', 'efficiency', 'factor', 'hours']):
        return get_energy_default(param_name, param_info)
    
    # Financial parameters
    elif any(word in param_lower for word in ['cost', 'price', 'rate', 'capex', 'opex']):
        return get_financial_default(param_name, param_info)
    
    # Time parameters
    elif any(word in param_lower for word in ['year', 'month', 'day', 'hour', 'period']):
        if 'year' in param_lower:
            return SystemDefaults.DEFAULT_ANALYSIS_YEARS
        else:
            return 1.0
    
    # Default fallback
    else:
        return 1.0
