# Self-Healing Query System

The Self-Healing Query System is a sophisticated feature of the PostgresDataProvider agent that enables robust and resilient data retrieval even when initial queries fail.

## Overview

In real-world scenarios, data retrieval often faces challenges:

- Requested data might not exist for specific time periods
- Geographic regions may have incomplete data coverage
- Data may be stored with different naming conventions
- API endpoints might have temporary availability issues

The Self-Healing Query System addresses these challenges by automatically generating fallback queries with progressively relaxed constraints until usable data is found.

## Key Concepts

### Progressive Constraint Relaxation

When an initial query fails, the system applies intelligent relaxation strategies:

1. **Time Constraint Relaxation**: Expanding date ranges or shifting time periods
2. **Geographic Scope Expansion**: Widening the geographic area (city → state → country)
3. **Energy Metric Simplification**: Falling back to related or proxy metrics
4. **General Constraint Removal**: Removing non-essential filters

### Intelligent Fallback Generation

The system uses LLMs to analyze the original query and generate appropriate fallbacks:

```python
async def generate_fallback_query(self, original_query, failed_attempts, strategy):
    """Generate a fallback query using a specific relaxation strategy"""
    prompt = f"""
    Original query: {original_query}
    Failed attempts: {failed_attempts}
    
    Generate a new PostgreSQL query using the '{strategy}' relaxation strategy.
    Explain your changes in a comment at the beginning of the query.
    """
    
    response = await self.llm_provider.get_completion(prompt)
    return self._extract_query_from_response(response)
```

### Metadata Tracking

Each query attempt is tracked with detailed metadata:

```json
{
  "original_query": "SELECT avg(capacity_factor) FROM wind_plants WHERE region='Texas' AND date BETWEEN '2022-01-01' AND '2022-12-31'",
  "attempts": [
    {
      "query": "SELECT avg(capacity_factor) FROM wind_plants WHERE region='Texas' AND date BETWEEN '2022-01-01' AND '2022-12-31'",
      "strategy": "original",
      "success": false,
      "error": "No data available for specified period",
      "timestamp": "2023-05-01T12:34:56Z"
    },
    {
      "query": "SELECT avg(capacity_factor) FROM wind_plants WHERE region='Texas' AND date BETWEEN '2021-01-01' AND '2022-12-31'",
      "strategy": "time_relaxation",
      "success": true,
      "rows_returned": 12,
      "timestamp": "2023-05-01T12:34:58Z"
    }
  ],
  "final_success": true,
  "processing_time_ms": 2345,
  "relaxation_steps": 1
}
```

## Implementation

### Core Components

The Self-Healing Query System consists of three main components:

1. **Query Analyzer**: Parses and understands the structure and constraints of the original query
2. **Strategy Selector**: Determines which relaxation strategy to apply based on query context
3. **Fallback Generator**: Creates new queries with relaxed constraints

### Code Structure

The primary implementation is in the `PostgresDataProvider` class:

```python
async def execute_query(self, query: str, allow_self_healing=True, max_fallback_attempts=3):
    """Execute a query with self-healing capabilities"""
    # Initial query attempt
    result = await self._execute_raw_query(query)
    
    # Return if successful or self-healing is disabled
    if result["success"] or not allow_self_healing:
        return result
    
    # Prepare for fallback attempts
    attempts = [{"query": query, "strategy": "original", "success": False, "error": result["error"]}]
    
    # Try fallback strategies
    for attempt in range(max_fallback_attempts):
        strategy = self._select_relaxation_strategy(query, attempts)
        fallback_query = await self._generate_fallback_query(query, attempts, strategy)
        
        fallback_result = await self._execute_raw_query(fallback_query)
        attempts.append({
            "query": fallback_query,
            "strategy": strategy,
            "success": fallback_result["success"],
            "error": fallback_result.get("error"),
            "rows_returned": len(fallback_result.get("data", [])) if fallback_result["success"] else 0
        })
        
        # Return on successful fallback
        if fallback_result["success"]:
            return {
                "success": True,
                "data": fallback_result["data"],
                "self_healing": {
                    "original_query": query,
                    "attempts": attempts,
                    "final_success": True,
                    "relaxation_steps": attempt + 1
                }
            }
    
    # Return failure after all attempts
    return {
        "success": False,
        "error": "Failed to retrieve data after multiple fallback attempts",
        "self_healing": {
            "original_query": query,
            "attempts": attempts,
            "final_success": False,
            "relaxation_steps": max_fallback_attempts
        }
    }
```

## Relaxation Strategies

### Time Relaxation

Expands time windows to capture more data:

- Expanding day ranges (±7 days)
- Expanding month ranges (±1-3 months)
- Shifting to previous years for seasonal data
- Using year-to-date or trailing twelve months

### Geographic Expansion

Progressively widens geographic scope:

- City → County → State → Region → Country → Global
- Including nearby or similar geographic areas
- Aggregating across multiple regions

### Metric Simplification

Falls back to related or proxy metrics:

- Capacity factor → Availability factor
- Hourly data → Daily averages
- Specific plant data → Plant type averages
- Measured data → Modeled/estimated data

### General Relaxation

Removes non-essential filters:

- Removing specific technology filters
- Generalizing plant classifications
- Reducing precision requirements
- Removing secondary constraints

## Configuration

The Self-Healing Query System can be configured through settings:

```json
{
  "self_healing": {
    "enabled": true,
    "max_fallback_attempts": 3,
    "strategies": ["time_relaxation", "geography_expansion", "metric_simplification", "general_relaxation"],
    "prioritize_strategies_by_query_type": true,
    "relaxation_factors": {
      "time": {
        "day_expansion": 7,
        "month_expansion": 1,
        "year_expansion": 1
      },
      "geography": ["state", "region", "country", "global"]
    },
    "return_all_attempts": true
  }
}
```

## Usage Examples

### Basic Usage

```python
# With self-healing enabled (default)
result = await postgres_provider.execute_query(
    "SELECT avg(capacity_factor) FROM wind_plants WHERE region='Texas' AND date BETWEEN '2022-01-01' AND '2022-12-31'"
)

# Disable self-healing for strict queries
result = await postgres_provider.execute_query(
    "SELECT revenue FROM financial_data WHERE project_id=123",
    allow_self_healing=False
)
```

### Custom Fallback Attempts

```python
# Increase fallback attempts for important queries
result = await postgres_provider.execute_query(
    "SELECT hourly_price FROM electricity_market WHERE region='CAISO' AND date='2023-01-15'",
    max_fallback_attempts=5
)
```

### Accessing Self-Healing Metadata

```python
result = await postgres_provider.execute_query(query)
if result["success"]:
    data = result["data"]
    
    # Check if self-healing was activated
    if "self_healing" in result:
        original_query = result["self_healing"]["original_query"]
        attempts = result["self_healing"]["attempts"]
        relaxation_steps = result["self_healing"]["relaxation_steps"]
        
        print(f"Data retrieved after {relaxation_steps} relaxation steps")
        
        # Show the successful strategy
        successful_attempt = next(a for a in attempts if a["success"])
        print(f"Successful strategy: {successful_attempt['strategy']}")
```

## Best Practices

1. **Set appropriate max_fallback_attempts**: Balance between thoroughness and performance
2. **Consider query importance**: Disable self-healing for critical financial or regulatory data where approximations are not acceptable
3. **Monitor and analyze fallback patterns**: Use the metadata to identify common data gaps
4. **Customize relaxation strategies**: Tailor strategies to your specific data domains

## Limitations

- Not suitable for exact-match requirements (e.g., financial transactions)
- May return less specific data than originally requested
- Requires careful validation of relaxed results
- Additional processing time due to multiple query attempts
