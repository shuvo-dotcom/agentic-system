# Quickstart Guide

This guide will help you get up and running with the Agentic System in just a few minutes. By the end, you'll be able to process your first energy analysis query.

## Prerequisites

Before starting, ensure you have:

- Completed the [installation process](installation.md)
- Activated your Python virtual environment
- Set up your `.env` file with the required API keys

## Your First Analysis

### Step 1: Import the System

Create a new Python file called `first_analysis.py` and add the following code:

```python
import asyncio
from main import AgenticSystem

async def run_analysis():
    # Initialize the system
    system = AgenticSystem()
    
    # Process a simple query
    result = await system.process_query(
        "Calculate the LCOE for a solar project with CAPEX of $1000/kW, "
        "OPEX of $20/kW/year, capacity factor of 25%, and 25-year lifetime "
        "with 7% discount rate"
    )
    
    print("Analysis Results:")
    print(result)

# Run the analysis
if __name__ == "__main__":
    asyncio.run(run_analysis())
```

### Step 2: Run the Analysis

Execute the script from your terminal:

```bash
python first_analysis.py
```

The system will:
1. Parse the query to understand what calculation is required
2. Identify the necessary formula (LCOE in this case)
3. Extract parameters from the query
4. Perform the calculation
5. Validate the results
6. Return a structured response

### Step 3: Customize the Query

Try modifying the query to explore different calculations:

```python
# For Net Present Value (NPV) calculation
result = await system.process_query(
    "Calculate the NPV for a wind project with initial investment of $5M, "
    "annual revenue of $800K, annual costs of $200K, 20-year lifetime, "
    "and 8% discount rate"
)

# For Capacity Factor analysis
result = await system.process_query(
    "What is the average capacity factor for onshore wind farms in Texas "
    "based on the most recent available data?"
)
```

## Using Direct Agent Calls

For more specific needs, you can call agents directly:

```python
# Call the CalcExecutor agent directly
calc_result = await system.direct_agent_call("CalcExecutor", {
    "formula": "CAPEX / (capacity_factor * 8760 * lifetime * (1 - (1 / ((1 + discount_rate) ** lifetime))))",
    "parameters": {
        "CAPEX": 1000,  # $/kW
        "capacity_factor": 0.25,
        "lifetime": 25,  # years
        "discount_rate": 0.07
    }
})

print("Direct calculation result:", calc_result["data"]["result"])
```

## Working with Data Files

To analyze data from CSV files:

```python
# Load and analyze CSV data
result = await system.process_query(
    "Analyze the capacity factors in data/csv/example_plant_data.csv "
    "and calculate the average monthly values for 2022"
)
```

## What's Next?

Now that you've completed your first analysis, explore these features:

- **Advanced Calculations**: Try more complex formulas and scenarios
- **Data Integration**: Connect to external databases or APIs
- **Custom Exports**: Generate reports in different formats
- **Configuration**: Customize the system for your specific needs

For detailed documentation on these topics, refer to:

- [Advanced Features](../advanced-features/index.md)
- [Configuration Options](../configuration/index.md)
- [API Reference](../api-reference/index.md)

Happy analyzing!
