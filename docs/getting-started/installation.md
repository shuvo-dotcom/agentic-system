# Installation Guide

This guide will walk you through the process of setting up the Agentic System environment on your machine.

## Prerequisites

Before installing the Agentic System, make sure your system meets the following requirements:

- **Python**: Version 3.11 or higher
- **Operating System**: Linux, macOS, or Windows
- **Memory**: At least 4GB RAM (8GB recommended for larger datasets)
- **Storage**: At least 1GB free disk space
- **API Keys**: OpenAI API key (and optionally other API keys based on your needs)

## Installation Methods

### Method 1: Using pip (Recommended)

```bash
# Create a virtual environment
python -m venv agentic-env

# Activate the virtual environment
# On Linux/macOS
source agentic-env/bin/activate
# On Windows
agentic-env\Scripts\activate

# Install the package
pip install -r requirements.txt
```

### Method 2: From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/agentic-system.git
cd agentic-system

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux/macOS
source venv/bin/activate
# On Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration Setup

1. Create a `.env` file in the root directory by copying the example:

```bash
cp .env.example .env
```

2. Open the `.env` file and add your configuration settings:

```
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4

# Database Configuration
DATABASE_URL=sqlite:///./data.db
VECTOR_DB_PATH=./vector_db

# Logging Configuration
LOG_LEVEL=INFO
```

3. Create required directories:

```bash
mkdir -p logs data/csv data/input_files data/output_files
```

## Verify Installation

To verify that the installation was successful:

```bash
python -m main --check-installation
```

You should see a message confirming that the system is properly installed and configured.

## Database Setup (Optional)

If you're using PostgreSQL instead of the default SQLite:

1. Install PostgreSQL and create a new database
2. Update your `.env` file with the PostgreSQL connection string:

```
DATABASE_URL=postgresql://username:password@localhost:5432/agentic_db
```

3. Run the database initialization script:

```bash
python -m scripts.init_db
```

## Troubleshooting

If you encounter issues during installation:

- **Missing dependencies**: Make sure you're using Python 3.11+ and have installed all requirements
- **API key errors**: Check that your OpenAI API key is valid and correctly set in the `.env` file
- **Database connection issues**: Verify database credentials and that the database server is running

For more detailed troubleshooting guidance, see the [Troubleshooting Guide](../troubleshooting/installation_issues.md).

## Next Steps

Now that you've successfully installed the Agentic System, proceed to the [Quickstart Guide](quickstart.md) to run your first analysis.
