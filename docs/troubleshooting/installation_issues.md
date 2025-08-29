# Installation Issues

This page addresses common installation problems and their solutions.

## Environment Setup Issues

### Python Version Problems

**Problem**: System requires Python 3.11+ but you have an older version

**Symptoms**:
```
ERROR: Package requires Python>=3.11 but you have Python 3.x.x
```

**Solutions**:

#### For macOS:
```bash
# Using Homebrew
brew update
brew install python@3.11

# Make it the default Python
echo 'alias python="/usr/local/bin/python3.11"' >> ~/.zshrc
source ~/.zshrc
```

#### For Linux:
```bash
# For Ubuntu/Debian
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# Make it the default Python
echo 'alias python="python3.11"' >> ~/.bashrc
source ~/.bashrc
```

#### For Windows:
1. Download Python 3.11 installer from [python.org](https://www.python.org/downloads/)
2. Run installer with "Add Python to PATH" checked
3. Open a new command prompt to use the updated Python

### Virtual Environment Issues

**Problem**: Errors creating or activating virtual environment

**Symptoms**:
```
Error: Command '['python', '-m', 'venv', 'venv']' returned non-zero exit status 1.
```

**Solutions**:

1. Install venv package if missing:
   ```bash
   # For Ubuntu/Debian
   sudo apt install python3.11-venv
   
   # For macOS
   pip3 install virtualenv
   ```

2. Use virtualenv as an alternative:
   ```bash
   pip install virtualenv
   virtualenv venv
   ```

3. Check file permissions:
   ```bash
   # Fix permissions on directory
   chmod 755 .
   ```

## Dependency Installation Issues

### Package Installation Failures

**Problem**: Installing requirements.txt fails

**Symptoms**:
```
ERROR: Failed building wheel for llvmlite
```

**Solutions**:

1. Install system dependencies:
   ```bash
   # For Ubuntu/Debian
   sudo apt-get install build-essential libffi-dev python-dev

   # For macOS
   brew install llvm
   ```

2. Update pip and setuptools:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

3. Install packages one by one to identify problematic ones:
   ```bash
   while read requirement; do pip install $requirement; done < requirements.txt
   ```

### SSL Certificate Errors

**Problem**: SSL certificate verification fails during package download

**Symptoms**:
```
Could not fetch URL: There was a problem confirming the ssl certificate
```

**Solutions**:

1. Update certificates:
   ```bash
   # For macOS
   /Applications/Python\ 3.11/Install\ Certificates.command
   
   # For Linux
   sudo apt-get install ca-certificates
   ```

2. Temporarily bypass (not recommended for production):
   ```bash
   pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
   ```

## API Key Configuration

### OpenAI API Key Issues

**Problem**: System can't find or authenticate with OpenAI API key

**Symptoms**:
```
openai.error.AuthenticationError: Incorrect API key provided
```

**Solutions**:

1. Verify key format - should be `sk-...` without extra spaces
2. Set environment variable directly:
   ```bash
   # For Linux/macOS
   export OPENAI_API_KEY="sk-..."
   
   # For Windows (Command Prompt)
   set OPENAI_API_KEY=sk-...
   
   # For Windows (PowerShell)
   $env:OPENAI_API_KEY="sk-..."
   ```

3. Check API key status in OpenAI dashboard
4. Create a new API key if necessary

### .env File Issues

**Problem**: System not reading .env file correctly

**Symptoms**:
```
KeyError: 'OPENAI_API_KEY'
```

**Solutions**:

1. Verify .env file location (must be in project root)
2. Check .env file format:
   ```
   OPENAI_API_KEY=sk-...
   ```
   No quotes needed, no spaces around `=`

3. Install python-dotenv if missing:
   ```bash
   pip install python-dotenv
   ```

4. Explicitly load .env file in your code:
   ```python
   from dotenv import load_dotenv
   load_dotenv(verbose=True)
   ```

## Database Setup Issues

### PostgreSQL Installation Problems

**Problem**: PostgreSQL installation or connection fails

**Symptoms**:
```
psycopg2.OperationalError: could not connect to server
```

**Solutions**:

#### For macOS:
```bash
# Install PostgreSQL with Homebrew
brew install postgresql
brew services start postgresql

# Create required database
createdb energy_data
```

#### For Linux:
```bash
# For Ubuntu/Debian
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create required database
sudo -u postgres createdb energy_data
```

#### For Docker:
```bash
# Use Docker Compose
docker-compose up -d postgres
```

### PostgreSQL Connection Issues

**Problem**: System can't connect to PostgreSQL database

**Solutions**:

1. Check connection parameters in `config/postgres_settings.json`
2. Verify PostgreSQL is running:
   ```bash
   # For macOS/Linux
   pg_isready
   
   # Check status on systemd systems
   sudo systemctl status postgresql
   ```

3. Test connection with psql:
   ```bash
   psql -h localhost -U postgres -d energy_data
   ```

4. Check firewall settings if connecting remotely

## System-Specific Issues

### macOS Specific Problems

**Problem**: Missing compiler tools on macOS

**Symptoms**:
```
xcrun: error: invalid active developer path
```

**Solution**:
```bash
xcode-select --install
```

### Linux Specific Problems

**Problem**: Missing system libraries

**Symptoms**:
```
ImportError: libGL.so.1: cannot open shared object file
```

**Solution**:
```bash
sudo apt-get update
sudo apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0
```

### Windows Specific Problems

**Problem**: Path length limitations

**Symptoms**:
```
ERROR: Could not install packages due to an OSError: [WinError 206]
```

**Solution**:
1. Enable long paths in Windows:
   - Run regedit
   - Navigate to `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
   - Set `LongPathsEnabled` to `1`
   - Restart computer

2. Use shorter installation directory

## Next Steps

If you've resolved your installation issues:

- Proceed to the [Quickstart Guide](../getting-started/quickstart.md)
- Verify installation with system diagnostics:
  ```bash
  python -m main --check-installation
  ```
- Run the test suite to ensure everything is working:
  ```bash
  python -m pytest tests/
  ```

If you're still experiencing issues:

- Check the [GitHub repository](https://github.com/yourusername/agentic-system/issues) for similar issues
- Post a detailed description of your problem including:
  - Your operating system and version
  - Python version
  - Full error message
  - Steps to reproduce
