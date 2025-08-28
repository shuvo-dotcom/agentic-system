# Git History Cleanup Guide

This document provides instructions for cleaning sensitive data from your Git repository history. Even if API keys or credentials are now invalid, it's still good practice to remove them from your history to prevent confusion and potential security issues.

## Using the Cleanup Script

We've provided a script (`clean_git_history.sh`) that automates the process of cleaning sensitive data from your Git history.

### Prerequisites

- BFG Repo-Cleaner (the script will attempt to install it using Homebrew if not present)
- Git command-line tools
- Access to push to your repository

### Running the Script

1. Make the script executable:
   ```bash
   chmod +x clean_git_history.sh
   ```

2. Run the script:
   ```bash
   ./clean_git_history.sh
   ```

3. Follow the on-screen instructions. The script will:
   - Create a patterns file for sensitive data replacement
   - Clone a mirror copy of your repository
   - Run BFG to clean the history
   - Prepare the repository for force-pushing

4. After the script completes, you'll need to force-push the cleaned repository:
   ```bash
   cd /path/to/temporary/directory/agentic-system.git
   git push --force
   ```

5. Clean up temporary files as instructed by the script

### What the Script Removes

The script is configured to replace:

- OpenAI API Keys
- Project IDs
- MongoDB Connection Strings

## Manual Cleanup Method

If you prefer to clean your repository history manually:

### Using BFG Repo-Cleaner

1. Install BFG:
   ```bash
   brew install bfg
   ```

2. Create a patterns file:
   ```
   # Create file: git_cleanup_patterns.txt
   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==>REMOVED-API-KEY
   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==>REMOVED-API-KEY
   proj_hq4gfL5gbCQvZEKgV4PNLQz0==>REMOVED-PROJECT-ID
   ```

3. Clone a fresh mirror of your repository:
   ```bash
   git clone --mirror git@github.com:shuvo-dotcom/agentic-system.git
   ```

4. Run BFG:
   ```bash
   bfg --replace-text git_cleanup_patterns.txt agentic-system.git
   ```

5. Clean up the repository:
   ```bash
   cd agentic-system.git
   git reflog expire --expire=now --all
   git gc --prune=now --aggressive
   ```

6. Force-push the changes:
   ```bash
   git push --force
   ```

## After Cleaning the Repository

Regardless of which method you use, notify all collaborators that you've rewritten history. They will need to:

1. Re-clone the repository, OR
2. Reset their local copies:
   ```bash
   git fetch
   git reset --hard origin/main
   ```

## Preventing Future Issues

To avoid committing sensitive data in the future:

1. Use environment variables and .env files (excluded via .gitignore)
2. Consider using git-secrets or pre-commit hooks to prevent committing sensitive data
3. Regularly audit your codebase for sensitive information
4. Use tools like detect-secrets to scan for accidental credential commits
5. Follow the guidelines in SECURITY.md
