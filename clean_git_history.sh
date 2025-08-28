#!/bin/bash
# Script to clean API keys and sensitive data from Git history
# This script uses BFG Repo-Cleaner which must be installed first

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===== Git Repository History Cleaner =====${NC}"
echo -e "${YELLOW}This script will clean sensitive data from your Git history.${NC}"
echo -e "${YELLOW}WARNING: This will rewrite Git history and require force-pushing.${NC}"
echo ""

# Check if BFG is installed
if ! command -v bfg &> /dev/null; then
    echo -e "${RED}BFG Repo-Cleaner is not installed.${NC}"
    echo -e "${YELLOW}Installing BFG using Homebrew...${NC}"
    
    if ! command -v brew &> /dev/null; then
        echo -e "${RED}Homebrew is not installed. Please install Homebrew first or BFG manually.${NC}"
        echo "Visit: https://rtyley.github.io/bfg-repo-cleaner/"
        exit 1
    fi
    
    brew install bfg
    
    if ! command -v bfg &> /dev/null; then
        echo -e "${RED}Failed to install BFG. Please install it manually.${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}BFG installed successfully.${NC}"
fi

echo -e "${YELLOW}Creating patterns file for sensitive data replacement...${NC}"

# Create a patterns file with all the sensitive data to replace
cat > git_cleanup_patterns.txt << 'EOL'
# OpenAI API Keys
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==>REMOVED-API-KEY
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==>REMOVED-API-KEY

# Project IDs
proj_hq4gfL5gbCQvZEKgV4PNLQz0==>REMOVED-PROJECT-ID

# MongoDB Connection String
mongodb\+srv://test_toconnect:BPhKfz73xEcHw5lz@cluster0.d5dfq.mongodb.net/\?retryWrites=true&w=majority&appName=Cluster0==>mongodb://localhost:27017/
EOL

echo -e "${GREEN}Patterns file created.${NC}"
echo ""

# Ask for confirmation
echo -e "${YELLOW}Are you sure you want to proceed? This will require a force-push to update the remote repository.${NC}"
read -p "Type 'yes' to confirm: " confirmation

if [ "$confirmation" != "yes" ]; then
    echo -e "${RED}Operation cancelled.${NC}"
    rm git_cleanup_patterns.txt
    exit 1
fi

echo ""
echo -e "${BLUE}=== Starting Repository Cleanup ===${NC}"

# Create a temporary directory for the mirror repository
TEMP_DIR=$(mktemp -d)
REPO_URL=$(git config --get remote.origin.url)
REPO_NAME="agentic-system"

echo -e "${YELLOW}Cloning a fresh mirror copy of the repository...${NC}"
git clone --mirror "$REPO_URL" "$TEMP_DIR/$REPO_NAME.git"

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to clone the repository.${NC}"
    rm git_cleanup_patterns.txt
    rm -rf "$TEMP_DIR"
    exit 1
fi

echo -e "${GREEN}Repository cloned successfully.${NC}"
echo -e "${YELLOW}Running BFG to clean the repository history...${NC}"

# Run BFG to replace sensitive data
bfg --replace-text git_cleanup_patterns.txt "$TEMP_DIR/$REPO_NAME.git"

if [ $? -ne 0 ]; then
    echo -e "${RED}BFG failed to clean the repository.${NC}"
    rm git_cleanup_patterns.txt
    rm -rf "$TEMP_DIR"
    exit 1
fi

echo -e "${GREEN}BFG completed successfully.${NC}"
echo -e "${YELLOW}Cleaning up the repository...${NC}"

# Clean up the repository
cd "$TEMP_DIR/$REPO_NAME.git"
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo -e "${GREEN}Repository cleaned successfully.${NC}"
echo ""
echo -e "${BLUE}=== Next Steps ===${NC}"
echo -e "${YELLOW}To complete the process, you need to force-push the cleaned repository:${NC}"
echo ""
echo -e "${GREEN}cd $TEMP_DIR/$REPO_NAME.git${NC}"
echo -e "${GREEN}git push --force${NC}"
echo ""
echo -e "${RED}WARNING: Force-pushing will rewrite history on the remote repository.${NC}"
echo -e "${RED}All collaborators will need to re-clone or carefully reset their local copies.${NC}"
echo ""
echo -e "${YELLOW}After verifying everything is correct, you can remove the temporary files:${NC}"
echo -e "${GREEN}rm git_cleanup_patterns.txt${NC}"
echo -e "${GREEN}rm -rf $TEMP_DIR${NC}"

# Don't delete the patterns file or temp directory yet, in case the user wants to check them
echo ""
echo -e "${BLUE}=== Operation Complete ===${NC}"
