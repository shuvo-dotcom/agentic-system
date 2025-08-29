# Documentation Setup Guide

This guide explains how to set up, build, and deploy the documentation for the Agentic System.

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Git

## Setup

1. Clone the repository:

```bash
git clone https://github.com/shuvo-dotcom/agentic-system.git
cd agentic-system
```

2. Run the setup script:

```bash
bash scripts/setup_docs.sh
```

This script will:
- Create a virtual environment for documentation
- Install MkDocs and required plugins
- Set up the basic documentation structure

## Local Development

To serve the documentation locally:

```bash
mkdocs serve
```

This will start a local server at http://127.0.0.1:8000/ where you can preview the documentation.

## Building the Documentation

To build the documentation site:

```bash
mkdocs build
```

This will create a `site` directory containing the built HTML files.

## Adding Content

The documentation is written in Markdown and organized into sections:

- `docs/index.md`: Main landing page
- `docs/getting-started/`: Onboarding materials
- `docs/configuration/`: Configuration guides
- `docs/troubleshooting/`: Troubleshooting information

## Deployment

The documentation is automatically deployed when changes are pushed to the main branch via GitHub Actions. The workflow is defined in `.github/workflows/docs.yml`.

Manual deployment can be done with:

```bash
mkdocs gh-deploy --force
```

## Documentation Structure

```
agentic-system/
├── mkdocs.yml                          # Main configuration
├── README_DOCS.md                      # Setup instructions (this file)
├── .github/workflows/docs.yml          # Auto-deployment
├── docs/
│   ├── index.md                        # Professional homepage
│   ├── stylesheets/extra.css           # Custom styling
│   ├── javascripts/mathjax.js          # Math support
│   ├── assets/README.md                # Image guidelines
│   ├── getting-started/                # Getting started guides
│   ├── configuration/                  # Configuration documentation
│   └── troubleshooting/                # Troubleshooting guides
└── scripts/
    ├── setup_docs.sh                   # Documentation setup script
    └── install_mkdocs.sh               # MkDocs installer
```

## Customization

- Theme: The documentation uses the Material theme for MkDocs, customizable in `mkdocs.yml`
- Styling: Custom CSS can be added in `docs/stylesheets/extra.css`
- Math Support: MathJax is configured in `docs/javascripts/mathjax.js`

## Contributing to Documentation

1. Create a new branch for your documentation changes
2. Make your changes following the Markdown format
3. Submit a pull request with a clear description of your changes
4. Request a review from the documentation maintainers
