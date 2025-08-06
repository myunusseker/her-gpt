# her-gpt

A Python project created with VS Code.

## Description

This is a Python project template with a basic structure to get you started with your development.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   ```bash
   # On Linux/Mac
   source venv/bin/activate
   
   # On Windows
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script:
```bash
python main.py
```

## Project Structure

```
her-gpt/
├── main.py              # Main entry point
├── requirements.txt     # Project dependencies
├── README.md           # This file
├── .gitignore          # Git ignore rules
└── .github/
    └── copilot-instructions.md  # Copilot customization
```

## Development

This project includes development tools:
- `pytest` for testing
- `black` for code formatting
- `flake8` for linting
- `mypy` for type checking

## Contributing

1. Make your changes
2. Run tests: `pytest`
3. Format code: `black .`
4. Check linting: `flake8`
5. Type check: `mypy .`
