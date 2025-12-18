# Setup Guide

This guide explains how to properly set up the q-store-examples project following Python best practices.

## Project Structure

```
q-store-examples/
├── pyproject.toml          # Project configuration and dependencies (PEP 517/518)
├── README.md               # Project documentation
├── LICENSE                 # License file
├── MANIFEST.in             # Package data files
├── requirements.txt        # Pip requirements
├── pytest.ini              # Pytest configuration
├── Makefile               # Build automation
├── src/
│   └── q_store_examples/   # Main package (installable)
│       ├── __init__.py
│       ├── examples_v3_4.py
│       ├── utils.py
│       └── ...
├── tests/                  # Test files
│   ├── __init__.py
│   └── test_*.py
├── scripts/                # Utility scripts
│   └── *.py
└── data/                   # Data files
```

## Installation Methods

### Method 1: Editable Install (Development)

**Recommended for development** - changes to source files are immediately reflected:

```bash
# Clone the repository
git clone https://github.com/yucelz/q-store-examples.git
cd q-store-examples

# Install in editable mode
pip install -e .

# Or with optional dependencies
pip install -e ".[ml,data,dev]"
```

### Method 2: Regular Install

For regular use (not development):

```bash
pip install .

# Or from GitHub directly
pip install git+https://github.com/yucelz/q-store-examples.git
```

### Method 3: From requirements.txt

If you just want to run scripts without installing the package:

```bash
pip install -r requirements.txt

# Then run scripts with PYTHONPATH
PYTHONPATH=src python src/q_store_examples/examples_v3_4.py
```

## Running Examples

After installation with `pip install -e .`, you can run scripts directly:

```bash
# From anywhere in the system
python src/q_store_examples/examples_v3_4.py

# Or from the project root
python -m q_store_examples.examples_v3_4

# Or run scripts
python scripts/verify_installation.py
```

## Optional Dependencies

The project includes several optional dependency groups:

- `ml`: Machine learning dependencies (PyTorch, Transformers, etc.)
- `data`: Data processing (pandas, scikit-learn)
- `dev`: Development tools (pytest, black, mypy)
- `all`: All optional dependencies

Install specific groups:
```bash
pip install -e ".[ml]"           # Just ML dependencies
pip install -e ".[ml,data]"      # ML and data dependencies
pip install -e ".[all]"          # Everything
```

## Environment Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:
```
PINECONE_API_KEY=your_key_here
PINECONE_ENVIRONMENT=us-east-1
IONQ_API_KEY=your_key_here
```

## Verification

After installation, verify everything works:

```bash
# Check package is installed
pip show q-store-examples

# Run verification script
python scripts/verify_installation.py

# Run a test
pytest tests/test_basic.py -v
```

## Why This Structure?

This structure follows modern Python best practices:

1. **PEP 517/518 Compliance**: Uses `pyproject.toml` instead of `setup.py`
2. **Src Layout**: Prevents accidental imports of non-installed code
3. **Editable Install**: Development changes immediately available
4. **Proper Namespacing**: `q_store_examples` is properly importable
5. **Optional Dependencies**: Install only what you need
6. **Standard Tools**: Works with pip, build, twine, etc.

## Troubleshooting

### ModuleNotFoundError: No module named 'q_store_examples'

**Solution**: Install the package with `pip install -e .`

### Import errors after changes

**Solution**: If using regular install, reinstall after changes. With editable install (`-e`), changes are automatic.

### Scripts can't find modules

**Solution**: Either:
- Install package: `pip install -e .`
- Use PYTHONPATH: `PYTHONPATH=src python script.py`

## Development Workflow

```bash
# 1. Clone and setup
git clone <repo>
cd q-store-examples
pip install -e ".[dev]"

# 2. Make changes to src/q_store_examples/*.py

# 3. Test (changes are immediately available)
pytest tests/

# 4. Run examples
python src/q_store_examples/examples_v3_4.py

# 5. Format code
black src/ tests/

# 6. Commit and push
git commit -am "Your changes"
git push
```
