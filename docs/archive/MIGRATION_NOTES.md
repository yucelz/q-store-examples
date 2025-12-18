# Migration Notes - Standalone Q-Store Examples

## Overview

This repository has been converted from a subdirectory of the Q-Store project to a **standalone examples repository** that depends on Q-Store 3.4.3 as an external package.

## Changes Made

### 1. Dependency Management

#### Requirements Files
- **requirements.txt**: Updated to install `q-store==3.4.3` from PyPI
- **requirements-minimal.txt**: Updated to install `q-store==3.4.3` from PyPI
- Both files now include instructions for using either:
  - PyPI: `pip install q-store==3.4.3`
  - Local wheel: `pip install q_store-3.4.3-cp313-cp313-manylinux_2_17_x86_64.whl`

### 2. Makefile Updates

#### New Targets
- `make install`: Installs from PyPI (pip install -r requirements.txt)
- `make install-wheel`: NEW - Installs from local wheel file
- `make install-minimal`: Installs minimal dependencies from PyPI
- `make install-full`: Installs all dependencies from PyPI
- `make install-dev`: Installs with development tools (pytest, black, etc.)

#### Removed Commands
- Removed all `cd .. && pip install -e .` commands
- No longer assumes parent Q-Store directory exists

### 3. README.md Updates

#### Installation Options
1. **Option 1 - PyPI (Recommended)**: Install q-store==3.4.3 from PyPI
2. **Option 2 - Conda**: Create conda env and install q-store==3.4.3
3. **Option 3 - Local Wheel**: Install from local wheel file
4. **Option 4 - Minimal**: Install minimal dependencies only

#### Updated Sections
- Installation instructions now standalone
- Troubleshooting updated to use `pip install q-store==3.4.3`
- Development mode section clarified
- Repository links updated (when applicable)

### 4. Code Updates

#### Hardcoded Paths Removed
Fixed hardcoded paths in the following files:
- `src/q_store_examples/examples_v3_4.py`
- `src/q_store_examples/examples_v3_3_1.py`
- `src/q_store_examples/examples_v3_3.py`
- `src/q_store_examples/examples_v3_2.py`

Changed from:
```python
base_dir="/home/yucelz/yz_code/q-store/examples"
```

To:
```python
base_dir=os.getcwd()
```

#### Import Verification
All example files already use correct imports:
```python
from q_store import QuantumDatabase, DatabaseConfig
from q_store.core import ...
from q_store.ml import ...
from q_store.backends import ...
```

No relative imports found - all examples properly import from installed package.

### 5. Verification Script Updates

**scripts/verify_installation.py**:
- Moved `q_store` from optional to required imports
- Updated error messages to use `pip install q-store==3.4.3`
- Updated next steps to reflect standalone structure
- Fixed example paths (e.g., `src/q_store_examples/quantum_db_quickstart.py`)

## Installation Instructions

### Quick Start (PyPI)

```bash
# Clone this standalone repository
git clone https://github.com/yucelz/q-store-examples.git
cd q-store-examples

# Install Q-Store 3.4.3 and dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your API keys

# Verify installation
python scripts/verify_installation.py
```

### Quick Start (Local Wheel)

```bash
# Clone repository
git clone https://github.com/yucelz/q-store-examples.git
cd q-store-examples

# Copy wheel file to this directory
cp /path/to/q_store-3.4.3-cp313-cp313-manylinux_2_17_x86_64.whl .

# Install using make
make install-wheel

# Or manually:
pip install q_store-3.4.3-cp313-cp313-manylinux_2_17_x86_64.whl
pip install -r requirements.txt --no-deps

# Set up environment
cp .env.example .env
# Edit .env and add your API keys

# Verify installation
python scripts/verify_installation.py
```

## Running Examples

All examples remain the same, just run from the repository root:

```bash
# Basic examples
python src/q_store_examples/basic_example.py
python src/q_store_examples/quantum_db_quickstart.py
python src/q_store_examples/financial_example.py

# ML training examples
python src/q_store_examples/examples_v3_2.py --no-mock
python src/q_store_examples/examples_v3_3.py --no-mock
python src/q_store_examples/examples_v3_3_1.py --no-mock
python src/q_store_examples/examples_v3_4.py --no-mock  # Recommended!
```

## Environment Configuration

The `.env.example` file remains unchanged. Copy and configure:

```bash
cp .env.example .env
```

Required variables:
- `PINECONE_API_KEY`: Required for all examples
- `PINECONE_ENVIRONMENT`: Optional (defaults to us-east-1)

Optional variables:
- `IONQ_API_KEY`: For quantum features
- `IONQ_TARGET`: For quantum simulation (default: simulator)
- `HUGGING_FACE_TOKEN`: For gated models

## Compatibility

### Q-Store Version
- **Required**: q-store==3.4.3
- **Source**: PyPI or local wheel file
- **Python**: 3.8+ (wheel is for Python 3.13 on Linux x86_64)

### Breaking Changes
None - all example code remains compatible. Only the installation method changed.

### Migration Checklist

If you were using the old examples subdirectory:

- [x] Update requirements files
- [x] Update Makefile
- [x] Update README.md
- [x] Remove hardcoded paths
- [x] Update verification scripts
- [x] Test installation from PyPI
- [x] Test installation from wheel
- [x] Verify all examples run correctly

## Testing

After installation, verify everything works:

```bash
# Run verification script
python scripts/verify_installation.py

# Test basic example (no API keys needed for mock mode)
python src/q_store_examples/examples_v3_4.py

# Test with real backends (requires API keys)
python src/q_store_examples/examples_v3_4.py --no-mock
```

## Support

For issues or questions:
1. Check this migration guide
2. Review README.md
3. Check Q-Store documentation
4. Open an issue on GitHub

## Future Updates

To update Q-Store to a newer version:

```bash
# From PyPI
pip install --upgrade q-store

# Or from new wheel file
pip install q_store-X.Y.Z-*.whl --upgrade
```

Examples will work with any Q-Store 3.4.x version. For breaking changes in future major versions, check the Q-Store changelog.
