# Data Decomposition Project Structure

## 📁 **Final Project Structure**

```
/home/dsin/projects/data_decomposition/
├── src/                              📦 Source code (moved to src/)
│   └── data_decomposition/
│       ├── __init__.py              # Package init with public API
│       ├── aipw.py                  # AIPW estimator
│       ├── data_decomposer.py       # Data decomposition methods
│       ├── data_generator.py        # Data generators and noise models
│       ├── outcome.py               # Outcome models
│       ├── propensity.py            # Propensity models
│       └── types.py                 # Type definitions
├── tests/                           🧪 Tests (moved to separate folder)
│   ├── __init__.py                  # Test package initialization
│   ├── test_aipw.py                 # AIPW tests
│   ├── test_data.py                 # Data generation tests
│   ├── test_outcome.py              # Outcome model tests
│   └── test_propensity.py           # Propensity model tests
├── setup.py                         📦 Package setup and installation
├── requirements.txt                 📋 External dependencies
├── Makefile                         ⚙️  Build automation and testing
├── run_tests.sh                     🧪 Test execution script (executable)
├── CODE_OF_CONDUCT.md              📄 Code of conduct
├── CONTRIBUTING.md                  📄 Contributing guidelines
├── LICENSE.md                       📄 MIT License
└── README.md                        📄 Project documentation
```

## 🚀 **Usage Examples**

### **Running Tests:**
```bash
# Using the script directly
cd /home/dsin/projects/data_decomposition
./run_tests.sh

# Using Make
make test
make test-verbose

# Individual test files
make test-file FILE=test_aipw.py
```

### **Development Setup:**
```bash
cd /home/dsin/projects/data_decomposition

# Install dependencies
make install
# or
pip install -r requirements.txt

# Install for development
make dev-install
# or
pip install -e .[dev]

# Run full quality checks
make check-all
```

### **Using the Package:**
```python
# After installation, import the library
from data_decomposition import AIPW, Splitting, BinomialGaussian
from data_decomposition import LogisticPropensityModel, OLSOutcomeModel

# Or import specific components
from data_decomposition.aipw import AIPW
from data_decomposition.types import Dataset
```

## 🔧 **Key Changes Made**

### **1. Source Code Organization:**
- **Moved all main modules to `src/data_decomposition/`** - follows Python src-layout best practices
- **Self-contained package** - no external fbcode dependencies
- **MIT license headers** - added to all Python files

### **2. Test Organization:**
- **Tests moved to separate `tests/` directory**
- **Updated imports** - now use absolute imports (`data_decomposition.module`)
- **Proper test package structure** with `__init__.py`

### **3. Build & Test Infrastructure:**
- **Makefile** - comprehensive build automation with quality checks
- **run_tests.sh** - robust test execution script with fallbacks
- **setup.py** - proper package setup with src-layout support
- **requirements.txt** - documented external dependencies

### **4. Development Workflow:**
- **PYTHONPATH handling** - automatically configures `src/` directory
- **Cross-platform scripts** - work on Linux/Mac/Windows
- **Quality tooling** - formatting, linting, type checking integration

## 📝 **Available Make Commands**

```bash
make help           # Show all available commands
make test           # Run all tests
make test-verbose   # Run tests with verbose output
make test-coverage  # Run tests with coverage report
make install        # Install dependencies
make dev-install    # Install development dependencies
make format         # Format code with black
make lint           # Run linting checks
make type-check     # Run type checking
make check-all      # Full quality pipeline
make clean          # Clean generated files
```

The project is now fully organized, self-contained, and ready for development with comprehensive testing infrastructure!
