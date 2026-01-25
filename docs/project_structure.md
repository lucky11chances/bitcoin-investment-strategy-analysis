# Project Structure Explanation

## 📁 Overall Structure

```
bitcoin-investment-strategies/
├── src/                             # Source code folder (All Python code)
│   ├── strategies/                  # Strategy implementation modules
│   │   ├── __init__.py             # Strategy module init
│   │   ├── hodl.py                 # HODL strategy (training set)
│   │   ├── dca.py                  # DCA strategy (training set)
│   │   ├── dca_test.py             # DCA strategy (test set)
│   │   ├── quant_rf.py             # Quant strategy (training set)
│   │   └── quant_test.py           # Quant strategy (test set)
│   ├── __init__.py                 # Package init file
│   ├── config.py                   # Global configuration module
│   ├── metrics.py                  # Performance metrics calculation module
│   ├── utils.py                    # Common utility functions
│   └── main.py                     # Main program (full analysis)
├── data/                            # Data folder
│   ├── bitcoin_train_2010_2020 copy.csv    # Training set data
│   ├── bitcoin_test_2023_2024 copy.csv     # Test set data
│   └── bitcoin_valid_2021_2022 copy.csv    # Validation set data
├── docs/                            # Documentation folder
│   ├── README.md                    # Detailed technical documentation
│   ├── strategy_comparison_report.md  # Complete comparison analysis report
│   ├── quantitative_strategy_summary.md # Quant strategy technical details
│   ├── dca_and_hodl_assumptions.md    # Strategy assumptions analysis
│   └── project_structure.md           # This document
├── .git/                            # Git version control
├── .gitignore                       # Git ignore config
├── .venv/                           # Python virtual environment
├── run.py                           # Project entry point (Recommended)
└── README.md                        # Project introduction
```

## 🎯 Design Principles

### 1. **Professional src/ Structure**
- All Python source code is unified in the `src/` folder.
- Complies with Python project best practices.
- Clear separation of code and data.

### 2. **Modular Design**
- **config.py**: Single Source of Truth
  - All path configurations
  - All parameter constants
  - Pre-trained weights
  
- **metrics.py**: Unified indicator calculation
  - Sharpe Ratio
  - Sortino Ratio
  - Max Drawdown
  - Volatility
  - Avoids code duplication

- **utils.py**: Common utility functions
  - Data loading
  - Formatted output
  - Table printing

- **strategies/**: Strategy implementation module
  - Independent strategy files
  - Unified use of config/metrics/utils
  - Easy to extend new strategies

### 3. **Package Structure Design**
- Uses `__init__.py` to establish correct Python package structure.
- Supports module import: `from src.strategies import hodl_compute`
- Version management: Version number defined in `src.__init__.py`

### 4. **Clear Entry Point**
- `run.py`: Main entry point in the project root directory
  - Automatically configures Python path
  - Calls `src.main` module
  - Executable permissions (chmod +x)

## 🚀 Usage

### Method 1: Use run.py (Recommended)
```bash
python run.py
```

### Method 2: Run main module directly
```bash
python -m src.main
```

### Method 3: Run individual strategies
```bash
# HODL Strategy
python -m src.strategies.hodl

# DCA Strategy (Training Set)
python -m src.strategies.dca

# Quant Strategy (Training Set)
python -m src.strategies.quant_rf
```

## 📊 Data Flow

```
Data Files (data/*.csv)
    ↓
config.py (Path Config)
    ↓
utils.py (Data Loading)
    ↓
strategies/*.py (Strategy Calculation)
    ↓
metrics.py (Performance Evaluation)
    ↓
main.py (Integrated Analysis)
    ↓
run.py (User Entry)
```

## 🔧 Technical Advantages

### Improvements Compared to Before Refactoring

**Before Refactoring**:
```
test/
├── hodl.py
├── dca.py
├── dca_test.py
├── quant_rf.py
├── quant_test.py
├── config.py
├── metrics.py
├── utils.py
├── main.py
├── test_comparison.py    # Redundant
├── data/
└── docs/
```
- ❌ Messy files, flat structure
- ❌ Code duplication (Similar metrics calculation in every file)
- ❌ Hardcoded parameters scattered
- ❌ Does not comply with professional Python project standards

**After Refactoring**:
```
test/
├── src/                  # Professional structure
│   ├── strategies/      # Modular
│   ├── config.py       # Centralized config
│   ├── metrics.py      # Unified calculation
│   └── ...
├── data/               # Data separation
├── docs/              # Documentation separation
└── run.py             # Clear entry point
```
- ✅ Professional project structure
- ✅ Modular, maintainable
- ✅ High code reuse rate
- ✅ Easy to extend and test
- ✅ Complies with Python best practices

### Code Reuse Results

- Eliminated **150+ lines of duplicate code**
- All strategies share metrics calculation logic
- Unified configuration management (`config.py`)
- Common utility functions (`utils.py`)

## 📝 Maintenance Guide

### Adding New Strategies
1. Create a new .py file in `src/strategies/`.
2. Import and use `config`, `metrics`, `utils`.
3. Add export in `src/strategies/__init__.py`.
4. Integrate in `src/main.py` (Optional).

### Modifying Parameters
- Edit `src/config.py` directly.
- All strategies automatically use new parameters.

### Adding New Metrics
- Add calculation function in `src/metrics.py`.
- Update `calculate_all_metrics()` function.
- All strategies automatically get new metrics.

## 🌟 Why use src/ folder?

1. **Industry Standard**: Common practice for Python professional projects.
2. **Package Management**: Facilitates packaging the project into installable packages.
3. **Clear Import**: `from src.module import func` is more explicit.
4. **Test Isolation**: Test code can be placed in tests/ folder, separated from source code.
5. **Tool Compatibility**: Many development tools default to recognizing src/ structure.

## 📚 Related Documentation

- [README.md](../README.md) - Project Quick Start
- [docs/README.md](README.md) - Full Technical Documentation
- [docs/strategy_comparison_report.md](strategy_comparison_report.md) - Performance Analysis Report
- [docs/project_structure.md](project_structure.md) - This document

---

**Last Update**: 2024-12-14
**Project Version**: v1.0.0
