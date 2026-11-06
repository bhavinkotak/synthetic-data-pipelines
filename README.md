# Synthetic Data Pipelines - Code Project

<!-- v2-release -->
## Version 2.0 (November 2026)

The v2 release is a preregistered, reproducible rewrite of the paper:
**From Data Scarcity to Abundance? A Reproducible Benchmark of Quality-Gated Synthetic Data Pipelines for Tabular Model Training**

- Paper, code, data sources, results and tests: [`v2/`](v2/)
- Archived on Zenodo: [10.5281/zenodo.20619633](https://doi.org/10.5281/zenodo.20619633) (v1: [10.5281/zenodo.18445584](https://doi.org/10.5281/zenodo.18445584))
- Preregistered verdict: SUPPORTED
- Reproduce: `python v2/code/run_all.py`

Every number in the v2 paper is produced by this code from logged results. Claims from v1 that had no
supporting data were removed; `v2/REVIEW.md` lists them.
<!-- v2-release -->

## Author
**Bhavin Kotak**
- Work Email: bhavin.kotak@ibm.com
- Personal Email: kotak.bhavin@gmail.com

## Description
This project implements the synthetic data generation pipelines described in the research paper:
*"From Data Scarcity to Abundance: Scaling Model Training with LLM-Orchestrated Synthetic Data Pipelines"*

## Features
- **Statistical Synthetic Data Generation**: Generate realistic financial transaction data
- **Quality Validation Framework**: Multi-dimensional quality assessment (fidelity, diversity, utility, privacy)
- **Downstream Utility Testing**: Train and evaluate ML models on synthetic vs real data
- **Privacy Metrics**: Membership inference attack resistance testing
- **PDF Generation**: Convert markdown papers to PDF format

## Project Structure
```
synthetic-data-pipelines/
├── src/
│   ├── __init__.py
│   ├── synthetic_data_generator.py   # Main generation module
│   └── pdf_generator.py              # PDF conversion utility
├── tests/
│   └── test_synthetic_data.py        # Comprehensive test suite
├── data/                             # Generated datasets (created at runtime)
├── pdfs/                             # Generated PDFs (created at runtime)
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Generate Synthetic Data
```python
from src.synthetic_data_generator import (
    SyntheticDataConfig, 
    TransactionGenerator,
    SyntheticDataQualityValidator
)

# Configure generation
config = SyntheticDataConfig(
    n_samples=10000,
    fraud_rate=0.02,
    random_seed=42
)

# Generate data
generator = TransactionGenerator(config)
synthetic_data = generator.generate_dataset()

# Validate quality
validator = SyntheticDataQualityValidator(real_data, synthetic_data)
report = validator.generate_full_report()
```

### Run Main Demo
```bash
python src/synthetic_data_generator.py
```

### Run Tests
```bash
pytest tests/ -v
```

### Generate PDFs
```bash
python src/pdf_generator.py ../enhanced-papers/ -o pdfs/
```

## Quality Metrics

The validation framework evaluates synthetic data across four dimensions:

1. **Statistical Fidelity**: Wasserstein distance, KS test
2. **Diversity**: Range coverage, edge case representation
3. **Utility**: Downstream model performance (AUC-ROC comparison)
4. **Privacy**: Membership inference attack resistance

## License
MIT License

## Citation
If you use this code, please cite:
```bibtex
@article{kotak2026synthetic,
  title={From Data Scarcity to Abundance: Scaling Model Training with LLM-Orchestrated Synthetic Data Pipelines},
  author={Kotak, Bhavin},
  journal={IBM Research Technical Report},
  year={2026}
}
```
