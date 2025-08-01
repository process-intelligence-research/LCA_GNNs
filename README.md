# Environmental impacts prediction using graph neural networks on molecular graphs

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-AGPL--3.0-blue.svg)](LICENSE)
[![CI Pipeline](https://github.com/process-intelligence-research/LCA_GNNs/workflows/CI%20Pipeline/badge.svg)](https://github.com/process-intelligence-research/LCA_GNNs/actions)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](docker/README.md)

A machine learning framework for predicting environmental impacts from molecular SMILES strings using pre-trained Graph Neural Networks (GNNs). This repository provides ready-to-use models for Life Cycle Assessment (LCA) across 15 environmental impact categories.

## Features

- **🎯 Single Molecule Inference**: Predict environmental impacts from SMILES strings
- **📊 15 Environmental Impact Categories**: Climate change, acidification, eutrophication, toxicity, resource depletion, etc.
- **🧠 Multiple Model Types**: Molecular GNNs, country-specific GNNs, and energy-aware GNNs  
- **⚡ Ready-to-Use**: Pre-trained models with automatic denormalization
- **🔄 Batch Processing**: Process multiple molecules from Excel files
- **🌍 Country-Specific Predictions**: Regional environmental modeling with 90+ countries
- **🧪 Comprehensive Testing**: Full test suite with CI/CD pipeline
- **🔧 Development Ready**: Automated code quality checks and type safety

## Table of Contents

- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [Development & Testing](#development--testing)
- [Research Functions](#research-functions)
- [Model Architecture](#model-architecture)
- [Repository Structure](#repository-structure)
- [Configuration](#configuration)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License & Contact](#license--contact)
- [Contributors](#contributors)

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/process-intelligence-research/LCA_GNNs.git
cd LCA_GNNs

# Install dependencies
pip install -r requirements.txt
```
### Single Molecule Prediction
```bash
# Single environmental impact (e.g., Global Warming Impact)
python main.py --workflow inference \
    --model_path trained_models/GNN_C_single/GNN_C_Gwi_final_lr_5.00e-05.pth \
    --smiles "CCO" \
    --target_task "Gwi" \
    --country_name "Germany"

# All 15 environmental impacts at once (multitask)
python main.py --workflow inference \
    --model_path trained_models/GNN_C_multi_best.pth \
    --smiles "CCO" \
    --dataset_type "GNN_C" \
    --country_name "Germany" \
    --multitask

# Energy-focused environmental assessment
python main.py --workflow inference \
    --model_path trained_models/GNN_E_multi_best.pth \
    --smiles "CCO" \
    --dataset_type "GNN_E" \
    --country_name "Germany" \
    --multitask

# Pure molecular prediction (no geographical factors)
python main.py --workflow inference \
    --model_path trained_models/GNN_M_multi_best.pth \
    --smiles "CCO" \
    --dataset_type "GNN_M" \
    --multitask
```

### Batch Processing
```bash
# Process multiple molecules from Excel file
python main.py --workflow batch \
    --model_path trained_models/GNN_C_multi_best.pth \
    --data_path test_molecules.xlsx
```

**Required Excel format for batch processing:**
| SMILES | country_name | 
|--------|--------------|
| CCO    | Germany      | 
| CC(=O)O| Japan        | 
| c1ccccc1| China       | 

### Programmatic Usage
```python
from src.engines.predict_engines import predict_single_molecule

# GNN_C model (requires country_name)
results = predict_single_molecule(
    model_path="trained_models/GNN_C_single/GNN_C_Gwi_final_lr_5.00e-05.pth",
    smiles="CCO",
    country_name="Germany",  # Required for GNN_C
    dataset_type="GNN_C",
    multitask=True
)

# GNN_E model (uses country_name for automatic energy mix lookup)
results = predict_single_molecule(
    model_path="trained_models/GNN_E_single/GNN_E_Gwi_final_lr_5.00e-04.pth",
    smiles="CCO",
    country_name="Germany",  # Automatically retrieves energy mix
    dataset_type="GNN_E",
    multitask=True
)

# GNN_M model (no additional parameters needed)
results = predict_single_molecule(
    model_path="trained_models/GNN_M_model.pth",
    smiles="CCO",
    dataset_type="GNN_M",
    multitask=True
)

# Results format:
# {
#   "smiles": "CCO",
#   "country_name": "Germany", 
#   "predictions": {
#     "Acid": 1.23e-4,
#     "Gwi": 5.67e-3,
#     "CTUe": 8.90e-6,
#     # ... 12 more categories
#   },
#   "denormalized": True
# }
```

## Research Functions

**Note**: The following functions require access to proprietary training data and are intended for research/development use:

```bash
# Setup configuration templates
python main.py --workflow config

# Data preparation (requires proprietary dataset)
python main.py --workflow data --data_path training_data.xlsx

# Model training (requires proprietary dataset)  
python main.py --workflow train

# Final model training (requires proprietary dataset)
python main.py --workflow final_train
```

## Docker Deployment

Docker support is available for easy deployment and development. All Docker files are in the `docker/` folder.

**Quick Start:**
```bash
# Build and run (from project root)
docker build -f docker/Dockerfile -t lca-gnn:latest .
docker run --rm \
    -v "$(pwd)/trained_models:/app/trained_models:ro" \
    -v "$(pwd)/data:/app/data:ro" \
    lca-gnn:latest \
    python main.py --workflow inference \
    --model_path trained_models/GNN_C_Gwi.pth \
    --smiles "CCO" \
    --target_task "Gwi" \
    --country_name "Germany"

# Using helper scripts
docker/docker-helper.sh build              # Linux/Mac
docker\docker-helper.bat build             # Windows
```

**For comprehensive Docker documentation, deployment strategies, and troubleshooting, see [`docker/README.md`](docker/README.md).**

## Development & Testing

This project maintains high code quality with automated CI/CD pipeline. For detailed development guidelines, see the [Contributing](#contributing) section.

**Quick development setup:**
```bash
# Install dependencies including development tools
pip install -r requirements.txt

# Run all quality checks
python -m ruff check src/ tests/     # Linting
python -m ruff format src/ tests/    # Formatting  
python -m pyright src/               # Type checking
python -m pytest tests/ -v          # Testing
```

**CI Pipeline:** Automated checks for linting, formatting, type checking, and testing on all pull requests.

## Model Architecture

| Model Type | Description | Use Case | Single Molecule Inference |
|------------|-------------|----------|-------------------------|
| **QSPR** | Traditional ML on molecular descriptors | Baseline molecular property prediction | ❌ Not supported* |
| **GNN_M** | Graph neural networks on molecular structure | Advanced molecular property prediction | ✅ Supported |
| **GNN_C** | GNNs with country-specific features | Regional environmental impact modeling | ✅ Supported |
| **GNN_E** | GNNs with energy system features | Energy-focused environmental assessment | ✅ Supported |

> *QSPR models require pre-computed molecular descriptors and cannot be used for single molecule inference from SMILES strings. Use batch prediction with pre-computed descriptor data instead.

### Available Parameters for Inference

**Environmental Impact Categories (target_task):**
- `Acid`: Acidification potential
- `Gwi`: Global warming impact  
- `CTUe`: Ecotoxicity potential
- `ADP_f`: Abiotic depletion potential (fossil fuels)
- `Eutro_f`: Eutrophication potential (freshwater)
- `Eutro_m`: Eutrophication potential (marine)
- `Eutro_t`: Eutrophication potential (terrestrial)
- `CTUh`: Human toxicity potential
- `Ionising`: Ionising radiation potential
- `Soil`: Land use potential
- `ADP_e`: Abiotic depletion potential (elements)
- `ODP`: Ozone depletion potential
- `human_health`: Particulate matter formation potential
- `Photo`: Photochemical ozone formation potential
- `Water_use`: Water use potential

**Countries:** Available countries for GNN_C and GNN_E models are listed in `data/raw/energy_mapping.json`. Examples include "Germany", "United States", "Japan", "China", and 85+ others.

### Training Modes
- **Single-Task**: Separate models for each impact category
- **Multi-Task**: One model predicting all 15 categories simultaneously

## Repository Structure

```
LCA_GNNs/
├── .github/
│   └── workflows/           # CI/CD pipeline configuration
├── docker/                  # Docker deployment files
│   ├── Dockerfile           # Production Docker image
│   ├── Dockerfile.dev       # Development Docker image
│   ├── docker-compose.yml   # Docker Compose configuration
│   ├── docker-helper.sh     # Linux/Mac helper scripts
│   ├── docker-helper.bat    # Windows helper scripts
│   ├── .env.example        # Environment configuration template
│   └── README.md           # Comprehensive Docker documentation
├── src/
│   ├── config/              # Configuration management
│   ├── engines/             # Training, evaluation, and prediction engines
│   ├── models/              # GNN and QSPR model architectures
│   ├── data_processing/     # Dataset creation and preprocessing
│   ├── trainer/             # Core training loops
│   └── scripts.py           # Main pipeline interface
├── tests/                   # Test suite
│   ├── test_imports.py      # Import and structure validation
│   ├── test_inference.py    # Inference functionality tests
│   ├── test_fastapi.py      # API tests (when available)
│   └── conftest.py         # Pytest configuration
├── configs/                 # Configuration templates
├── examples/                # Usage examples and tests
├── data/                    # Data directory
├── trained_models/          # Model checkpoints
├── .dockerignore           # Docker build exclusions
├── pyproject.toml          # Python project configuration
├── main.py                  # CLI workflow interface
└── requirements.txt         # Project dependencies
```

## Configuration

### Project Configuration
The project uses `pyproject.toml` for Python tooling configuration:
- **Pyright**: Type checking configuration optimized for ML projects
- **Build system**: Standard Python packaging configuration

### Training Configuration
YAML-based configuration system with categories:

```yaml
# Example configuration
optimizer:
  learning_rate: 0.001
  weight_decay: 0.0001

training:
  epochs: 500
  batch_size: 20
  k_fold: 10
  task_mode: "single"  # or "multi"

data:
  dataset_type: "GNN_C"
  path: "./data"

model:
  model_type: "GNN_C_single"
  hidden_dim: 128
  num_layers: 3

experiment:
  enable_wandb: true
  project_name: "LCA_Environmental_Impact"
```

## Requirements

- **Python**: 3.9+
- **Dependencies**: PyTorch 2.0+, PyTorch Geometric, RDKit, scikit-learn
- **Optional**: CUDA-compatible GPU for faster training

Core dependencies:
```
torch==2.7.0
torch-geometric
rdkit
scikit-learn
pandas
wandb
numpy
pyyaml
openpyxl
tqdm
requests
```

Development dependencies:
```
pytest>=8.0.0
ruff>=0.1.0
pyright>=1.1.0
fastapi>=0.100.0
uvicorn>=0.20.0
```

For development setup:
```bash
# Install all dependencies including development tools
pip install -r requirements.txt

# Or install development dependencies separately
pip install pytest ruff pyright fastapi uvicorn
```

## Contributing

We welcome contributions! Please follow these guidelines:

### Development Workflow
1. **Fork the repository** and create a feature branch
2. **Install development dependencies**: `pip install -r requirements.txt`
3. **Make your changes** following the code style guidelines
4. **Run quality checks**:
   ```bash
   python -m ruff check src/ tests/          # Linting
   python -m ruff format src/ tests/         # Formatting
   python -m pyright src/                    # Type checking
   python -m pytest tests/ -v               # Testing
   ```
5. **Submit a pull request** with a clear description

### Code Style
- **Formatting**: Automatic formatting with Ruff
- **Linting**: Code quality checks with Ruff
- **Type Hints**: Encouraged but not strictly enforced (ML-friendly configuration)
- **Testing**: Add tests for new functionality in the `tests/` directory

### Project Structure Guidelines
- **Core functionality**: Place in `src/` directory
- **Tests**: Place in `tests/` directory with descriptive names
- **Examples**: Place in `examples/` directory (excluded from CI)
- **Documentation**: Update README.md and docstrings

The CI pipeline will automatically run all quality checks on pull requests.

## License & Contact

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See [LICENSE](LICENSE) for details.

The AGPL-3.0 license allows commercial use while ensuring that any modifications or derivative works are also made available under the same license terms. As co-developers, contributors have co-ownership and user rights that are not limited by the publication license.

 
## Contributors

| | | |
| --- | --- | --- |
| <img src="docs/photos/QingheGao.jpeg" width="50"> | [Qinghe Gao](https://www.pi-research.org/author/qinghe-gao/) | <a href="www.linkedin.com/in/qinghegao" rel="nofollow noreferrer"> <img src="https://i.sstatic.net/gVE0j.png" >  </a> <a href="https://scholar.google.com/citations?user=APquWnUAAAAJ&hl=en" rel="nofollow noreferrer"> <img src="docs/logos/google-scholar-square.svg" width="14">  </a> |
| <img src="docs/photos/Lukas.jpg" width="50"> | [Lukas Schulze Balhorn](https://www.pi-research.org/author/lukas-schulze-balhorn/) | <a href="https://www.linkedin.com/in/lukas-schulze-balhorn-12a3a4205/" rel="nofollow noreferrer"> <img src="https://i.sstatic.net/gVE0j.png" >  </a> <a href="https://scholar.google.com/citations?user=LZZ7piQAAAAJ&hl=en" rel="nofollow noreferrer"> <img src="docs/logos/google-scholar-square.svg" width="14">  </a> |
| <img src="docs/photos/Ale.jpg" width="50"> | [Alessandro Laera](https://www.pi-research.org/author/alessandro-laera/) | <a href="https://www.linkedin.com/in/alessandro-laera-007a70225/" rel="nofollow noreferrer"> <img src="https://i.sstatic.net/gVE0j.png" >  </a>  |
| <img src="docs/photos/Raoul.jpeg" width="50"> | [Raoul Meys](https://www.carbon-minds.com/about/meet-the-team/) | <a href="https://www.linkedin.com/in/raoulmeys/" rel="nofollow noreferrer"> <img src="https://i.sstatic.net/gVE0j.png" >  </a> <a href="https://scholar.google.com/citations?user=n0kI9WIAAAAJ&hl=en" rel="nofollow noreferrer"> <img src="docs/logos/google-scholar-square.svg" width="14">  </a> |
| <img src="docs/photos/Jonas.jpg" width="50"> | [Jonas Goßen](https://www.carbon-minds.com/about/meet-the-team/) | <a href="https://www.linkedin.com/in/jonas-gossen/" rel="nofollow noreferrer"> <img src="https://i.sstatic.net/gVE0j.png" >  </a> |
| <img src="docs/photos/Jana.jpg" width="50"> | [Jana M. Weber](https://www.tudelft.nl/ewi/over-de-faculteit/afdelingen/intelligent-systems/pattern-recognition-bioinformatics/the-delft-bioinformatics-lab/people/jana-m-weber) | <a href="https://www.linkedin.com/in/jana-marie-weber-a260081b0/?originalSubdomain=nl" rel="nofollow noreferrer"> <img src="https://i.sstatic.net/gVE0j.png" >  </a> <a href="https://scholar.google.nl/citations?user=bSZLDNMAAAAJ&hl=en" rel="nofollow noreferrer"> <img src="docs/logos/google-scholar-square.svg" width="14">  </a> |
| <img src="docs/photos/Gregor.jpg" width="50"> | [Gregor Wernet](https://www.linkedin.com/in/gregor-wernet-4132804/?original_referer=https%3A%2F%2Fwww%2Egoogle%2Ecom%2F&originalSubdomain=ch) | <a href="https://www.linkedin.com/in/gregor-wernet-4132804/?original_referer=https%3A%2F%2Fwww%2Egoogle%2Ecom%2F&originalSubdomain=ch"> <img src="https://i.sstatic.net/gVE0j.png" >  </a> <a href="https://scholar.google.com/citations?user=Izk96tQAAAAJ&hl=en" rel="nofollow noreferrer"> <img src="docs/logos/google-scholar-square.svg" width="14">  </a> |
| <img src="docs/photos/Artur.jpg" width="50"> | [Artur M. Schweidtmann](https://www.pi-research.org/author/artur-schweidtmann/) | <a href="https://www.linkedin.com/in/schweidtmann/" rel="nofollow noreferrer"> <img src="https://i.sstatic.net/gVE0j.png" >  </a> <a href="https://scholar.google.com/citations?user=g-GwouoAAAAJ&hl=en" rel="nofollow noreferrer"> <img src="docs/logos/google-scholar-square.svg" width="14">  </a> |
