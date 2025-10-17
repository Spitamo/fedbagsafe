# FedBagSafe: Robust and Personalized Federated Learning Framework

<!-- Badges Section -->
<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.7%2B-blue.svg?logo=python&logoColor=white" alt="Python 3.7+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-1.10%2B-%23ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://pytorch.org/vision/stable/index.html"><img src="https://img.shields.io/badge/TorchVision-0.11%2B-orange?logo=pytorch&logoColor=white" alt="TorchVision"></a>
  <a href="https://www.pytorchlightning.ai/"><img src="https://img.shields.io/badge/PyTorch%20Lightning-2.0%2B-purple?logo=pytorch-lightning&logoColor=white" alt="PyTorch Lightning"></a>
  <a href="https://numpy.org/"><img src="https://img.shields.io/badge/numpy-1.21%2B-informational?logo=numpy&logoColor=white" alt="NumPy"></a>
</p>

**FedBagSafe** is an advanced federated learning protocol for robust, personalized, and privacy-preserving training of deep learning models under adversarial and non-IID conditions.  
It implements a **bagged aggregation scheme, interpretable client-side validation, head-only model updates, and an adaptive fail-safe mechanism** to significantly improve resilience against label flipping, backdoor, and model poisoning attacks.

## Key Features

- **Bagged Aggregation:**  
  Generates multiple candidate models per round using aggregation over random client subsets. Enhances robustness by diluting malicious impact.
- **Client-Side Self-Validation:**  
  Each client autonomously evaluates all candidate models using simple accuracy/loss-based metrics, enabling personalized model selection per round.
- **Adaptive Fail-Safe:**  
  Ensures only beneficial model updates are adopted, preventing sudden accuracy drops.
- **Privacy by Design:**  
  Only head (classifier) layers are exchanged/aggregated, with support for data and update isolation and encrypted computations.
- **Extensive Benchmarking:**  
  Evaluated on FEMNIST, FashionMNIST, CIFAR-10, and CIFAR-100 under a comprehensive schedule of attacks.

## Structure

The codebase has been refactored into a clean, hierarchical structure:

```
fedbagsafe/
├── __init__.py              # Package initialization
├── config.py                # Configuration and constants
├── fedbagsafe.py            # Main FEDBAGSAFE algorithm
├── data/                    # Data loading and preprocessing
│   ├── datasets.py          # Custom dataset classes
│   └── loaders.py           # Non-IID partitioning
├── models/                  # Neural network architectures
│   ├── architectures.py     # Model definitions
│   └── initializers.py      # Model loading utilities
├── training/                # Training and aggregation
│   ├── finetune.py          # Fine-tuning utilities
│   └── aggregation.py       # Client selection strategies
└── utils/                   # Utility functions
    └── evaluation.py        # Evaluation metrics
```

**Key Components:**
- `data/`: Non-IID data partitioning (`noniid()`) and attack simulation (`CustomDataset`, `DatasetMode`)
- `models/`: ResNet9 architectures for CIFAR-10/100, Fashion-MNIST, and FEMNIST
- `training/`: Fine-tuning (`finetune()`), parameter freezing (`freeze_except_classifier()`), and aggregation strategies
- `utils/`: Model evaluation utilities
- Complete experimental pipeline in `FEDBAGSAFE()` for reproducible end-to-end evaluation

For detailed structure documentation, see [STRUCTURE.md](STRUCTURE.md).

## Setup

**Dependencies:**
- Python 3.7+
- PyTorch
- torchvision
- pytorch-lightning
- numpy
- matplotlib

Install requirements:
```bash
pip install -r requirements.txt
# Or manually:
pip install torch torchvision pytorch-lightning numpy matplotlib
```

**Data preparation:**  
- Download FEMNIST, FashionMNIST, CIFAR-10, and CIFAR-100 using torchvision or as specified in the code comments (the FEMNIST loader expects a pre-processed `.pt` file).

## Usage

### Quick Start

```bash
# Run the main entry point
python main.py

# Or use the backward compatibility wrapper
python FedBagSafe.py
```

### Programmatic Usage

```python
from fedbagsafe import FEDBAGSAFE, seed_everything

# Initialize random seed
seed_everything()

# Run FedBagSafe on a dataset
FEDBAGSAFE('cifar10', n_rounds=100, n_nodes=100, mode=False)
```

**Parameters:**
- `DATASET_NAME`: One of `'cifar10'`, `'cifar100'`, `'fashionmnist'`, `'femnist'`
- `n_rounds`: Number of communication rounds (typ. `100`)
- `n_nodes`: Number of clients (typ. `100`)
- `mode`: Set to `True` to use controlled client selection (default `False` for natural random bagging)

**Example:**  
Train and evaluate on all four benchmarks:
```python
configs = ['cifar10', 'cifar100', 'fashionmnist', 'femnist']
for config in configs:
    FEDBAGSAFE(config, 100, 100, mode=False)
```

### Using Individual Components

```python
# Import specific components
from fedbagsafe.data import dataloader, CustomDataset, DatasetMode
from fedbagsafe.models import ResNet9CIFAR
from fedbagsafe.training import finetune, selective_weighted_average_aggregation
from fedbagsafe.utils import evaluate_model_lightning

# Use components individually for custom workflows
data = dataloader('cifar10')
model = ResNet9CIFAR(num_classes=10)
# ... custom workflow
```

## Attack Scenarios Supported

- **Label Flipping**
- **Backdoor Injection (mixed and full)**
- **Model Poisoning**

Each attack can be independently controlled/activated as demonstrated in the code and is seamlessly integrated with the data loader.

## Experimental Results

The framework logs and prints comprehensive metrics, including:
- Classification Accuracy
- Cross-Entropy Loss
- Composite Score (Accuracy / Loss)
- Attack Success Rate (ASR, for backdoor/label flip phases)

See the main paper for performance results and quantitative comparisons.

## Customization

- Modify `noniid()` and dataset samplers for different non-IID scenarios or class/batch size.
- Model architectures can be replaced with any compatible PyTorch module.
- Attack definitions easily extended by editing `CustomDataset` and `DatasetMode`.

## Benefits of Modular Structure

1. **Modularity**: Each component is in its own file, making it easier to understand and maintain
2. **Reusability**: Components can be imported and used independently
3. **Testability**: Individual modules can be tested in isolation
4. **Scalability**: Easy to add new datasets, models, or attack types
5. **Clarity**: Clear separation of concerns (data, models, training, evaluation)

**Acknowledgements:**  
The framework leverages PyTorch Lightning and torchvision for seamless experimentation, and extensive design is inspired by contemporary advances in robust and personalized federated learning.
