# FedBagSafe Code Structure

This document describes the hierarchical organization of the FedBagSafe codebase.

## Directory Structure

```
fedbagsafe/
├── __init__.py              # Package initialization and exports
├── config.py                # Configuration, constants, and seed initialization
├── fedbagsafe.py            # Main FEDBAGSAFE algorithm implementation
├── data/                    # Data loading and preprocessing
│   ├── __init__.py
│   ├── datasets.py          # Custom dataset classes (CustomDataset, FEMNIST)
│   └── loaders.py           # Data loading and non-IID partitioning
├── models/                  # Neural network architectures
│   ├── __init__.py
│   ├── architectures.py     # Model definitions (ResNet9 variants)
│   └── initializers.py      # Model loading utilities
├── training/                # Training and aggregation logic
│   ├── __init__.py
│   ├── finetune.py          # Fine-tuning and parameter freezing
│   └── aggregation.py       # Client selection and model aggregation
└── utils/                   # Utility functions
    ├── __init__.py
    └── evaluation.py        # Model evaluation metrics
```

## Module Descriptions

### 1. `config.py`
- Device configuration
- Training parameters (N_NODES, N_ROUNDS)
- Attack state schedule for federated rounds
- Reproducibility utilities (seed_everything)

### 2. `data/` Module

#### `datasets.py`
- `DatasetMode`: Enum for different attack modes (CLEAN, MIXED_BACKDOOR, FULL_BACKDOOR, LABEL_FLIP, MODEL_POISONING)
- `CustomDataset`: Custom PyTorch dataset with attack simulation capabilities
- `FEMNIST`: Custom FEMNIST dataset loader

#### `loaders.py`
- `noniid()`: Creates non-IID data partitions for federated learning
- `create_round_dataloaders()`: Creates dataloaders for all rounds with attack configurations
- `dataloader()`: Main function that orchestrates all dataloader creation

### 3. `models/` Module

#### `architectures.py`
- `ImageClassificationBase`: Base PyTorch Lightning module with training/validation logic
- `conv_block()`: Helper function for convolutional blocks
- `ResNet9CIFAR`: ResNet-based model for CIFAR-10/100
- `ResNet9FashionMNIST`: ResNet9 architecture for Fashion-MNIST
- `ResNet9FEMNIST`: ResNet9 architecture for FEMNIST

#### `initializers.py`
- `load_cifar_model()`: Load pretrained CIFAR models
- `load_fashionmnist_model()`: Load pretrained Fashion-MNIST models
- `load_femnist_model()`: Load pretrained FEMNIST models

### 4. `training/` Module

#### `finetune.py`
- `freeze_except_classifier()`: Freezes all model parameters except classifier layers
- `get_dataloader_by_state()`: Returns appropriate dataloader based on attack state
- `finetune()`: Fine-tunes models for all clients in a given round

#### `aggregation.py`
- `selective_weighted_average_aggregation()`: Aggregates selected client models
- `random_client_selection()`: Random client selection with controlled attack aggregation
- `real_random_client_selection()`: Pure random client selection for aggregation

### 5. `utils/` Module

#### `evaluation.py`
- `evaluate_model_lightning()`: Evaluates model accuracy using PyTorch Lightning
- `compute_loss_lightning()`: Computes average loss for a model

### 6. `fedbagsafe.py`
Main implementation of the FEDBAGSAFE algorithm that:
- Loads pretrained models
- Iterates through federated rounds
- Performs client fine-tuning
- Aggregates models with bagging
- Evaluates candidates and selects best model
- Implements fail-safe mechanism

## Usage

### Using the Modular Structure

```python
from fedbagsafe import FEDBAGSAFE, seed_everything

# Initialize random seed
seed_everything()

# Run FedBagSafe on a dataset
FEDBAGSAFE('cifar10', n_rounds=100, n_nodes=100, mode=False)
```

### Using Specific Components

```python
# Import specific components
from fedbagsafe.data import dataloader, CustomDataset, DatasetMode
from fedbagsafe.models import ResNet9CIFAR
from fedbagsafe.training import finetune, selective_weighted_average_aggregation
from fedbagsafe.utils import evaluate_model_lightning

# Use components individually
data = dataloader('cifar10')
model = ResNet9CIFAR(num_classes=10)
# ... custom workflow
```

### Running Main Script

```bash
# Run the main entry point
python main.py

# Or use the backward compatibility wrapper
python FedBagSafe.py
```

## Backward Compatibility

The original `FedBagSafe.py` file has been converted to a backward compatibility wrapper that imports from the new modular structure. This ensures existing code that imports from `FedBagSafe.py` will continue to work.

## Benefits of This Structure

1. **Modularity**: Each component is in its own file, making it easier to understand and maintain
2. **Reusability**: Components can be imported and used independently
3. **Testability**: Individual modules can be tested in isolation
4. **Scalability**: Easy to add new datasets, models, or attack types
5. **Clarity**: Clear separation of concerns (data, models, training, evaluation)
6. **Maintainability**: Changes to one component don't affect others

## Key Design Principles

- **Separation of Concerns**: Data, models, training, and evaluation are separated
- **Single Responsibility**: Each module has a clear, focused purpose
- **DRY (Don't Repeat Yourself)**: Common functionality is centralized
- **Clean Imports**: All modules have proper `__init__.py` files for clean imports
- **Documentation**: Each module and function has docstrings explaining its purpose
