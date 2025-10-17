# FedBagSafe Quick Reference Guide

## Quick Start

### Installation
```bash
git clone https://github.com/Spitamo/fedbagsafe.git
cd fedbagsafe
pip install -r requirements.txt
```

### Run Experiments
```bash
# Run all datasets
python main.py

# Or use backward compatibility
python FedBagSafe.py
```

## Import Guide

### Main Algorithm
```python
from fedbagsafe import FEDBAGSAFE, seed_everything

seed_everything()  # Initialize random seeds
FEDBAGSAFE('cifar10', n_rounds=100, n_nodes=100, mode=False)
```

### Data Loading
```python
from fedbagsafe.data import dataloader, CustomDataset, DatasetMode

# Load data with attack configurations
data_loaders = dataloader('cifar10', malicious_clients=24)

# Create custom dataset
dataset = CustomDataset(images, labels, mode=DatasetMode.CLEAN)
```

### Models
```python
from fedbagsafe.models import ResNet9CIFAR, ResNet9FashionMNIST, ResNet9FEMNIST
from fedbagsafe.models import load_cifar_model

# Create model
model = ResNet9CIFAR(num_classes=10)

# Load pretrained model
model = load_cifar_model(eval_loader, n_classes=10, modelname='cifar10')
```

### Training
```python
from fedbagsafe.training import finetune, freeze_except_classifier
from fedbagsafe.training import selective_weighted_average_aggregation

# Fine-tune models
node_models = finetune(data, modelname, global_model, n_nodes, round_id, state)

# Aggregate models
aggregated = selective_weighted_average_aggregation(node_models, global_model, mode)
```

### Evaluation
```python
from fedbagsafe.utils import evaluate_model_lightning, compute_loss_lightning

# Evaluate model
accuracy = evaluate_model_lightning(model, test_loader)
loss = compute_loss_lightning(model, test_loader)
score = accuracy / loss
```

## Configuration

### Constants
```python
from fedbagsafe.config import N_NODES, N_ROUNDS, ATTACK_STATES, device

print(f"Nodes: {N_NODES}, Rounds: {N_ROUNDS}")
print(f"Device: {device}")
```

### Attack Modes
```python
from fedbagsafe.data import DatasetMode

DatasetMode.CLEAN            # Normal training
DatasetMode.MIXED_BACKDOOR   # 50% backdoor poisoning
DatasetMode.FULL_BACKDOOR    # 100% backdoor poisoning
DatasetMode.LABEL_FLIP       # Label flipping attack
DatasetMode.MODEL_POISONING  # Model poisoning attack
```

## Common Workflows

### 1. Basic Training
```python
from fedbagsafe import FEDBAGSAFE, seed_everything

seed_everything()
FEDBAGSAFE('fashionmnist', n_rounds=100, n_nodes=100)
```

### 2. Custom Data Pipeline
```python
from fedbagsafe.data import noniid, create_round_dataloaders, DatasetMode

# Create non-IID partitions
central_data, central_labels, nodes_data, nodes_labels, \
    test_data, test_labels, test_ds, train_ds, n_classes = noniid('cifar10')

# Create dataloaders with attacks
attack_config = {i: DatasetMode.MIXED_BACKDOOR if i < 24 else DatasetMode.CLEAN 
                 for i in range(100)}
loaders = create_round_dataloaders(nodes_data, nodes_labels, 100, 100, attack_config)
```

### 3. Custom Model Evaluation
```python
from fedbagsafe.models import ResNet9CIFAR
from fedbagsafe.utils import evaluate_model_lightning
from fedbagsafe.data import dataloader

# Load data
data = dataloader('cifar10')
eval_loader = data[1]

# Create and evaluate model
model = ResNet9CIFAR(num_classes=10)
accuracy = evaluate_model_lightning(model, eval_loader)
print(f"Accuracy: {accuracy:.4f}")
```

### 4. Custom Training Loop
```python
from fedbagsafe.data import dataloader
from fedbagsafe.models import load_cifar_model
from fedbagsafe.training import finetune, real_random_client_selection
from fedbagsafe.utils import evaluate_model_lightning

# Setup
data = dataloader('cifar10')
eval_loader = data[1]
model = load_cifar_model(eval_loader, n_classes=10, modelname='cifar10')

# Training loop
for round_id in range(10):
    # Fine-tune nodes
    node_models = finetune(data, 'cifar10', model, 100, round_id, 'benign')
    
    # Aggregate
    aggregated_models = real_random_client_selection(node_models, model, 'cifar10')
    
    # Evaluate and select best
    best_acc = 0
    for agg_model, contributors in aggregated_models:
        acc = evaluate_model_lightning(agg_model, eval_loader)
        if acc > best_acc:
            best_acc = acc
            model = agg_model
    
    print(f"Round {round_id}: Best Acc = {best_acc:.4f}")
```

## File Organization

```
Your Project/
├── fedbagsafe/          # Import from here
│   ├── data/           # Data loading: dataloader, noniid, CustomDataset
│   ├── models/         # Models: ResNet9*, load_*_model
│   ├── training/       # Training: finetune, aggregation functions
│   ├── utils/          # Utils: evaluate_model_lightning, compute_loss
│   ├── config.py       # Constants: N_NODES, N_ROUNDS, device
│   └── fedbagsafe.py   # Main: FEDBAGSAFE function
├── main.py             # Entry point
├── FedBagSafe.py       # Backward compatibility
└── requirements.txt    # Dependencies
```

## Datasets Supported

- **CIFAR-10**: `'cifar10'` - 10 classes, 32×32 images
- **CIFAR-100**: `'cifar100'` - 100 classes, 32×32 images
- **Fashion-MNIST**: `'fashionmnist'` - 10 classes, 28×28 grayscale
- **FEMNIST**: `'femnist'` - 62 classes, 28×28 grayscale

## Attack Types

| Attack | Mode | Description |
|--------|------|-------------|
| Benign | `'benign'` | Normal training, no attacks |
| Backdoor | `'backdoor'` | Backdoor pattern injection |
| Label Flip | `'labelflip'` | Random label flipping |
| Model Poison | `'modelpoisoning'` | Gaussian noise injection |

## Troubleshooting

### Import Error
```python
# Make sure you're in the right directory
import sys
sys.path.insert(0, '/path/to/fedbagsafe')
from fedbagsafe import FEDBAGSAFE
```

### CUDA/CPU
```python
# Check device
from fedbagsafe.config import device
print(f"Using device: {device}")
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

## Documentation

- **README.md**: Overview and setup
- **STRUCTURE.md**: Detailed module documentation
- **REFACTORING_SUMMARY.md**: Refactoring details
- **This file**: Quick reference guide

## Support

For detailed documentation, see:
- Module docstrings: `help(fedbagsafe.data.dataloader)`
- STRUCTURE.md: Comprehensive module guide
- README.md: Getting started guide
