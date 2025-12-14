# FedBagSafe: Robust and Personalized Federated Learning Framework

<!-- Badges Section -->
<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.7%2B-blue.svg?logo=python&logoColor=white" alt="Python 3.7+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-1.10%2B-%23ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://pytorch.org/vision/stable/index.html"><img src="https://img.shields.io/badge/TorchVision-0.11%2B-orange?logo=pytorch&logoColor=white" alt="TorchVision"></a>
  <a href="https://www.pytorchlightning.ai/"><img src="https://img.shields.io/badge/PyTorch%20Lightning-2.0%2B-purple?logo=pytorch-lightning&logoColor=white" alt="PyTorch Lightning"></a>
  <a href="https://numpy.org/"><img src="https://img.shields.io/badge/numpy-1.21%2B-informational?logo=numpy&logoColor=white" alt="NumPy"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
</p>


<p align="center">
<a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5867902">
<img src="https://img.shields.io/badge/Paper-SSRN-blueviolet?style=flat&logo=arxiv&logoColor=white" alt="SSRN Paper">
</a>
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

- `noniid()` and custom data loaders: Realistically non-IID and adversarial partitioning of datasets across clients.
- `CustomDataset`, `DatasetMode`: Flexible attack simulation and dataset preprocessing.
- Model architectures defined for each benchmark.
- `fine_tune()`, `freeze_except_classifier()`, and related functions: Head-only local updating and defense logic.
- Multiple aggregation functions, including **FedBagSafe’s random bagging**.
- Per-round evaluation, including accuracy, loss, and (for attacks) backdoor success rate.
- Complete experimental pipeline (`FEDBAGSAFE()`) for reproducible end-to-end evaluation.

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
pip install torch torchvision pytorch-lightning numpy matplotlib
```

**Data preparation:**  
- Download FEMNIST, FashionMNIST, CIFAR-10, and CIFAR-100 using torchvision or as specified in the code comments (the FEMNIST loader expects a pre-processed `.pt` file).

## Usage

Main entry point is the function:
```python
FEDBAGSAFE(DATASET_NAME, n_rounds, n_nodes, mode=False)
```
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


**Acknowledgements:**  
The framework leverages PyTorch Lightning and torchvision for seamless experimentation, and extensive design is inspired by contemporary advances in robust and personalized federated learning.
