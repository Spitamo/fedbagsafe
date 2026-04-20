import torch


def get_device(device_type: str = 'cpu') -> torch.device:
    """get appropriate device"""
    if device_type == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')