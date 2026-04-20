from enum import Enum

class DatasetMode(Enum):
    """attack modes for dataset"""
    CLEAN = 0
    MIXED_BACKDOOR = 1
    FULL_BACKDOOR = 2
    LABEL_FLIP = 3
    MODEL_POISONING = 4