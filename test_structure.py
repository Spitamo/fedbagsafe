#!/usr/bin/env python
# coding: utf-8

"""
Test script to verify the modular structure works correctly.
"""

import sys

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports from fedbagsafe package...")
    
    try:
        # Test main package imports
        from fedbagsafe import FEDBAGSAFE, seed_everything, device
        print("✓ Main package imports successful")
        
        # Test config imports
        from fedbagsafe.config import N_NODES, N_ROUNDS, ATTACK_STATES
        print("✓ Config imports successful")
        
        # Test data imports
        from fedbagsafe.data import CustomDataset, DatasetMode, FEMNIST, noniid, dataloader
        print("✓ Data module imports successful")
        
        # Test model imports
        from fedbagsafe.models import (
            ResNet9CIFAR, ResNet9FashionMNIST, ResNet9FEMNIST,
            load_cifar_model, load_fashionmnist_model, load_femnist_model
        )
        print("✓ Models module imports successful")
        
        # Test training imports
        from fedbagsafe.training import (
            finetune, freeze_except_classifier,
            selective_weighted_average_aggregation,
            random_client_selection, real_random_client_selection
        )
        print("✓ Training module imports successful")
        
        # Test utils imports
        from fedbagsafe.utils import evaluate_model_lightning, compute_loss_lightning
        print("✓ Utils module imports successful")
        
        # Test backward compatibility
        import FedBagSafe
        print("✓ Backward compatibility wrapper imports successful")
        
        print("\n✅ All imports successful! The modular structure is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_constants():
    """Test that constants are accessible."""
    print("\nTesting constants...")
    
    try:
        from fedbagsafe.config import N_NODES, N_ROUNDS, ATTACK_STATES
        assert N_NODES == 100, "N_NODES should be 100"
        assert N_ROUNDS == 100, "N_ROUNDS should be 100"
        assert len(ATTACK_STATES) == 100, "ATTACK_STATES should have 100 entries"
        print("✓ Constants are correctly defined")
        return True
    except Exception as e:
        print(f"❌ Constants test failed: {str(e)}")
        return False

def test_enums():
    """Test that enums are accessible."""
    print("\nTesting enums...")
    
    try:
        from fedbagsafe.data import DatasetMode
        assert hasattr(DatasetMode, 'CLEAN'), "DatasetMode should have CLEAN"
        assert hasattr(DatasetMode, 'MIXED_BACKDOOR'), "DatasetMode should have MIXED_BACKDOOR"
        assert hasattr(DatasetMode, 'FULL_BACKDOOR'), "DatasetMode should have FULL_BACKDOOR"
        assert hasattr(DatasetMode, 'LABEL_FLIP'), "DatasetMode should have LABEL_FLIP"
        assert hasattr(DatasetMode, 'MODEL_POISONING'), "DatasetMode should have MODEL_POISONING"
        print("✓ Enums are correctly defined")
        return True
    except Exception as e:
        print(f"❌ Enums test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("FedBagSafe Modular Structure Tests")
    print("=" * 80)
    
    results = []
    results.append(test_imports())
    results.append(test_constants())
    results.append(test_enums())
    
    print("\n" + "=" * 80)
    if all(results):
        print("✅ All tests passed!")
        print("=" * 80)
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        print("=" * 80)
        sys.exit(1)
