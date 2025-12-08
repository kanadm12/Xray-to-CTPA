#!/usr/bin/env python3
"""
Diagnostic script to check if DataLoader workers are spawning correctly.
"""
import torch
from torch.utils.data import DataLoader, TensorDataset
import os
import time

def test_workers():
    """Test if workers spawn with current configuration."""
    print("=" * 80)
    print("DataLoader Worker Spawn Test")
    print("=" * 80)
    
    # Create dummy dataset
    dummy_data = torch.randn(1000, 3, 32, 32)
    dummy_labels = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(dummy_data, dummy_labels)
    
    # Test configurations
    configs = [
        {"num_workers": 0, "persistent_workers": False, "multiprocessing_context": None},
        {"num_workers": 4, "persistent_workers": False, "multiprocessing_context": 'spawn'},
        {"num_workers": 4, "persistent_workers": True, "multiprocessing_context": 'spawn'},
        {"num_workers": 20, "persistent_workers": True, "multiprocessing_context": 'spawn'},
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n--- Test {i}: {config} ---")
        
        # Remove None values from config
        loader_config = {k: v for k, v in config.items() if v is not None}
        if config["num_workers"] == 0:
            loader_config.pop("persistent_workers", None)
        
        try:
            loader = DataLoader(
                dataset,
                batch_size=4,
                shuffle=True,
                pin_memory=True,
                **loader_config
            )
            
            # Count processes before iteration
            pid = os.getpid()
            os.system(f"ps aux | grep {pid} | grep -v grep | wc -l")
            
            print(f"Starting iteration with {config['num_workers']} workers...")
            
            # Iterate a few batches
            for j, (data, labels) in enumerate(loader):
                if j == 0:
                    print(f"First batch received, shape: {data.shape}")
                    time.sleep(2)  # Give workers time to spawn
                    print(f"Process count after first batch:")
                    os.system(f"ps aux | grep {pid} | grep -v grep | wc -l")
                if j >= 2:
                    break
            
            print(f"✓ Test passed with {config['num_workers']} workers")
            
            # Cleanup
            del loader
            time.sleep(1)
            
        except Exception as e:
            print(f"✗ Test failed: {e}")
    
    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)

if __name__ == "__main__":
    test_workers()
