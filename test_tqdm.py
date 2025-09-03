#!/usr/bin/env python3
"""Test script for tqdm integration with train_accelerate.py"""

import sys
import os
from tqdm.auto import tqdm
import time

def test_tqdm_functionality():
    """Test basic tqdm functionality"""
    print("Testing tqdm progress bars...")
    
    # Test 1: Basic progress bar
    print("\n1. Basic progress bar:")
    with tqdm(total=10, desc="Basic test", unit="item") as pbar:
        for i in range(10):
            time.sleep(0.1)
            pbar.set_postfix({"item": i, "status": "processing"})
            pbar.update(1)
    
    # Test 2: Training-like progress bar
    print("\n2. Training-like progress bar:")
    total_steps = 50
    with tqdm(total=total_steps, desc="Training", unit="step") as pbar:
        for step in range(total_steps):
            # Simulate training step
            loss = 1.0 / (step + 1) + 0.1
            lr = 0.001 * (1 - step / total_steps)
            
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'lr': f'{lr:.2e}',
                'step/s': f'{2.5:.2f}'
            })
            pbar.update(1)
            time.sleep(0.05)
    
    # Test 3: Evaluation progress bar
    print("\n3. Evaluation progress bar:")
    with tqdm(total=20, desc="Evaluating", unit="batch", leave=False) as pbar:
        for batch in range(20):
            val_loss = 0.5 / (batch + 1) + 0.05
            pbar.set_postfix({
                'val_loss': f'{val_loss:.4f}',
                'avg_loss': f'{val_loss:.4f}'
            })
            pbar.update(1)
            time.sleep(0.02)
    
    print("\n✅ All tqdm tests passed!")

def test_import():
    """Test importing train_accelerate with tqdm"""
    try:
        # Test imports
        from train_accelerate import DiffusionDetTrainer, parse_args, setup_config
        print("✅ train_accelerate imports successful")
        
        # Test tqdm import
        from tqdm.auto import tqdm
        print("✅ tqdm import successful")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test basic functionality
    test_tqdm_functionality()
    
    # Test imports
    print("\n" + "="*50)
    print("Testing imports...")
    test_import()
