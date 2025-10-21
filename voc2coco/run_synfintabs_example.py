#!/usr/bin/env python3
"""
Example usage of the modified synFinTabs_make.py script to create COCO dataset
"""

import subprocess
import sys
from pathlib import Path

def run_conversion_example():
    """Example of how to run the SynFinTabs to COCO conversion"""
    
    # Example command line arguments
    cmd = [
        sys.executable, "synFinTabs_make.py",
        "--output_dir", "coco_dataset",
        "--split", "train", 
        "--num_samples", "100",  # Start with small number for testing
        "--resolution", "512",
        "--sampling_seed", "42"
    ]
    
    print("Running SynFinTabs to COCO conversion with the following command:")
    print(" ".join(cmd))
    print()
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    
    try:
        result = subprocess.run(cmd, cwd=script_dir, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Conversion completed successfully!")
            print("\nOutput:")
            print(result.stdout)
        else:
            print("✗ Conversion failed!")
            print("\nError output:")
            print(result.stderr)
            print("\nStandard output:")
            print(result.stdout)
            
    except Exception as e:
        print(f"✗ Failed to run conversion: {e}")

def show_usage():
    """Show usage information for the script"""
    print("SynFinTabs to COCO Dataset Conversion")
    print("=" * 40)
    print()
    print("Usage:")
    print("python synFinTabs_make.py [OPTIONS]")
    print()
    print("Required options:")
    print("  --output_dir DIR        Root directory for output data")
    print("  --split SPLIT          Split to process (train, validation, test)")
    print()
    print("Optional options:")
    print("  --num_samples N        Maximum number of samples to generate")
    print("  --resolution SIZE      Target resolution for resizing images (default: 512)")
    print("  --sampling_seed SEED   Random seed for sampling (default: 42)")
    print("  --debug               Debug mode")
    print()
    print("Output structure:")
    print("output_dir/")
    print("  └── {split}/")
    print("      ├── images/")
    print("      │   ├── train_000001.jpg")
    print("      │   ├── train_000002.jpg")
    print("      │   └── ...")
    print("      └── annotations/")
    print("          └── instances_{split}.json")
    print()
    print("Categories in COCO format:")
    print("  1: table_row")
    print("  2: table_cell")
    print()

if __name__ == "__main__":
    show_usage()
    
    response = input("Do you want to run an example conversion? (y/n): ")
    if response.lower().startswith('y'):
        run_conversion_example()
    else:
        print("Example conversion skipped.")