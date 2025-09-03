import json
import random
import argparse
from pathlib import Path

def create_val_subset(input_file, output_file, subset_ratio=0.5, seed=42):
    """
    Create a subset of validation data from a COCO format JSON file.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to save the subset JSON file
        subset_ratio (float): Ratio of data to keep (0.5 = 50%)
        seed (int): Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Load the original JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract images and create a subset
    original_images = data['images']
    subset_size = int(len(original_images) * subset_ratio)
    
    print(f"Original dataset size: {len(original_images)} images")
    print(f"Creating subset with {subset_size} images ({subset_ratio*100}%)")
    
    # Randomly sample images
    subset_images = random.sample(original_images, subset_size)
    
    # Get image IDs from the subset
    subset_image_ids = {img['id'] for img in subset_images}
    
    # Filter annotations to only include those for subset images
    subset_annotations = [
        ann for ann in data['annotations'] 
        if ann['image_id'] in subset_image_ids
    ]
    
    # Create the new dataset structure
    subset_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'categories': data['categories'],
        'images': subset_images,
        'annotations': subset_annotations
    }
    
    # Save the subset to the output file
    with open(output_file, 'w') as f:
        json.dump(subset_data, f, indent=2)
    
    print(f"Subset saved to: {output_file}")
    print("Subset contains:")
    print(f"  - {len(subset_images)} images")
    print(f"  - {len(subset_annotations)} annotations")
    print(f"  - {len(subset_data['categories'])} categories")

def main():
    parser = argparse.ArgumentParser(description='Create a subset of validation data from COCO format JSON')
    parser.add_argument('--input', '-i', 
                       default='datasets/PubTables-1M/val.json',
                       help='Input JSON file path')
    parser.add_argument('--output', '-o',
                       default='datasets/PubTables-1M/val_50percent.json', 
                       help='Output JSON file path')
    parser.add_argument('--ratio', '-r', type=float, default=0.5,
                       help='Subset ratio (default: 0.5 for 50%)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' not found!")
        return
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the subset
    create_val_subset(
        input_file=args.input,
        output_file=args.output,
        subset_ratio=args.ratio,
        seed=args.seed
    )

if __name__ == "__main__":
    main()