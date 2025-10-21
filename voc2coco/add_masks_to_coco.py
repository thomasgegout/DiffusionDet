#!/usr/bin/env python3
"""
Script to add segmentation masks to COCO annotations.
Since table cells are rectangular, we generate masks from bounding boxes.
"""

import json
import argparse
from typing import List, Dict, Any
from tqdm import tqdm


def bbox_to_segmentation(bbox: List[float]) -> List[List[float]]:
    """
    Convert a bounding box to a segmentation mask (polygon).
    
    Args:
        bbox: [x, y, width, height] in COCO format
        
    Returns:
        segmentation: List of polygon coordinates [[x1,y1,x2,y2,x3,y3,x4,y4]]
    """
    x, y, w, h = bbox
    # Create a rectangular polygon from the bounding box
    # COCO segmentation format: [x1,y1,x2,y2,x3,y3,x4,y4,...]
    segmentation = [
        x, y,           # top-left
        x + w, y,       # top-right  
        x + w, y + h,   # bottom-right
        x, y + h        # bottom-left
    ]
    return [segmentation]


def add_masks_to_annotations(input_file: str, output_file: str) -> None:
    """
    Add segmentation masks to COCO annotations based on bounding boxes.
    
    Args:
        input_file: Path to input COCO JSON file
        output_file: Path to output COCO JSON file with masks
    """
    print(f"Loading annotations from {input_file}...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"Processing {len(data['annotations'])} annotations...")
    
    # Add segmentation masks to each annotation
    for ann in tqdm(data['annotations'], desc="Adding masks"):
        if 'bbox' in ann and len(ann['bbox']) == 4:
            # Generate segmentation from bounding box
            ann['segmentation'] = bbox_to_segmentation(ann['bbox'])
            
            # Ensure area is calculated correctly
            x, y, w, h = ann['bbox']
            ann['area'] = w * h
            
            # Set iscrowd to 0 (not a crowd annotation)
            ann['iscrowd'] = 0
        else:
            print(f"Warning: Invalid bbox for annotation {ann.get('id', 'unknown')}: {ann.get('bbox', 'missing')}")
            # Set empty segmentation for invalid bboxes
            ann['segmentation'] = []
            ann['iscrowd'] = 0
    
    print(f"Saving annotated data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(data, f)
    
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Add segmentation masks to COCO annotations")
    parser.add_argument("--input", "-i", required=True, help="Input COCO JSON file")
    parser.add_argument("--output", "-o", required=True, help="Output COCO JSON file with masks")
    
    args = parser.parse_args()
    
    add_masks_to_annotations(args.input, args.output)


if __name__ == "__main__":
    main()