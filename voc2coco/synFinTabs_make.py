"""
Copyright (C) 2023 Microsoft Corporation

Process SynFinTabs dataset to create input/output pairs for table structure detection.
Creates white background images with table structure lines from the SynFinTabs dataset.
"""

import argparse
import copy
import json
from pathlib import Path
from datetime import datetime

import numpy
import tqdm
from PIL import Image
from datasets import load_dataset


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir", default="synfintabs_data", help="Root directory for output data"
    )
    parser.add_argument(
        "--split", default="train", help="Split to process (train, validation, test)"
    )
    parser.add_argument(
        "--white_output",
        action="store_true",
        help="Generate white background image + table lines",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug Mode",
    )
    parser.add_argument(
        "--num_samples", type=int, help="Maximum number of samples to generate"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of workers for threads"
    )
    parser.add_argument(
        "--resolution", type=int, default=512, help="Target resolution for resizing images"
    )
    parser.add_argument(
        "--sampling_seed", type=int, default=42, help="Random seed for sampling"
    )
    parser.add_argument(
        "--save_combined_image",
        action="store_true",
        help="Save combined image (input + edited)",
    )
    return parser.parse_args()

def create_coco_annotation_from_bbox(bbox, category_id, annotation_id, image_id):
    """Create COCO format annotation from bbox coordinates.
    
    Args:
        bbox: (x1, y1, x2, y2) format
        category_id: Category ID for the annotation
        annotation_id: Unique annotation ID
        image_id: Image ID this annotation belongs to
    
    Returns:
        COCO format annotation dictionary
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    
    return {
        'id': annotation_id,
        'image_id': image_id,
        'category_id': category_id,
        'bbox': [x1, y1, width, height],  # COCO format: [x, y, width, height]
        'area': width * height,
        'iscrowd': 0,
        'segmentation': []  # Not using segmentation
    }

def extract_table_structure(rows):
    """Extract table structure elements from SynFinTabs format,
    adjusting bboxes so cells touch without pixel gaps using midpoint splitting.
    Works on a deepcopy of rows (non-destructive)."""
    
    rows = copy.deepcopy(rows)  # avoid mutating the input
    table_rows = []
    table_cells = []

    for row in rows:
        row_bbox = row['bbox']
        table_rows.append(row_bbox)

        cells = row['cells']

        # --- Fix horizontal borders within the row ---
        for i in range(len(cells) - 1):
            x1, y1, x2, y2 = cells[i]['bbox']
            nx1, ny1, nx2, ny2 = cells[i+1]['bbox']

            gap = nx1 - x2 - 1
            x2 = x2 + gap // 2 + 1 
            nx1 = x2


            cells[i]['bbox'] = (x1, y1, x2, y2)
            cells[i+1]['bbox'] = (nx1, ny1, nx2, ny2)

    # --- Fix vertical borders between successive rows ---
    for r in range(len(rows) - 1):
        row1, row2 = rows[r], rows[r+1]
        cells1, cells2 = row1['cells'], row2['cells']

        # only compare pairs of cells that exist in both rows
        for c in range(max(len(cells1), len(cells2))):
            
            c1 = min(c, len(cells1)-1)
            c2 = min(c, len(cells2)-1)
            x1, y1, x2, y2 = cells1[c1]['bbox']
            nx1, ny1, nx2, ny2 = cells2[c2]['bbox']

            gap = ny1 - y2 - 1
            y2 = y2 + gap // 2 + 1
            ny1 = y2
        
            cells1[c1]['bbox'] = (x1, y1, x2, y2)
            cells2[c2]['bbox'] = (nx1, ny1, nx2, ny2)

    # --- Collect adjusted bboxes ---
    for row in rows:
        for cell in row['cells']:
            table_cells.append(cell['bbox'])

    return table_rows, table_cells


def main():
    args = get_args()

    # Load the SynFinTabs dataset
    print("Loading SynFinTabs dataset...")
    synfintabs_dataset = load_dataset("ethanbradley/synfintabs")
    
    split = args.split
    output_directory = args.output_dir
    num_samples = args.num_samples
    
    # Create output directories
    images_dir = Path(f"{output_directory}/{split}/images")
    annotations_dir = Path(f"{output_directory}/{split}/annotations")
    images_dir.mkdir(exist_ok=True, parents=True)
    annotations_dir.mkdir(exist_ok=True, parents=True)

    # Get samples from the dataset
    dataset_split = synfintabs_dataset[split]
    total_samples = len(dataset_split) if num_samples is None else min(num_samples, len(dataset_split))
    
    # Shuffle if needed
    indices = list(range(len(dataset_split)))
    numpy.random.seed(args.sampling_seed)
    numpy.random.shuffle(indices)
    indices = indices[:total_samples]

    # Initialize COCO format dictionary
    coco_data = {
        "info": {
            "year": 2024,
            "version": "1.0",
            "description": "SynFinTabs dataset converted to COCO format for table structure detection",
            "contributor": "SynFinTabs to COCO converter",
            "url": "",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "table_row", "supercategory": "table_structure"},
            {"id": 2, "name": "table_cell", "supercategory": "table_structure"}
        ]
    }
    
    annotation_id = 1
    
    print(f"Processing {total_samples} samples...")
    
    for idx, sample_idx in enumerate(tqdm.tqdm(indices)):
        sample = dataset_split[sample_idx]
        
        # Get image and table structure
        image = sample['image']
        rows = sample['rows']
        
        # Create filename
        image_filename = f"{split}_{sample_idx:06d}.jpg"
        image_path = images_dir / image_filename
        
        # Resize image if specified
        if args.resolution:
            original_size = image.size
            scale_x = args.resolution / original_size[0]
            scale_y = args.resolution / original_size[1]
            scale = min(scale_x, scale_y)
            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Scale bounding boxes accordingly
            for row in rows:
                row['bbox'] = [
                    int(row['bbox'][0] * scale),
                    int(row['bbox'][1] * scale),
                    int(row['bbox'][2] * scale),
                    int(row['bbox'][3] * scale)
                ]
                for cell in row['cells']:
                    cell['bbox'] = [
                        int(cell['bbox'][0] * scale),
                        int(cell['bbox'][1] * scale),
                        int(cell['bbox'][2] * scale),
                        int(cell['bbox'][3] * scale)
                    ]
        
        # Save image
        image.save(image_path)
        
        # Create image info
        image_info = {
            "id": idx + 1,
            "file_name": image_filename,
            "width": image.size[0],
            "height": image.size[1]
        }
        coco_data["images"].append(image_info)
        
        # Extract and adjust table structure using the provided function
        table_rows, table_cells = extract_table_structure(rows)
        
        # Create annotations for table rows
        for row_bbox in table_rows:
            annotation = create_coco_annotation_from_bbox(
                bbox=row_bbox,
                category_id=1,  # table_row
                annotation_id=annotation_id,
                image_id=idx + 1
            )
            coco_data["annotations"].append(annotation)
            annotation_id += 1
        
        # Create annotations for table cells
        for cell_bbox in table_cells:
            annotation = create_coco_annotation_from_bbox(
                bbox=cell_bbox,
                category_id=2,  # table_cell
                annotation_id=annotation_id,
                image_id=idx + 1
            )
            coco_data["annotations"].append(annotation)
            annotation_id += 1
    
    # Save COCO annotations
    annotations_file = annotations_dir / f"instances_{split}.json"
    with open(annotations_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print("\nDataset creation complete!")
    print(f"Images saved to: {images_dir}")
    print(f"Annotations saved to: {annotations_file}")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Categories: {len(coco_data['categories'])}")




if __name__ == "__main__":
    main()

