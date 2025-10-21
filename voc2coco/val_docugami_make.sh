#!/bin/bash

python voc2coco/voc2coco.py \
    --ann_paths_list table-data/recognition30/val_test/val_test_list.txt \
    --ann_paths_list_base_dir table-data/recognition30/val_test/annotations \
    --labels voc2coco/labels.txt \
    --output voc2coco/val_test.json \
    --ext xml

python voc2coco/add_masks_to_coco.py \
    -i voc2coco/val_test.json \
    -o voc2coco/val_test_docugami_with_masks.json