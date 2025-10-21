#!/bin/bash

python voc2coco/voc2coco.py \
    --ann_paths_list table-data/recognition30/train/train_list.txt \
    --ann_paths_list_base_dir table-data/recognition30/train/annotations \
    --labels voc2coco/labels.txt \
    --output voc2coco/train.json \
    --ext xml

python voc2coco/add_masks_to_coco.py \
    -i voc2coco/train.json \
    -o voc2coco/train_with_masks.json 