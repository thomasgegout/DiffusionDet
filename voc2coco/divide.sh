#!/bin/bash

python voc2coco/divide_val_dataset.py --input voc2coco/val.json --output voc2coco/val_06percent.json --ratio 0.06