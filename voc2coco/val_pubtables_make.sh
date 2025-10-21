#!/bin/bash

python voc2coco/voc2coco.py \
    --ann_paths_list ~/.cache/huggingface/hub/datasets--bsmock--pubtables-1m/snapshots/35b1c097807e0b07ec5313879b85956b7b3890db/PubTables-1M-Structure/val_filelist.txt \
    --ann_paths_list_base_dir ~/.cache/huggingface/hub/datasets--bsmock--pubtables-1m/snapshots/35b1c097807e0b07ec5313879b85956b7b3890db/PubTables-1M-Structure/ \
    --labels voc2coco/labels.txt \
    --output voc2coco/val.json \
    --ext xml
