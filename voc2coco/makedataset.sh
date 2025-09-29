source .venv/bin/activate && python voc2coco/voc2coco.py \
--ann_paths_list "table-data/recognition30/train/trainlist.txt" \
--ann_paths_list_base_dir "/home/exouser/DiffusionDet/table-data/recognition30/train/annotations" \
--labels voc2coco/labels.txt \
--output voc2coco/train.json \
--ext xml