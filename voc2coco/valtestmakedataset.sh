source .venv/bin/activate && python voc2coco/voc2coco.py \
--ann_paths_list "table-data/recognition30/val_test/val_test_list.txt" \
--ann_paths_list_base_dir "/home/exouser/DiffusionDet/table-data/recognition30/val_test/annotations" \
--labels voc2coco/labels.txt \
--output voc2coco/val_test.json \
--ext xml