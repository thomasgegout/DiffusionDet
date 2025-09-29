from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog

register_coco_instances("PubTables-1M_train", {}, "../DiffusionDet/datasets/PubTables-1M/train.json", "/Users/thomasgegout/.cache/huggingface/hub/datasets--bsmock--pubtables-1m/snapshots/35b1c097807e0b07ec5313879b85956b7b3890db/PubTables-1M-Structure/images")

dataset_dicts = DatasetCatalog.get("PubTables-1M_train")
print(f"Loaded {len(dataset_dicts)} samples from PubTables-1M_train")