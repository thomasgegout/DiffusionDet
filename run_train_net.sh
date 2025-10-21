#!/bin/bash

source .venv/bin/activate && python train_net.py --num-gpus 1 --config-file configs/diffdet.tables.res50.cpu.yaml