@echo off
python scripts\train.py --dataset FB15k-237 --version v1 --device cuda:0 --amp
