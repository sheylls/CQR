@echo off
python scripts\train.py --dataset NELL-995 --version v1 --device cuda:0 --amp
