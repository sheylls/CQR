@echo off
python scripts\train.py --dataset WN18RR --version v1 --device cuda:0 --amp
