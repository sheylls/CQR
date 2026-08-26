@echo off
python scripts\evaluate.py --checkpoint runs\FB15k-237_v1\best.pt --split test --device cuda:0
