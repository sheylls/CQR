# CQR: Component Quotient Reasoning for Inductive Knowledge Graph Completion

This repository provides the PyTorch implementation of **Component Quotient Reasoning (CQR)** for inductive knowledge graph completion. It contains the training and evaluation pipeline used with the FB15k-237, WN18RR, and NELL-995 inductive benchmarks.

## Overview

CQR combines two complementary structural views of a knowledge graph:

- an A*-style entity-level reasoner that explores a query-dependent local graph; and
- a component-level reasoner that operates on a quotient graph obtained after removing the queried relation family.

The two branches are learned jointly and fused at the score level. CQR uses graph structure only: it does not use entity descriptions, relation text, lexical features, or pretrained language models.

## Requirements

- Python 3.10 or later
- PyTorch 2.1 or later
- NumPy 1.24 or later
- SciPy 1.10 or later
- A CUDA-capable GPU

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The training runner requires CUDA and exits explicitly when a GPU is unavailable.

## Datasets

The repository includes four inductive splits (`v1`-`v4`) for each benchmark:

| Dataset | Versions | Training graph | Inference graph |
| --- | --- | --- | --- |
| FB15k-237 | v1-v4 | G1 | G2 |
| WN18RR | v1-v4 | G1 | G2 |
| NELL-995 | v1-v4 | G1 | G2 |

Each split follows this layout:

```text
datasets/inductive/<dataset>/<version>/
├── G1/
│   ├── entities.txt
│   ├── relations.txt
│   ├── train.txt
│   └── valid.txt
└── G2/
    ├── entities.txt
    ├── relations.txt
    ├── train.txt
    ├── valid.txt
    └── test.txt
```

G1 is used for training and validation. G2 contains unseen entities and is used only for inductive inference. Run the following command to list the available splits:

```bash
python scripts/list_datasets.py
```

Dataset provenance is documented in [THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md).

## Training

The following command trains CQR on FB15k-237 v1:

```bash
python scripts/train.py \
  --dataset FB15k-237 \
  --version v1 \
  --device cuda:0 \
  --amp
```

To train on the other benchmarks:

```bash
python scripts/train.py --dataset WN18RR --version v1 --device cuda:0 --amp
python scripts/train.py --dataset NELL-995 --version v1 --device cuda:0 --amp
```

Replace `v1` with `v2`, `v3`, or `v4` to select another split. Hyperparameters are stored in `configs/inductive/` and can be overridden from the command line; use `python scripts/train.py --help` for the complete option list.

By default, a run is written to:

```text
runs/<dataset>_<version>/
├── config.json
├── history.json
├── last.pt
└── best.pt
```

Generated checkpoints and logs are excluded from version control.

## Evaluation

The training script selects `best.pt` using validation MRR and never evaluates on the test split. Evaluate the selected checkpoint separately:

```bash
python scripts/evaluate.py \
  --checkpoint runs/FB15k-237_v1/best.pt \
  --split test \
  --device cuda:0
```

Evaluation uses filtered ranking with true average ties:

```text
rank = 1 + count(score > target_score) + 0.5 × count(other ties)
```

## Reproducibility

- All model parameters are initialized randomly and trained end to end.
- The current target edge and its inverse are removed from the reasoning graph during training.
- Hyperparameter selection and checkpoint selection use G1 validation data only.
- G2 validation and test queries are reserved for final inductive evaluation.
- Known positive answers are filtered before ranking.
- Random seeds and the effective configuration are saved with every run.

Run the lightweight integrity check with:

```bash
python scripts/smoke_test.py
```

## Repository Structure

```text
configs/inductive/   experiment configurations
cqr_mass/            model, data loader, and evaluation modules
datasets/inductive/  FB15k-237, WN18RR, and NELL-995 splits
scripts/train.py     training entry point
scripts/evaluate.py  checkpoint evaluation
scripts/smoke_test.py
```

A concise technical description is available in [METHOD.md](METHOD.md).

## Acknowledgements

The benchmark organization follows the inductive protocols used by [A*Net](https://github.com/DeepGraphLearning/AStarNet) and [AdaProp](https://github.com/LARS-research/AdaProp). Dataset files remain subject to their original source terms.
