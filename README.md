# CARES

## Code

This is the source code for Paper: _Context-aware Session-based Recommendation with Graph Neural Networks_.

## Requirements

- Python 3
- PyTorch == 1.8.1
- torch-geometric == 2.0.4
- tqdm

## Usage

### Data preprocessing:

The code for data preprocessing is extended by [SR-GNN](https://github.com/CRIPAC-DIG/SR-GNN).
Put the raw data into the folder {dataset}/raw/, and run:
For diginetica and yoochoose
```
python process.py --dataset diginetica
python process.py --dataset yoochoose
```
For tmall
```
python process_tmall.py 
```

### Train and evaluate the model:

```
python build_relation_graph.py --dataset diginetica 
python main.py --dataset diginetica
```
