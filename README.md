# KANX: Fast Implementation (Approximation) of Kolmogorov-Arnold Network in JAX

Work in progress

## Introduction

Fast Kolmogorov-Arnold Network in JAX based on [`fast-kan`](https://github.com/ZiyaoLi/fast-kan) using [`equinox`](https://github.com/patrick-kidger/equinox).

The original implementation of KAN is [`pykan`](https://github.com/KindXiaoming/pykan).

## Installation
```bash
pip install .
pip install -r requirements.txt
```

## Example

KANX comes with an example on MNIST:

```bash
python examples/train_mnist.py
```

## Benchmark

We tested the implementation on MNIST and report the following wall-time for 3000 epochs:

| Architecture    | Wall time (sec)|
| -------- | ------- |
| CPU (i5-1135G7)  | 130.51   |
| CPU (i9-12900K) | 67.85     |
| GPU (RTX 3070 Ti)    | 13.55    |

Plots from the GPU experiment:


<img width="800" alt="mlp_kan_compare" src="examples/accuracy.png">
<img width="800" alt="mlp_kan_compare" src="examples/loss.png">

More experiments to come...