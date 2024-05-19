# KANX: Fast Implementation (Approximation) of Kolmogorov-Arnold Network in JAX

Work in progress

## Introduction

Fast Kolmogorov-Arnold Network in JAX based on [`fast-kan`](https://github.com/ZiyaoLi/fast-kan) using [`equinox`](https://github.com/patrick-kidger/equinox).

The original implementation of KAN is [`pykan`](https://github.com/KindXiaoming/pykan).

## Installation
```bash
pip install -r requirements.txt
```

## Example

KANX comes with an example on MNIST:

```bash
python examples/train_mnist.py
```

## Benchmark

Preliminary benchmarks show that this implementation takes about 1 minute to train on 11th Gen Intel Core i5-1135G7 (my laptop).