# Particle Swarm Optimization (PSO) Hyperparameter Optimization

This Python module implements hyperparameter optimization using Particle Swarm Optimization (PSO) for various machine learning algorithms in classification task. PSO is a population-based optimization technique inspired by the social behavior of birds flocking or fish schooling.

## Overview

The PSOOptimizer class provided in this module allows users to optimize hyperparameters for four different types of machine learning algorithms:

* K-Nearest Neighbors (KNN)
* Random Forest (RF)
* Decision Tree (DT)
* Support Vector Classifier (SVC)

The optimization process aims to find the best set of hyperparameters that maximize the accuracy of the respective classifier on a given dataset.

## Requirements

- Python 3.x
- Required Python packages: numpy, joblib, scikit-learn, tqdm

Make sure to install these dependencies using pip before using the module.



## Usage

1. Install the `pso-optimizer` library:

```bash
pip install pso-optimizer
```
2. Example usage is in `main.py` file.

Files
* `main.py`: The main script to run PSO hyperparameter optimization.
* `pso_optimizer.py`: Contains the PSOOptimizer class for PSO optimization.
* `hyperparameter_mappings.py`: Contains mappings for hyperparameters used in different machine learning models.
* `README.md`: This file.

## Acknowledgments

The implementation of PSO hyperparameter optimization is inspired by the paper "The Particle Swarm — Explosion, Stability, and Convergence in a Multidimensional Complex Space" by Clerc and Kennedy.

## Citation

If you use this package in your work, please cite it using the following information:
@software{pso_optimizer,
  author       = {Mert Bayraktar},
  year         = {2024},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/mBayraktar12/PSO-Hyperparameter-Selection/tree/main}},
  version      = {1.0.0}
}
