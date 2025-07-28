# MLP and SVM Classifiers from Scratch using NumPy

This project implements a simple **Multi-Layer Perceptron (MLP)** and **Support Vector Machine (SVM)** from scratch using **NumPy**. It compares the performance of both models on various synthetic classification datasets including Half-Circles and Moons.

## 📊 Datasets Used
- Half-Circles (1000 and 10000 samples)
- Moons (1000 and 10000 samples)

## ✨ Features
- Custom dataset generation and standardization
- MLP implementation using ReLU and Softmax
- SVM implementation with Linear, Polynomial, RBF, and Sigmoid kernels
- 5-Fold Cross-Validation with performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- Visualization of decision boundaries

## 🧠 Model Highlights

### MLP (from scratch)
- Fully connected 2-layer neural network
- He initialization, ReLU activation, Softmax output
- Trained using gradient descent on cross-entropy loss

### SVM (from scratch)
- Dual formulation
- Gradient ascent optimization
- Kernel support: Linear, Polynomial, RBF, Sigmoid

## 🔧 Hyperparameters

| Hyperparameter     | Values Tested       | Final Choice  |
|--------------------|---------------------|----------------|
| Hidden Units       | 16, 32, 64          | 32             |
| Learning Rate      | 0.1, 0.01, 0.001    | 0.01           |
| Epochs             | 50, 100, 150        | 150            |
| Activation         | ReLU, LeakyReLU, Tanh | ReLU         |
| Initialization     | Random, Xavier, He  | He             |

## 📈 Results Summary

| Dataset            | Model | Accuracy | F1-Score |
|--------------------|--------|----------|----------|
| Moons (1000)       | MLP    | 86.8%    | 86.8%    |
| Moons (10000)      | MLP    | 88.1%    | 88.1%    |
| Half-Circles (1000)| MLP    | 98.2%    | 98.2%    |
| Half-Circles (10000)| MLP   | 97.6%    | 97.6%    |
| Moons (1000)       | SVM    | ~89% (linear kernel) | - |

