# PINN Option Pricing Benchmark

> A reproducible benchmark of Physics-Informed Neural Networks (PINNs) for option pricing: evaluating accuracy, arbitrage-freeness, and C++ latency against classical numerical methods.

## 📖 Overview

Physics-Informed Neural Networks (PINNs) combine neural networks with partial differential equations (PDEs) by enforcing that the neural approximation satisfies the underlying physics. In quantitative finance, this promises a unified differentiable framework for both forward pricing and inverse calibration.

However, the existing literature lacks a **controlled comparison** against classical methods (finite differences, Monte Carlo, binomial trees) using identical parameters, domains, and evaluation metrics. 

This project bridges that gap. It provides an end-to-end, open-source benchmark evaluating:
1. **Accuracy**: RMSE and max error against the Black-Scholes closed-form solution.
2. **Arbitrage-freeness**: Put-call parity, monotonicity, convexity, and arbitrage bounds.
3. **Performance**: Training time (Python/PyTorch) vs. inference latency (C++).

## 📂 Project Structure

```text
├── cpp/                  # C++ numerical solvers and benchmark runner
│   ├── CMakeLists.txt
│   ├── include/          # Header-only implementations (Eigen-based)
│   └── src/              # Benchmark runner and unit tests
├── python/               # Python/PyTorch implementations
│   ├── notebooks/        # Jupyter notebooks for M1 (residuals) and M5 (PINN)
│   └── requirements.txt
├── docs/                 # LaTeX source for the accompanying research paper
│   ├── main.tex
│   └── sections/         # Modular sections (finance basics, FD, PINN, etc.)
├── results/              # Benchmark outputs (CSV, JSON, plots)
└── README.md
