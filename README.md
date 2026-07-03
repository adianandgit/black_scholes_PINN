# BlackScholesPINN

A Python implementation of Physics-Informed Neural Networks (PINNs) for solving the Black-Scholes partial differential equation used in option pricing.

---

## 📌 What is this?

This repository demonstrates how to use Physics-Informed Neural Networks (PINNs) to learn the solution of the **Black-Scholes equation** — a foundational model in financial mathematics for pricing European call options.

PINNs are neural networks that are trained not just on data, but also on the **underlying physical (or financial) laws** described by differential equations.

---

## 🚀 Features

- ✅ Clean modular design
- ✅ Configurable via `config.json`
- ✅ Supports noisy synthetic data generation
- ✅ Enforces PDE constraint using autograd
- ✅ Lightweight and dependency-free (only PyTorch + NumPy + matplotlib)
- ✅ Fully reproducible

---

## 🧠 What You’ll Learn

- How to generate synthetic financial data using the Black-Scholes formula
- How to train a neural network to obey a PDE using automatic differentiation
- How to combine **data loss** and **PDE loss** in a single objective
- How to modularize ML code for experimentation and reuse

---

## 🗂 Project Structure

```
.
├── bs_pinn/                     # Package with all model components
│   ├── network.py               # Neural network architecture
│   ├── dataset.py               # Data generation utilities
│   ├── losses.py                # PDE residual and loss computation
│   ├── trainer.py               # High level training interface
│   └── utils.py                 # Black-Scholes analytic helpers
├── config.json                  # Hyperparameters
├── main.py                      # Command line entry point
├── example/BlackScholesModel.ipynb
└── README.md
```



## ⚙️ How to Use

1. Install dependencies

```bash
pip install torch numpy matplotlib scipy
```

2. Train the PINN

```bash
python main.py
```

3. Modify configuration

All training and model parameters can be changed in `config.json`, including:

- `K` — Strike price of the option  
- `T` — Time to maturity (in years)  
- `r` — Risk-free interest rate  
- `sigma` — Volatility of the underlying asset  
- `N_data` — Number of synthetic data points to generate  
- `bias` — Constant value added to the synthetic labels  
- `noise_variance` — Standard deviation of Gaussian noise added to synthetic data  
- `min_S`, `max_S` — Range for sampling stock prices (`S`)  
- `epochs` — Number of training iterations  
- `lr` — Learning rate for the optimizer  
- `log_interval` — Number of epochs between log printouts  
- `model_path` — Path where the trained model will be saved  

---

## Output Example

After training, the model compares its predicted call prices with the true Black-Scholes analytical solution at time `t = 0`. A typical output plot shows the learned function overlayed with ground truth.

---

## Background

The Black-Scholes model describes the price of a European call option as a solution to the following partial differential equation:

```
∂C/∂t + 0.5 * σ² * S² * ∂²C/∂S² + r * S * ∂C/∂S - r * C = 0
```

Where:
- `C` is the call option price  
- `S` is the stock price  
- `σ` is the volatility  
- `r` is the risk-free interest rate  
- `t` is time to maturity

This project uses a Physics-Informed Neural Network (PINN) to approximate the solution by minimizing both data error and residuals of the PDE.

---

## Author

Piero Paialunga  
PhD in Aerospace Engineering  
Focused on AI for Physics, Finance, and Engineering Problems
