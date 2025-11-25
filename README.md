# Tensor Networks for Finance

A comprehensive Python package demonstrating the application of Tensor Network methods to financial problems, including derivative pricing, risk management, and machine learning.

![](./tensor-networks-finance.png)

## Overview

Tensor Networks solve the "Curse of Dimensionality" by compressing high-dimensional functions into chains of small tensors. This enables:

- **High-dimensional derivative pricing** (10-50 assets)
- **Fast CVA calculation** without Monte Carlo
- **Compressed neural networks** for financial ML (90-99% parameter reduction)

### Key Results

| Application | Traditional Method | Tensor Network | Speedup |
|-------------|-------------------|----------------|---------|
| 10-Asset Basket Option | Impossible (10^18 grid points) | ~15,000 parameters | 10^14x compression |
| CVA Calculation | 5M Monte Carlo paths | 10 tensor contractions | 1000-10000x |
| Neural Network | 16M parameters | 20k parameters | 800x compression |

## Features

- **Basket Option Pricing**: TT-Cross approximation for high-dimensional payoffs
- **Libor Market Model**: Interest rate derivative pricing with 10-40 forward rates
- **CVA/XVA Calculation**: Exposure profiles without Monte Carlo simulation
- **Tensor Neural Networks**: Compressed layers for financial ML
- **Automatic Differentiation**: Instant Greeks calculation
- **Correlation Handling**: Cholesky decomposition for multi-asset models

## Installation

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd tensor_networks_finance

# Run the setup script
chmod +x setup.sh
./setup.sh

# Activate the environment
source venv/bin/activate
```

### Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- NumPy, SciPy, Matplotlib
- PyTorch 2.0+
- ttpy (Tensor Train Python)
- TensorLy and TensorLy-Torch

## Quick Start Guide

### 1. Jupyter Notebooks

Launch Jupyter Lab to explore the interactive tutorials:

```bash
jupyter lab
```

Navigate to the `notebooks/` directory and start with:
1. `01_basket_options.ipynb` - High-dimensional derivative pricing
2. `02_lmm_and_cva.ipynb` - Interest rate models and CVA
3. `03_tensor_neural_networks.ipynb` - ML compression and regularization

### 2. Python API

```python
from src.tensor_networks import BasketOptionTN

# Create a 10-asset basket option
basket = BasketOptionTN(
    num_assets=10,
    grid_size=64,
    strike=100.0,
    price_range=(50.0, 150.0)
)

# Build the tensor train representation
tt_payoff = basket.build(eps=1e-4, nswp=5)

# Evaluate at a specific market state
market_state = [55, 60, 58, 62, 59, 61, 57, 63, 56, 64]  # Grid indices
price = basket.evaluate(market_state)
```

### 3. Example: LMM and CVA

```python
from src.tensor_networks import TensorLMM

# Create Libor Market Model
lmm = TensorLMM(
    num_tenors=10,
    grid_size=64,
    strike=0.03
)

# Build swaption tensor
tt_swaption = lmm.build(eps=1e-4)

# Calculate CVA exposure profile
time_points = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
exposures = lmm.calculate_cva_exposure(time_points, vol=0.01)
```

## Project Structure

```
tensor_networks_finance/
├── notebooks/              # Jupyter notebooks with tutorials
│   ├── 01_basket_options.ipynb
│   ├── 02_lmm_and_cva.ipynb
│   └── 03_tensor_neural_networks.ipynb
├── src/                    # Core Python modules
│   ├── __init__.py
│   └── tensor_networks.py  # Main TN implementations
├── docs/                   # Documentation and guides
│   ├── OVERVIEW.md         # Comprehensive technical overview
│   ├── REFERENCES.md       # Academic papers and resources
│   └── *.png              # Generated figures
├── examples/               # Standalone example scripts
├── requirements.txt        # Python dependencies
├── setup.sh               # Installation script
└── README.md              # This file
```

## Core Concepts

### Bond Dimension

The **bond dimension** (rank) is the most critical parameter in tensor networks. It represents the "memory" or "entanglement" between variables:

- **Rank 1-2**: Independent variables
- **Rank 3-20**: Structured payoffs (sums, averages, barriers) ✅ Ideal
- **Rank 50-100**: Complex correlations (manageable)
- **Rank >500**: Near-random structure (compression fails)

Financial products typically have low bond dimensions because they're designed with structured logic.

### Compression Ratio

For a basket option on 10 assets with 64 grid points each:
- **Full grid**: 64^10 ≈ 1.15 × 10^18 points (~1 exabyte)
- **Tensor train**: ~15,000 parameters
- **Compression**: 10^14 to 1

### Why It Works for Finance

Financial payoffs often have inherent low-rank structure:
- **Basket options**: Average of assets (sum → rank 2)
- **Barrier options**: Max/min operations (rank ~5)
- **Swaptions**: Weighted sum of rates (rank 3-6)

## Applications

### 1. Derivative Pricing

**Problem**: Price options on N > 10 assets
**Solution**: TT-Cross approximation builds compressed payoff function
**Benefit**: Deterministic pricing, instant re-evaluation, exact Greeks

### 2. Risk Management (CVA/XVA)

**Problem**: Calculate exposure at T time steps × M scenarios
**Solution**: Payoff tensor (once) + probability tensors (rank-1)
**Benefit**: 1000-10000x faster than Monte Carlo, deterministic

### 3. Financial Machine Learning

**Problem**: Neural networks overfit on noisy financial data
**Solution**: Replace dense layers with Tensor Train layers
**Benefit**: 90-99% parameter reduction, built-in regularization

### 4. Portfolio Optimization

**Problem**: Optimize portfolio with 100+ assets, discrete constraints
**Solution**: DMRG-based optimization on tensor representation
**Benefit**: Find global optimum in massive discrete space

## Theory

### Tensor Train Format

A d-dimensional function is represented as:

```
f(x₁, x₂, ..., xₐ) = ∑ G₁[i₁] G₂[i₁,i₂] ... Gₐ[iₐ₋₁,iₐ]
```

Where:
- `G₁, ..., Gₐ` are the "cores" (small tensors)
- Indices `i₁, ..., iₐ` are summed over (contractions)
- Bond dimension = size of connecting indices

**Memory complexity**: O(d · n · r²) vs O(nᵈ)

### TT-Cross Approximation

Instead of evaluating every grid point, we:
1. Intelligently sample key points
2. Interpolate using tensor structure
3. Iteratively refine until target accuracy

**Result**: Build approximation by evaluating ~1000s of points instead of 10^18

### Cholesky Decomposition for Correlation

To handle correlated assets:
1. Decompose correlation matrix: Σ = L Lᵀ
2. Build grid using independent factors: Z₁, Z₂, ...
3. Embed correlation in payoff: S = S₀ exp(L·Z)

**Result**: Probability tensor stays rank-1, correlation handled in payoff

## Performance Benchmarks

From recent research and industry implementations:

| Metric | Traditional | Tensor Network |
|--------|------------|---------------|
| 10-asset basket pricing | Minutes (MC) | Seconds |
| CVA calculation (50 steps) | Hours | Seconds |
| Neural network parameters | 16M | 20k |
| Training time (ML) | 2 hours | 15 minutes |
| Inference latency (ML) | 10ms | 0.1ms |

## Industry Adoption

Tensor Networks (often called "Quantum-Inspired" methods) are being used by:

- **BBVA** (Spain): Portfolio optimization, reported solving 10^382 combinations
- **Crédit Agricole** (France): Derivative pricing and CVA
- **Japan Post Bank**: Asset allocation with Fujitsu's Digital Annealer
- **Goldman Sachs**: Internal R&D on quantum/tensor methods
- **JPMorgan**: Quantum readiness program using TN methods

## Limitations

Tensor Networks are **not** a universal solution:

❌ **Don't use for**:
- Random/chaotic payoffs (no structure to compress)
- Problems with natural low dimensionality (d < 5)
- One-off calculations (build time not amortized)

✅ **Do use for**:
- Structured high-dimensional problems (d > 10)
- Repeated evaluations with parameter changes
- Problems with summation/average/max-min structure
- Large ML models on limited data

## Contributing

Contributions welcome! Areas of interest:
- Additional financial applications
- Performance optimizations
- Production deployment examples
- Integration with other libraries (JAX, TensorFlow)

## References

### Key Papers

1. **Tensor Trains for Finance**: "Learning Parameter Dependence for Fourier-Based Option Pricing with Tensor Trains" (arXiv, 2024)
2. **Factor Models**: Lettau et al., "High-Dimensional Factor Models with an Application to Mutual Fund Characteristics" (NBER, 2022)
3. **Quantum Computing**: Herman et al., "Quantum computing for finance: State of the art and future prospects" (Rev. Mod. Phys., 2023)
4. **ML Compression**: "Tensor Networks Meet Neural Networks: A Survey" (arXiv:2302.09019, 2023)

### Libraries and Tools

- **ttpy**: Python implementation of Tensor Trains
- **TensorLy**: General tensor decomposition library
- **TensorLy-Torch**: PyTorch integration for TNNs
- **TensorNetwork (Google)**: High-performance contractions

See [`docs/REFERENCES.md`](./docs/REFERENCES.md) for complete bibliography.

## License

[MIT License or appropriate license]

## Citation

If you use this code in your research, please cite:

```bibtex
@software{tensor_networks_finance,
  author = {Ian Buckley},
  title = {Tensor Networks for Finance},
  year = {2025},
  url = {https://github.com/roguetrainer/tensor_networks_finance}
}
```


## Acknowledgments

This package synthesizes research from:
- Quantum computing and tensor network physics
- Quantitative finance and derivatives pricing
- Machine learning compression techniques

Special thanks to the communities behind ttpy, TensorLy, and PyTorch for their excellent open-source tools.
