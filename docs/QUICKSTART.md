# Quick Start Guide

## Installation (5 minutes)

### Option 1: Automated Setup (Recommended)

```bash
# Clone repository
git clone <repository-url>
cd tensor_networks_finance

# Run setup script
chmod +x setup.sh
./setup.sh

# Activate environment
source venv/bin/activate
```

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Verify Installation

```bash
python -c "import tt; import torch; import tensorly; print('✓ All packages installed')"
```

## Your First Example (2 minutes)

### Option A: Jupyter Notebook (Interactive)

```bash
jupyter lab
```

Open `notebooks/01_basket_options.ipynb` and run all cells.

### Option B: Command Line (Quick)

```bash
cd examples
python basket_option_standalone.py
```

## What You'll See

The script will:
1. Configure an 8-asset basket option
2. Build a Tensor Train representation
3. Show compression ratio (typically 10^12 to 1)
4. Validate accuracy (errors < 10^-4)
5. Demonstrate fast re-pricing (< 1 ms per scenario)

**Expected output**:
```
Configuration:
  Number of assets: 8
  Strike price: $100
  Grid points per asset: 64

Traditional Grid Method:
  Total points: 2.81e+14
  Memory required: 2.01e+06 GB
  Status: ✗ Impossible

Building Tensor Train approximation...
✓ Tensor Train built successfully!

Compression Analysis:
  TT parameters: 15,234
  Compression ratio: 1.85e+10x
  Bond dimensions: [1, 3, 4, 4, 4, 3, 2, 1]
  Max rank: 4
```

## Next Steps

### 1. Explore Notebooks (30-60 minutes each)

**Notebook 1**: `01_basket_options.ipynb`
- High-dimensional derivative pricing
- Understanding bond dimensions
- Compression analysis

**Notebook 2**: `02_lmm_and_cva.ipynb`
- Interest rate models
- CVA without Monte Carlo
- Exposure profiles

**Notebook 3**: `03_tensor_neural_networks.ipynb`
- ML compression
- Overfitting prevention
- Automatic differentiation for Greeks

### 2. Try Your Own Problems

```python
from src.tensor_networks import BasketOptionTN

# Define your option
basket = BasketOptionTN(
    num_assets=12,        # Try 10-15 assets
    grid_size=64,
    strike=105.0,
    price_range=(80.0, 120.0)
)

# Build tensor
tt = basket.build(eps=1e-4)

# Price it
market_state = [60, 58, 62, 59, 61, 57, 63, 56, 64, 55, 65, 54]
price = basket.evaluate(market_state)
```

### 3. Understand the Theory

Read `docs/OVERVIEW.md` for:
- Mathematical foundations
- Why it works for finance
- Implementation details
- Production considerations

## Common Issues

### Import Error: ttpy not found

```bash
pip install ttpy
```

### Import Error: torch not found

```bash
pip install torch
```

### Memory Error

Reduce `grid_size` or `num_assets`:
```python
basket = BasketOptionTN(num_assets=5, grid_size=32, strike=100)
```

### Convergence Issues

Increase number of sweeps:
```python
tt = basket.build(eps=1e-4, nswp=10)
```

## Understanding the Output

### Bond Dimensions

```
Bond dimensions: [1, 3, 4, 4, 4, 3, 2, 1]
```

- Start at 1 (single value)
- Peak in middle (max complexity)
- End at 1 (single result)
- **Low values (2-5) indicate good compression**

### Compression Ratio

```
Compression ratio: 1.85e+10x
```

This is **real compression**:
- Full grid: 281 trillion numbers
- Tensor train: 15,234 numbers
- Ratio: 18 billion to 1

### Validation Errors

```
Mean error: 0.000012
Max error: 0.000089
```

- Typical: 10^-4 to 10^-6
- Controlled by `eps` parameter
- Trade-off: Lower `eps` → more parameters

## Performance Tips

### Speed Up Building

1. **Use GPU** (if available):
```python
import tensorly
tensorly.set_backend('pytorch')
```

2. **Reduce accuracy requirement**:
```python
tt = basket.build(eps=1e-3)  # Faster but less accurate
```

3. **Start with fewer assets**:
```python
# Prototype with 5 assets, then scale to 10-15
```

### Speed Up Evaluation

1. **Batch pricing**:
```python
# Price multiple scenarios at once
scenarios = [state1, state2, state3, ...]
prices = [basket.evaluate(s) for s in scenarios]
```

2. **Cache tensors**:
```python
# Build once, reuse many times
tt = basket.build(eps=1e-4)
# ... use tt for thousands of pricings
```

## Troubleshooting

### Build takes too long

**Causes**:
- Too many assets (>15)
- Too many grid points (>100)
- Target accuracy too tight (<1e-6)

**Solutions**:
- Reduce `grid_size` to 32 or 48
- Increase `eps` to 1e-3
- Reduce `num_assets` for testing

### High bond dimensions (>20)

**Causes**:
- Payoff has no structure to compress
- Wrong problem for tensor networks

**Solutions**:
- Check if payoff is suitable (sums, averages work best)
- Try different problem formulation
- Consider if d < 10 (may not need TNs)

### Validation errors too large

**Causes**:
- Insufficient sweeps
- Grid size too coarse
- Numerical precision issues

**Solutions**:
- Increase `nswp` to 10-20
- Increase `grid_size`
- Check for numerical overflow (extreme values)

## Comparing to Monte Carlo

```python
import time

# Tensor Network
start = time.time()
tt_price = basket.evaluate(state)
tn_time = time.time() - start

# Monte Carlo (pseudo-code)
start = time.time()
mc_price = monte_carlo_price(state, n_paths=100000)
mc_time = time.time() - start

print(f"TN: {tt_price:.6f} in {tn_time*1000:.2f} ms")
print(f"MC: {mc_price:.6f} in {mc_time*1000:.2f} ms")
print(f"Speedup: {mc_time/tn_time:.0f}x")
```

**Expected results**:
- TN: ~0.1-1 ms
- MC: ~100-1000 ms
- Speedup: 100-10000x

## Learning Path

### Beginner (Day 1)
1. ✓ Run `basket_option_standalone.py`
2. ✓ Read notebook 1
3. ✓ Try different parameters

### Intermediate (Week 1)
1. Complete all 3 notebooks
2. Implement simple custom payoff
3. Read theory in `OVERVIEW.md`

### Advanced (Month 1)
1. Implement LMM from scratch
2. Add correlation handling
3. Integrate with existing pricer
4. Explore production deployment

## Getting Help

### Documentation
- `README.md`: Project overview
- `docs/OVERVIEW.md`: Technical deep dive
- `docs/REFERENCES.md`: Papers and resources

### Code Examples
- `notebooks/`: Interactive tutorials
- `examples/`: Standalone scripts
- `src/`: Core implementations

### Community
- GitHub Issues: Technical questions
- Discussions: Implementation advice

## Key Concepts to Remember

1. **Bond Dimension = Complexity**
   - Low (2-5): Great compression ✓
   - Medium (5-20): Still good
   - High (>50): Compression failing

2. **TNs Love Structure**
   - Sums, averages: Perfect
   - Barriers, max/min: Good
   - Random, chaotic: Bad

3. **Build Once, Use Many**
   - Upfront cost to build
   - Fast repeated evaluation
   - Best for parameter sweeps

4. **Not a Universal Solution**
   - Only for d > 10 typically
   - Needs structured payoffs
   - Not for one-off calculations

## Success Checklist

Before moving to production:
- [ ] Understand bond dimensions
- [ ] Validated against MC or known values
- [ ] Tested edge cases
- [ ] Profiled memory usage
- [ ] Documented parameter choices
- [ ] Set up error monitoring

## What's Next?

After mastering basics:

1. **Explore CVA**: Notebook 2 shows risk management
2. **Try Neural Networks**: Notebook 3 shows ML compression
3. **Read Papers**: See `REFERENCES.md`
4. **Contribute**: Add new examples, fix bugs
5. **Deploy**: Production integration guide in `OVERVIEW.md`

## Quick Reference Commands

```bash
# Setup
./setup.sh
source venv/bin/activate

# Run examples
python examples/basket_option_standalone.py

# Start Jupyter
jupyter lab

# Run tests (if implemented)
pytest tests/

# Deactivate environment
deactivate
```

---

**Time investment**:
- Setup: 5 minutes
- First example: 2 minutes
- Understand basics: 1 hour
- Master package: 1 day
- Production ready: 1 week

**Questions?** Open an issue on GitHub or check the documentation!
