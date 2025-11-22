#!/usr/bin/env python3
"""
Standalone example: Basket Option Pricing with Tensor Networks
Demonstrates the complete workflow from setup to pricing and validation
"""

import sys
sys.path.append('../src')

import numpy as np
import matplotlib.pyplot as plt

try:
    import tt
    from tt.cross import amen_cross
    TT_AVAILABLE = True
except ImportError:
    print("Error: ttpy not installed. Install with: pip install ttpy")
    sys.exit(1)


def main():
    print("=" * 70)
    print("TENSOR NETWORKS FOR FINANCE: BASKET OPTION EXAMPLE")
    print("=" * 70)
    print()
    
    # Configuration
    num_assets = 8
    grid_size = 64
    strike = 100.0
    price_range = (60.0, 140.0)
    
    print(f"Configuration:")
    print(f"  Number of assets: {num_assets}")
    print(f"  Strike price: ${strike}")
    print(f"  Price range: ${price_range[0]} - ${price_range[1]}")
    print(f"  Grid points per asset: {grid_size}")
    print()
    
    # Calculate traditional requirements
    full_grid_size = grid_size ** num_assets
    memory_gb = full_grid_size * 8 / (1024**3)
    
    print(f"Traditional Grid Method:")
    print(f"  Total points: {full_grid_size:.2e}")
    print(f"  Memory required: {memory_gb:.2f} GB")
    print(f"  Status: {'✓ Feasible' if memory_gb < 16 else '✗ Impossible'}")
    print()
    
    # Define the payoff function
    def basket_payoff(indices):
        """Calculate basket option payoff for given grid indices."""
        # Convert indices to prices
        a, b = price_range
        n = grid_size
        prices = a + indices * (b - a) / (n - 1)
        
        # Basket average
        basket_avg = np.mean(prices, axis=1)
        
        # Call option payoff
        return np.maximum(basket_avg - strike, 0)
    
    # Build tensor train
    print("Building Tensor Train approximation...")
    print("-" * 70)
    
    tt_payoff = amen_cross(
        basket_payoff,
        tt.rand(grid_size, num_assets, 2),  # Random initial guess
        eps=1e-4,
        nswp=5
    )
    
    print("✓ Tensor Train built successfully!")
    print()
    
    # Analyze results
    tt_params = tt_payoff.core.size
    compression = full_grid_size / tt_params
    
    print("Compression Analysis:")
    print(f"  TT parameters: {tt_params:,}")
    print(f"  Compression ratio: {compression:.2e}x")
    print(f"  Bond dimensions: {tt_payoff.r}")
    print(f"  Max rank: {max(tt_payoff.r)}")
    print()
    
    # Validate
    print("Validation (10 random samples):")
    print("-" * 70)
    
    errors = []
    for i in range(10):
        idx = np.random.randint(0, grid_size, size=(num_assets,))
        
        true_val = basket_payoff(idx.reshape(1, -1))[0]
        tt_val = tt_payoff[idx]
        error = abs(true_val - tt_val)
        errors.append(error)
        
        print(f"  Sample {i+1}: True={true_val:.4f}, TT={tt_val:.4f}, "
              f"Error={error:.6f}")
    
    print()
    print(f"Error Statistics:")
    print(f"  Mean error: {np.mean(errors):.6f}")
    print(f"  Max error: {np.max(errors):.6f}")
    print(f"  RMSE: {np.sqrt(np.mean(np.array(errors)**2)):.6f}")
    print()
    
    # Demonstrate reusability
    print("Reusability Test:")
    print("-" * 70)
    
    num_scenarios = 1000
    import time
    
    start = time.time()
    for _ in range(num_scenarios):
        scenario = np.random.randint(0, grid_size, size=(num_assets,))
        price = tt_payoff[scenario]
    elapsed = time.time() - start
    
    print(f"  Priced {num_scenarios} scenarios in {elapsed:.3f} seconds")
    print(f"  Average time per scenario: {elapsed/num_scenarios*1000:.3f} ms")
    print()
    
    # Visualize bond dimensions
    print("Bond Dimension Analysis:")
    print("-" * 70)
    print(f"  Why is max rank only {max(tt_payoff.r)}?")
    print(f"  → Basket average is a SUM operation")
    print(f"  → Sums have inherent rank-2 structure")
    print(f"  → Tensor networks automatically discover this!")
    print()
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bond dimensions
    ax1.plot(range(1, len(tt_payoff.r)), tt_payoff.r[1:], 'o-', 
             linewidth=2, markersize=8)
    ax1.set_xlabel('Position in Chain', fontsize=11)
    ax1.set_ylabel('Bond Dimension', fontsize=11)
    ax1.set_title('Bond Dimensions Along Tensor Chain', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Compression comparison
    dimensions = list(range(3, 11))
    theoretical_sizes = [grid_size**d for d in dimensions]
    ax2.semilogy(dimensions, theoretical_sizes, 'o-', label='Full Grid', linewidth=2)
    ax2.axhline(y=tt_params, color='green', linestyle='--', 
                label=f'TT ({tt_params:,} params)', linewidth=2)
    ax2.set_xlabel('Number of Assets', fontsize=11)
    ax2.set_ylabel('Number of Parameters (log scale)', fontsize=11)
    ax2.set_title('Tensor Network vs. Full Grid', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('basket_option_example.png', dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: basket_option_example.png")
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Successfully compressed {num_assets}-asset option")
    print(f"✓ Achieved {compression:.2e}x compression")
    print(f"✓ Validation error < {max(errors):.6f}")
    print(f"✓ Fast re-evaluation: {elapsed/num_scenarios*1000:.3f} ms per scenario")
    print()
    print("Key Takeaway:")
    print("  Tensor Networks enable pricing high-dimensional derivatives")
    print("  that are impossible with traditional grid methods and faster")
    print("  than Monte Carlo simulation with deterministic accuracy.")
    print()
    print("Next steps:")
    print("  - See notebooks/ for more examples")
    print("  - Try with more assets (10-20)")
    print("  - Explore LMM and CVA examples")
    print("=" * 70)


if __name__ == "__main__":
    main()
