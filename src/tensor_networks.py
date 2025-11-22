"""
Tensor Networks for Finance
Core utilities for tensor network operations in financial applications
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Callable

try:
    import tt
    from tt.cross import amen_cross
    TT_AVAILABLE = True
except ImportError:
    TT_AVAILABLE = False
    print("Warning: ttpy not available. Some functionality will be limited.")


class BasketOptionTN:
    """
    Tensor Network representation of a basket option.
    Uses Tensor Train (TT) format to compress high-dimensional payoff functions.
    """
    
    def __init__(self, num_assets: int, grid_size: int, strike: float, 
                 price_range: Tuple[float, float] = (50.0, 150.0)):
        """
        Initialize the Basket Option Tensor Network.
        
        Args:
            num_assets: Number of assets in the basket (dimensions)
            grid_size: Number of grid points per asset
            strike: Strike price
            price_range: Tuple of (min_price, max_price) for the grid
        """
        self.d = num_assets
        self.n = grid_size
        self.K = strike
        self.a, self.b = price_range
        self.tt_payoff = None
        
        if not TT_AVAILABLE:
            raise ImportError("ttpy is required for BasketOptionTN. Install with: pip install ttpy")
    
    def index_to_price(self, indices: np.ndarray) -> np.ndarray:
        """Convert grid indices to actual asset prices."""
        return self.a + indices * (self.b - self.a) / (self.n - 1)
    
    def payoff_function(self, indices: np.ndarray) -> np.ndarray:
        """
        Calculate the basket option payoff for given grid coordinates.
        
        Args:
            indices: Array of shape (batch_size, num_assets) with grid indices
            
        Returns:
            Array of payoffs: max(average(prices) - strike, 0)
        """
        prices = self.index_to_price(indices)
        basket_avg = np.mean(prices, axis=1)
        return np.maximum(basket_avg - self.K, 0)
    
    def build(self, eps: float = 1e-4, nswp: int = 5) -> 'tt.tensor':
        """
        Build the Tensor Train representation using cross approximation.
        
        Args:
            eps: Target accuracy tolerance
            nswp: Number of optimization sweeps
            
        Returns:
            The compressed tensor train object
        """
        print(f"Building TT for {self.d}-dimensional basket option...")
        print(f"Grid size: {self.n}^{self.d} = {self.n**self.d:.2e} points")
        
        self.tt_payoff = amen_cross(
            self.payoff_function,
            tt.rand(self.n, self.d, 2),
            eps=eps,
            nswp=nswp
        )
        
        # Calculate compression metrics
        full_size = self.n ** self.d
        tt_size = self.tt_payoff.core.size
        compression = full_size / tt_size
        
        print(f"✓ TT built successfully")
        print(f"  Bond dimension: {self.tt_payoff.r}")
        print(f"  TT parameters: {tt_size:,}")
        print(f"  Compression ratio: {compression:.2e}x")
        
        return self.tt_payoff
    
    def evaluate(self, indices: np.ndarray) -> float:
        """Evaluate the TT approximation at specific grid indices."""
        if self.tt_payoff is None:
            raise ValueError("Must call build() before evaluate()")
        return self.tt_payoff[indices]
    
    def validate(self, num_samples: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validate the TT approximation against true values.
        
        Returns:
            Tuple of (true_values, tt_values)
        """
        if self.tt_payoff is None:
            raise ValueError("Must call build() before validate()")
            
        true_vals = []
        tt_vals = []
        
        for _ in range(num_samples):
            idx = np.random.randint(0, self.n, size=(self.d,))
            true_val = self.payoff_function(idx.reshape(1, -1))[0]
            tt_val = self.tt_payoff[idx]
            
            true_vals.append(true_val)
            tt_vals.append(tt_val)
        
        return np.array(true_vals), np.array(tt_vals)


class TensorLMM:
    """
    Libor Market Model using Tensor Networks.
    Represents interest rate derivatives in high-dimensional rate spaces.
    """
    
    def __init__(self, num_tenors: int, grid_size: int, strike: float,
                 rate_range: Tuple[float, float] = (0.001, 0.080),
                 tenor_fraction: float = 0.5):
        """
        Initialize the Tensor LMM.
        
        Args:
            num_tenors: Number of forward rate periods
            grid_size: Grid points per rate
            strike: Swaption strike rate
            rate_range: (min_rate, max_rate) bounds
            tenor_fraction: Fraction of year per tenor (0.5 = semi-annual)
        """
        self.d = num_tenors
        self.n = grid_size
        self.K = strike
        self.a, self.b = rate_range
        self.delta = tenor_fraction
        self.tt_payoff = None
        
        if not TT_AVAILABLE:
            raise ImportError("ttpy is required for TensorLMM")
    
    def index_to_rate(self, indices: np.ndarray) -> np.ndarray:
        """Convert grid indices to forward rates."""
        return self.a + indices * (self.b - self.a) / (self.n - 1)
    
    def swap_rate_payoff(self, indices: np.ndarray) -> np.ndarray:
        """
        Calculate payer swaption payoff.
        Payoff = max(swap_rate - strike, 0)
        """
        F = self.index_to_rate(indices)
        
        # Simplified swap rate calculation
        # In practice, would use proper discount factors
        numerator = np.sum(F * self.delta, axis=1)
        denominator = np.sum(np.ones_like(F) * self.delta, axis=1)
        swap_rate = numerator / denominator
        
        return np.maximum(swap_rate - self.K, 0)
    
    def build(self, eps: float = 1e-4, nswp: int = 5):
        """Build the Tensor Train representation of the swaption."""
        print(f"Building Tensor LMM for {self.d} tenors...")
        
        self.tt_payoff = amen_cross(
            self.swap_rate_payoff,
            tt.rand(self.n, self.d, 2),
            eps=eps,
            nswp=nswp
        )
        
        print(f"✓ Tensor LMM built")
        print(f"  Max rank: {max(self.tt_payoff.r)}")
        print(f"  Parameters: {self.tt_payoff.core.size:,}")
        
        return self.tt_payoff
    
    def calculate_cva_exposure(self, time_steps: List[float], 
                               vol: float = 0.01) -> List[float]:
        """
        Calculate CVA exposure profile without Monte Carlo.
        
        Args:
            time_steps: List of future time points
            vol: Interest rate volatility
            
        Returns:
            List of expected exposures at each time step
        """
        if self.tt_payoff is None:
            raise ValueError("Must call build() before calculating CVA")
        
        exposures = []
        mu = 0.04  # Initial rate assumption
        
        # Pre-compute grid
        grid_values = np.linspace(self.a, self.b, self.n)
        
        for t in time_steps:
            if t == 0:
                exposures.append(0.0)
                continue
            
            # Build probability distribution at time t
            sigma = vol * np.sqrt(t)
            pdf_1d = np.exp(-0.5 * ((grid_values - mu) / sigma) ** 2)
            pdf_1d /= np.sum(pdf_1d)
            
            # Create rank-1 joint PDF (assumes independence)
            pdf_vectors = [tt.tensor(pdf_1d) for _ in range(self.d)]
            probability_tensor = tt.mkron(pdf_vectors)
            
            # Calculate expected exposure via tensor contraction
            ee = tt.dot(self.tt_payoff, probability_tensor)
            exposures.append(float(ee))
            
            print(f"Time {t:.1f}y: Expected Exposure = {ee:.6f}")
        
        return exposures


class TensorLayer(nn.Module):
    """
    Tensor Train layer for PyTorch neural networks.
    Provides massive parameter compression for financial ML models.
    
    Note: For production use, consider tensorly-torch which provides
    more optimized implementations.
    """
    
    def __init__(self, input_dims: Tuple[int, ...], output_dims: Tuple[int, ...],
                 rank: int = 8):
        """
        Initialize a factorized linear layer.
        
        Args:
            input_dims: Tuple describing input tensor shape
            output_dims: Tuple describing output tensor shape
            rank: Bond dimension (controls compression)
        """
        super().__init__()
        
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.rank = rank
        self.num_dims = len(input_dims)
        
        # Validate dimensions
        if len(input_dims) != len(output_dims):
            raise ValueError("Input and output must have same number of dimensions")
        
        # Initialize TT cores
        self.cores = nn.ParameterList()
        
        for i in range(self.num_dims):
            # First core
            if i == 0:
                core_shape = (input_dims[i], output_dims[i], rank)
            # Last core
            elif i == self.num_dims - 1:
                core_shape = (rank, input_dims[i], output_dims[i])
            # Middle cores
            else:
                core_shape = (rank, input_dims[i], output_dims[i], rank)
            
            core = torch.randn(core_shape) * 0.01
            self.cores.append(nn.Parameter(core))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the tensor layer.
        
        Args:
            x: Input tensor of shape (batch, prod(input_dims))
            
        Returns:
            Output tensor of shape (batch, prod(output_dims))
        """
        batch_size = x.shape[0]
        
        # Reshape input to factorized form
        x = x.view(batch_size, *self.input_dims)
        
        # Contract through TT cores
        # This is a simplified implementation
        # For production, use optimized tensor contraction libraries
        
        result = x
        for i, core in enumerate(self.cores):
            # Perform contraction (simplified)
            if i == 0:
                result = torch.einsum('b...i,ior->b...or', result, core)
            elif i == self.num_dims - 1:
                result = torch.einsum('b...ri,rio->b...o', result, core)
            else:
                result = torch.einsum('b...ri,rior->b...or', result, core)
        
        # Flatten back to 2D
        output_size = np.prod(self.output_dims)
        return result.view(batch_size, output_size)
    
    def parameter_count(self) -> int:
        """Calculate total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def compare_layer_sizes(input_size: int, output_size: int, 
                       factorization_dims: Tuple[int, ...],
                       rank: int = 8) -> dict:
    """
    Compare parameter counts between dense and tensor layers.
    
    Args:
        input_size: Flattened input dimension
        output_size: Flattened output dimension
        factorization_dims: Tuple for reshaping (must multiply to input_size)
        rank: Tensor train rank
        
    Returns:
        Dictionary with comparison metrics
    """
    # Dense layer
    dense_params = input_size * output_size + output_size
    
    # Tensor layer (approximate)
    num_dims = len(factorization_dims)
    tensor_params = 0
    for i in range(num_dims):
        if i == 0:
            tensor_params += factorization_dims[i] * factorization_dims[i] * rank
        elif i == num_dims - 1:
            tensor_params += rank * factorization_dims[i] * factorization_dims[i]
        else:
            tensor_params += rank * factorization_dims[i] * factorization_dims[i] * rank
    
    compression_ratio = dense_params / tensor_params
    
    return {
        'dense_params': dense_params,
        'tensor_params': tensor_params,
        'compression_ratio': compression_ratio,
        'memory_saved_mb': (dense_params - tensor_params) * 4 / (1024**2)
    }


def cholesky_correlation_transform(correlation_matrix: np.ndarray) -> np.ndarray:
    """
    Perform Cholesky decomposition for correlation handling in TNs.
    
    Args:
        correlation_matrix: Symmetric positive definite correlation matrix
        
    Returns:
        Lower triangular matrix L where Σ = L @ L.T
    """
    return np.linalg.cholesky(correlation_matrix)


def independent_to_correlated(z_factors: np.ndarray, 
                              cholesky_matrix: np.ndarray,
                              initial_prices: np.ndarray,
                              drift: float = 0.0) -> np.ndarray:
    """
    Transform independent factors to correlated asset prices.
    
    Args:
        z_factors: Independent standard normal factors
        cholesky_matrix: Cholesky decomposition of correlation matrix
        initial_prices: Starting prices for assets
        drift: Drift term
        
    Returns:
        Correlated asset prices
    """
    # Apply correlation structure: ln(S) = ln(S0) + drift + L @ Z
    log_returns = drift + cholesky_matrix @ z_factors
    return initial_prices * np.exp(log_returns)


if __name__ == "__main__":
    print("Tensor Networks for Finance - Core Module")
    print("=" * 50)
    print("\nAvailable classes:")
    print("  - BasketOptionTN: Basket option pricing")
    print("  - TensorLMM: Libor Market Model")
    print("  - TensorLayer: Compressed neural network layer")
    print("\nAvailable functions:")
    print("  - compare_layer_sizes: Compare dense vs tensor layers")
    print("  - cholesky_correlation_transform: Handle asset correlations")
    print("  - independent_to_correlated: Transform factors to prices")
