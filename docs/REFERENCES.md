# References and Resources

## Academic Papers

### Tensor Networks for Finance

1. **Learning Parameter Dependence for Fourier-Based Option Pricing with Tensor Trains**
   - Authors: Various
   - Year: 2024
   - Link: arXiv (search for recent versions)
   - Focus: Basket options, parameter-dependent pricing, speedups over Monte Carlo
   - Key result: 10^3 to 10^4 speedup for 10-11 asset basket options

2. **Tensor Train Methods for Sequential State Estimation**
   - Focus: Bayesian filtering, time series
   - Application: Dynamic portfolio allocation, real-time risk

3. **Quantized Tensor Trains for Solving the Vlasov Equation**
   - Application: High-dimensional PDEs similar to financial models
   - Technique: Time evolution using MPOs

### Factor Models and Asset Pricing

4. **High-Dimensional Factor Models with an Application to Mutual Fund Characteristics**
   - Authors: Martin Lettau et al.
   - Year: 2022
   - Source: NBER Working Paper
   - Focus: Using tensor CP decomposition to generalize PCA
   - Key insight: Finding deep latent structures in market data beyond standard PCA

5. **Tensor Methods in Statistics**
   - Authors: Anandkumar et al.
   - Focus: Method of moments, latent variable models
   - Application: Financial factor extraction

### Quantum Computing and Finance

6. **Quantum computing for finance: State of the art and future prospects**
   - Authors: Herman, Shaydulin, Pistoia, et al.
   - Year: 2023
   - Source: Reviews of Modern Physics
   - Focus: Comprehensive review including quantum-inspired classical algorithms
   - Sections on: Portfolio optimization, option pricing, risk analysis

7. **Quantum amplitude estimation in finance**
   - Focus: Monte Carlo acceleration using quantum algorithms
   - Connection: Tensor networks simulate quantum amplitude estimation classically

8. **Variational Quantum Algorithms for Computational Fluid Dynamics**
   - Application: Similar PDEs to Black-Scholes in high dimensions
   - Technique: QAOA-inspired optimization on tensor networks

### Machine Learning with Tensors

9. **Tensor Networks Meet Neural Networks: A Survey**
   - Year: 2023
   - Source: arXiv:2302.09019
   - Focus: Comprehensive survey of TNNs, tensor layers
   - Applications: Time series forecasting, compression
   - Financial ML: Section on overfitting prevention

10. **Tensorizing Neural Networks**
    - Authors: Novikov et al.
    - Focus: TT-layers in fully connected networks
    - Result: 90%+ parameter reduction with maintained accuracy

11. **Tensor Regression Networks**
    - Focus: Low-rank tensor regression for high-dimensional data
    - Financial application: Multi-asset return prediction

### Computational Finance

12. **Deep Stochastic Optimization in Finance**
    - Authors: Mete Soner
    - Year: 2022/2024
    - Focus: Deep learning for optimal control, American options
    - Review of: Tensor methods alongside neural approaches

13. **Tensor Methods for Multidimensional Backward Stochastic Differential Equations**
    - Focus: BSDE solution using tensor trains
    - Application: Hedging, optimal stopping

### Methodology Papers

14. **TT-Cross Approximation for Multidimensional Arrays**
    - Authors: Oseledets, Tyrtyshnikov
    - Focus: The fundamental cross-approximation algorithm
    - Key insight: Sample O(dnr^2) instead of O(n^d) points

15. **Tensor-Train Decomposition**
    - Authors: Oseledets
    - Year: 2011
    - Source: SIAM Journal on Scientific Computing
    - The foundational paper on TT format

16. **Matrix Product States and Projected Entangled Pair States**
    - Focus: Physics origins of tensor networks
    - Connection: Financial applications use identical math

## Industry Reports and Whitepapers

### Bank and Financial Institution Research

17. **BBVA Portfolio Optimization with Quantum-Inspired Methods**
    - Organization: BBVA + Multiverse Computing
    - Result: Solving 10^382 portfolio combinations
    - Method: DMRG-inspired tensor optimization

18. **Crédit Agricole CIB: Derivative Pricing and CVA**
    - Focus: Pilot projects on exotic derivatives
    - Result: "Conclusive results" on speedup and memory reduction

19. **Japan Post Bank: Digital Annealer for Asset Allocation**
    - Partner: Fujitsu
    - Focus: Tensor-like optimization on specialized hardware
    - Result: Improved profit margins on liquid asset portfolio

### Technology Companies

20. **Google TensorNetwork Library Documentation**
    - Source: github.com/google/TensorNetwork
    - Focus: High-performance tensor contractions
    - Backends: TensorFlow, JAX, NumPy

21. **Multiverse Computing: Quantum-Inspired Finance**
    - Various case studies and technical reports
    - Applications: Portfolio optimization, VaR, fraud detection

22. **Terra Quantum: Financial Use Cases**
    - Focus: European banking sector adoption
    - Applications: Credit risk, regulatory compliance

## Books

23. **Tensor Networks for Big Data Analytics and Large-Scale Optimization**
    - Authors: Cichocki et al.
    - Comprehensive textbook covering theory and applications

24. **A Practical Introduction to Tensor Networks**
    - Authors: Bridgeman, Chubb
    - Accessible introduction from physics perspective

25. **Machine Learning with Tensor Networks**
    - Focus: TNN architectures, training algorithms
    - Chapter on financial applications

## Software and Libraries

### Core Tensor Network Libraries

26. **ttpy** (Python)
    - Source: github.com/oseledets/ttpy
    - Features: TT decomposition, cross-approximation, arithmetic
    - Best for: Academic research, algorithm development

27. **TensorLy** (Python)
    - Source: tensorly.org
    - Features: Multiple decompositions (CP, Tucker, TT)
    - Best for: General tensor operations

28. **TensorLy-Torch** (Python/PyTorch)
    - Features: TNN layers, GPU acceleration
    - Best for: Deep learning integration

29. **TensorNetwork** (Python, Google)
    - Source: github.com/google/TensorNetwork
    - Features: High-performance contractions, multiple backends
    - Best for: Production systems, GPU/TPU

30. **ITensor** (C++/Julia)
    - Source: itensor.org
    - Features: DMRG, MPO/MPS
    - Best for: Large-scale physics-style problems

### Deep Learning Frameworks

31. **PyTorch**
    - Automatic differentiation (Greeks calculation)
    - Dynamic computation graphs
    - Ecosystem: torchvision, tensorly-torch

32. **JAX**
    - Functional transformations: grad, jit, vmap
    - XLA compilation for maximum speed
    - Best for: Production inference engines

33. **TensorFlow**
    - Mature ecosystem
    - TensorNetwork library integration
    - TPU support

## Online Resources

### Tutorials and Courses

34. **Matrix Product States Tutorial**
    - Source: tensornetwork.org
    - Focus: MPS/DMRG from physics perspective

35. **Tensor Methods in Machine Learning**
    - Various university courses (MIT, Stanford)
    - Materials often available on YouTube

36. **Quantum Machine Learning Course**
    - Source: PennyLane (Xanadu)
    - Connection: Quantum circuits as tensor networks

### Documentation

37. **ttpy Documentation**
    - Installation, API reference, examples
    - Source: github.com/oseledets/ttpy

38. **TensorLy Tutorials**
    - Comprehensive tutorials on tensor decompositions
    - Source: tensorly.org/stable/

39. **PyTorch Automatic Differentiation Guide**
    - Essential for Greeks calculation
    - Source: pytorch.org/tutorials/

## Conferences and Workshops

### Key Venues

40. **International Conference on Machine Learning (ICML)**
    - Tensor methods in ML track

41. **Neural Information Processing Systems (NeurIPS)**
    - TNN papers, quantum ML workshop

42. **Quantitative Methods in Finance (QMF)**
    - Computational finance track

43. **SIAM Conference on Financial Mathematics**
    - High-dimensional methods sessions

### Quantum Computing Events

44. **Q2B (Quantum for Business)**
    - Industry focus, finance applications

45. **IEEE Quantum Week**
    - Quantum-inspired classical algorithms track

## Community Resources

### Forums and Discussion

46. **Tensor Network Slack/Discord**
    - Active community of researchers

47. **Quantitative Finance Stack Exchange**
    - Questions on high-dimensional pricing methods

48. **GitHub Discussions**
    - Individual project repositories

### Blogs and Articles

49. **Towards Data Science**
    - Various articles on tensor methods in ML

50. **Quantum Computing Report**
    - Industry adoption news, case studies

## Related Mathematical Topics

### Background Reading

51. **Multilinear Algebra**
    - Foundation for tensor operations

52. **Matrix Factorization Methods**
    - SVD, QR, Cholesky (used in correlation handling)

53. **Numerical Linear Algebra**
    - Essential for understanding tensor contractions

54. **Graphical Models**
    - Connection to tensor network diagrams

55. **Quantum Information Theory**
    - Source of many tensor network concepts

## Stay Updated

### Key Researchers to Follow

- Ivan Oseledets (tensor trains)
- Eugene Tyrtyshnikov (approximation theory)
- Andrzej Cichocki (tensor decompositions)
- Miles Stoudenmire (ITensor, DMRG)
- Anima Anandkumar (tensor methods in ML)

### ArXiv Searches

Recommended search terms:
- "tensor train finance"
- "tensor networks option pricing"
- "quantum-inspired finance"
- "tensorized neural networks"
- "high-dimensional PDE finance"

### Industry News Sources

- **QC Ware Newsletter**: Quantum computing in finance
- **Financial Technology Report**: Emerging technologies
- **Risk.net**: Computational finance innovations

## Practical Implementation Guides

### Code Examples

56. **GitHub: tensor-networks-finance**
    - This repository
    - Complete implementations of basket options, LMM, CVA

57. **Google Colab Notebooks**
    - Various public notebooks on tensor methods
    - Search: "tensor train option pricing"

### Benchmarking Studies

58. **Tensor Train vs. Monte Carlo: A Comparison**
    - Various independent studies
    - Typical result: 1000-10000x speedup

59. **TNN Compression Ratios in Practice**
    - Studies on real financial ML models
    - Results: 90-99% parameter reduction

## Citation Guidelines

When citing this work or using tensor networks in finance research:

**For tensor methods generally**:
```bibtex
@article{oseledets2011tensor,
  title={Tensor-train decomposition},
  author={Oseledets, Ivan V},
  journal={SIAM Journal on Scientific Computing},
  year={2011}
}
```

**For TT-Cross**:
```bibtex
@article{oseledets2010tt,
  title={TT-cross approximation for multidimensional arrays},
  author={Oseledets, Ivan V and Tyrtyshnikov, Eugene E},
  journal={Linear Algebra and its Applications},
  year={2010}
}
```

**For financial applications**:
```bibtex
@misc{tensor_networks_finance,
  author = {Ian Hincks},
  title = {Tensor Networks for Finance},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/tensor_networks_finance}
}
```

## Acknowledgments

This resource list builds on work from:
- The quantum physics community (MPS/DMRG origins)
- The numerical linear algebra community (approximation theory)
- The machine learning community (TNN applications)
- The quantitative finance community (real-world problems)

---

**Note**: This is a living document. For updates and corrections, please submit issues or pull requests to the repository.

**Last updated**: November 2024
