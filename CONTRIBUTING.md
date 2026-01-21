# Contributing to RFX

Thank you for your interest in contributing to RFX! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project follows a simple code of conduct: be respectful, be constructive, and be collaborative. We welcome contributors of all experience levels.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/RFX.git
   cd RFX
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/chriskuchar/RFX.git
   ```

## How to Contribute

### Types of Contributions

| Type | Description |
|------|-------------|
| Bug fixes | Fix issues in existing code |
| Features | Add new functionality |
| Documentation | Improve docs, examples, comments |
| Tests | Add or improve test coverage |
| Performance | Optimize existing code |

### What We're Looking For

- Bug fixes with clear reproduction steps
- Performance improvements with benchmarks
- New features that align with the project roadmap
- Documentation improvements and examples
- Test coverage for untested code paths

## Development Setup

### Prerequisites

- Python 3.7+
- CMake 3.12+
- C++17 compiler (GCC 7+ or Clang 5+)
- CUDA 11.0+ (for GPU development)

### Build from Source

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[dev]"

# Build C++/CUDA extensions
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_classification.py

# Run with coverage
pytest --cov=RFX tests/
```

## Coding Standards

### Python

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints for function signatures
- Write docstrings for public functions (NumPy style)

```python
def compute_importance(self, X: np.ndarray) -> np.ndarray:
    """
    Compute feature importance scores.
    
    Parameters
    ----------
    X : np.ndarray
        Input features of shape (n_samples, n_features).
    
    Returns
    -------
    np.ndarray
        Importance scores of shape (n_features,).
    """
```

### C++/CUDA

- Use consistent naming: `snake_case` for functions, `CamelCase` for classes
- Add comments for complex algorithms
- Avoid `std::cout` in production code (causes Jupyter crashes)
- Use `cudaStreamSynchronize(0)` instead of `cudaDeviceSynchronize()` for Jupyter safety

```cpp
// Good: Jupyter-safe synchronization
CUDA_CHECK_VOID(cudaStreamSynchronize(0));

// Avoid: Can cause hangs in Jupyter
// cudaDeviceSynchronize();
```

### Commit Messages

Use clear, descriptive commit messages:

```
feat: Add INT8 quantization for low-rank proximity

- Implement compute_int8_scaling_gpu function
- Add persistent buffers to avoid per-tree allocation
- Fix race condition in scale computation
```

Prefixes:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `perf:` Performance improvement
- `test:` Tests
- `refactor:` Code refactoring

## Pull Request Process

### Before Submitting

1. **Sync with upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests** and ensure they pass

3. **Update documentation** if needed

4. **Add tests** for new functionality

### Submitting

1. Push to your fork:
   ```bash
   git push origin your-branch-name
   ```

2. Open a Pull Request on GitHub

3. Fill out the PR template with:
   - Description of changes
   - Related issue numbers
   - Testing performed

### Review Process

- PRs require at least one approval before merging
- Address reviewer feedback promptly
- Keep PRs focused and reasonably sized

## Reporting Issues

### Bug Reports

Include:
- RFX version (`rf.__version__`)
- Python version
- CUDA version (if applicable)
- Operating system
- Minimal code to reproduce
- Full error traceback

### Feature Requests

Include:
- Clear description of the feature
- Use case / motivation
- Example of desired API (if applicable)

## Project Structure

```
RFX/
├── cuda/           # CUDA kernels
├── src/            # C++ source files
├── include/        # C++ headers
├── python/         # Python bindings (pybind11)
├── examples/       # Example scripts and notebooks
├── tests/          # Test suite
└── docs/           # Documentation
```

## Questions?

- Open a [GitHub Issue](https://github.com/chriskuchar/RFX/issues)
- Tag with `question` label

---

Thank you for contributing to RFX!

