# Contributing to BrainTop

Thank you for your interest in contributing to BrainTop! This document provides guidelines and instructions for contributing.

## Getting Started

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/jesusvilela/braintop.git
   cd braintop
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run tests to verify setup**
   ```bash
   pytest tests/ -v
   ```

## Development Workflow

### 1. Create a Branch

Create a feature branch for your work:
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Follow these guidelines:

- **Code Style**: Follow PEP 8 conventions
- **Documentation**: Document all public APIs
- **Type Hints**: Use type hints for function parameters and returns
- **Tests**: Write tests for new functionality

### 3. Run Tests

Before committing, ensure all tests pass:
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=braintop --cov-report=html
```

### 4. Format Code

Format your code using black:
```bash
black src/braintop tests examples
```

Lint with flake8:
```bash
flake8 src/braintop tests examples
```

### 5. Commit Changes

Write clear, descriptive commit messages:
```bash
git add .
git commit -m "Add feature: brief description"
```

### 6. Submit Pull Request

Push your branch and create a pull request:
```bash
git push origin feature/your-feature-name
```

## Code Guidelines

### Module Structure

```python
"""
Module docstring describing purpose.

Detailed explanation if needed.
"""

from typing import Dict, List, Optional
import numpy as np

# Class definitions
class MyClass:
    """Class docstring."""
    
    def __init__(self, param: str):
        """Initialize with clear parameter documentation."""
        pass
```

### Adding New Topology Layers

To add a new topology layer:

1. **Create the layer class**
   ```python
   # src/braintop/layers/my_layer.py
   from ..core.topology import TopologyLayer
   import numpy as np
   
   class MyLayer(TopologyLayer):
       """Brief description of the layer."""
       
       def __init__(self, name: str = "my_layer", **kwargs):
           super().__init__(name, **kwargs)
           # Initialize layer-specific parameters
       
       def compute_positions(self) -> Dict[str, np.ndarray]:
           """Compute 3D positions for nodes."""
           # Implementation
           pass
       
       def compute_geometry(self) -> Dict[str, Any]:
           """Compute geometric properties."""
           # Implementation
           pass
   ```

2. **Add to layers/__init__.py**
   ```python
   from .my_layer import MyLayer
   __all__ = [..., "MyLayer"]
   ```

3. **Write tests**
   ```python
   # tests/test_my_layer.py
   def test_my_layer_creation():
       layer = MyLayer()
       assert layer.name == "my_layer"
   ```

4. **Add example**
   ```python
   # examples/my_layer_example.py
   from braintop import MyLayer, Renderer3D
   # ... example code
   ```

### Adding Integration Connectors

To add a new LLM connector:

1. **Subclass LLMConnector**
   ```python
   from braintop.integration import LLMConnector
   
   class MyConnector(LLMConnector):
       def extract_topology(self):
           # Extract from your system
           pass
       
       def get_metadata(self):
           # Return metadata
           pass
   ```

2. **Add tests and examples**

## Testing Guidelines

### Test Structure

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Group related tests in classes if needed

### Writing Tests

```python
import pytest
from braintop.layers import MyLayer

def test_feature():
    """Test description."""
    layer = MyLayer()
    # Arrange
    layer.add_node("node1", weight=1.0)
    
    # Act
    result = layer.compute_positions()
    
    # Assert
    assert "node1" in result
    assert isinstance(result["node1"], np.ndarray)
```

### Running Specific Tests

```bash
# Single test file
pytest tests/test_layers.py -v

# Single test function
pytest tests/test_layers.py::test_conceptual_layer_creation -v

# With markers
pytest -m "not slow" -v
```

## Documentation Guidelines

### Docstring Format

Use Google-style docstrings:

```python
def my_function(param1: str, param2: int = 0) -> bool:
    """
    Brief description.
    
    Longer description if needed, explaining behavior,
    edge cases, and usage patterns.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 0)
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When param2 is negative
    
    Example:
        >>> result = my_function("test", 5)
        >>> print(result)
        True
    """
    pass
```

### API Documentation

When adding new APIs:

1. Add docstrings to all public methods
2. Update `API.md` with new API reference
3. Add examples to `README.md` if relevant
4. Update `CHANGELOG.md`

## Pull Request Process

1. **Ensure tests pass**: All existing and new tests must pass
2. **Update documentation**: Keep docs in sync with code
3. **Add to CHANGELOG**: Document your changes
4. **Request review**: Tag maintainers for review
5. **Address feedback**: Make requested changes
6. **Squash commits**: Clean up commit history before merge

## Code Review Checklist

Before submitting, verify:

- [ ] All tests pass
- [ ] Code follows style guidelines
- [ ] Documentation is complete
- [ ] No security vulnerabilities
- [ ] Examples work (if applicable)
- [ ] CHANGELOG.md updated
- [ ] No unnecessary dependencies added

## Questions?

- Open an issue for bugs or feature requests
- Use discussions for questions and ideas
- Tag maintainers for urgent issues

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to BrainTop! 🎉
