# Contributing to Braintop

Thank you for your interest in contributing to Braintop! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/jesusvilela/braintop.git
   cd braintop
   ```

2. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Run tests**:
   ```bash
   pytest
   ```

## Project Structure

```
braintop/
├── src/braintop/       # Main package source
│   ├── core/           # Core data structures and visualizer
│   ├── layers/         # Layer implementations
│   ├── integrations/   # External system integrations
│   ├── utils/          # Helper utilities
│   └── visualizers/    # Visualization components
├── tests/              # Test suite
├── examples/           # Example scripts
└── docs/               # Documentation
```

## Making Changes

1. **Create a new branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines below

3. **Add tests** for new functionality

4. **Run tests and linting**:
   ```bash
   pytest
   black src/ tests/
   flake8 src/
   mypy src/
   ```

5. **Commit your changes**:
   ```bash
   git commit -m "Description of your changes"
   ```

6. **Push and create a pull request**

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Use docstrings for classes and public methods
- Format code with Black (line length: 100)
- Maximum line length: 100 characters

## Adding New Layer Types

To add a new layer type:

1. Create a new file in `src/braintop/layers/`
2. Extend `BaseLayer` class
3. Implement `generate_nodes()` and `compute_positions()` methods
4. Add the layer type to `LayerType` enum in `core/topology.py`
5. Add tests in `tests/test_layers.py`
6. Update documentation

Example:
```python
from braintop.layers.base import BaseLayer
from braintop.core.topology import LayerType

class MyCustomLayer(BaseLayer):
    def __init__(self, layer_id: str, name: str = "Custom Layer"):
        super().__init__(layer_id, name, LayerType.CUSTOM)
    
    def generate_nodes(self, num_nodes: int, **kwargs):
        # Implementation
        pass
    
    def compute_positions(self, data, **kwargs):
        # Implementation
        pass
```

## Adding New Integrations

To add a new integration:

1. Create a new file in `src/braintop/integrations/`
2. Extend `BaseIntegration` class
3. Implement required methods: `connect()`, `disconnect()`, `get_embeddings()`, `get_model_info()`
4. Add tests in `tests/test_integrations.py`
5. Update documentation

## Testing

- Write tests for all new functionality
- Maintain test coverage above 70%
- Use pytest for testing
- Mock external dependencies

## Documentation

- Update README.md for new features
- Add docstrings to all public classes and methods
- Update ARCHITECTURE.md for structural changes
- Add examples for new functionality

## Pull Request Process

1. Ensure all tests pass
2. Update documentation
3. Add entry to CHANGELOG (if applicable)
4. Request review from maintainers
5. Address review feedback

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Help others learn and grow

## Questions?

Open an issue on GitHub or reach out to the maintainers.

Thank you for contributing to Braintop!
