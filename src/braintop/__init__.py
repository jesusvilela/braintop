"""
BrainTop - A beautiful rich 3D topology visualizer for LLMs

This package provides a modular architecture for visualizing complex
topology structures of Large Language Models, agents, and orchestrators.
"""

__version__ = "0.1.0"

from .core.topology import TopologyLayer, TopologyGraph
from .visualization.renderer import Renderer3D
from .layers.conceptual import ConceptualLayer
from .layers.riemannian import RiemannianLayer
from .layers.neural import NeuralLayer

__all__ = [
    "TopologyLayer",
    "TopologyGraph",
    "Renderer3D",
    "ConceptualLayer",
    "RiemannianLayer",
    "NeuralLayer",
]
