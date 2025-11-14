"""Braintop - A beautiful rich 3D topology visualizer for LLMs.

This package provides a modular framework for visualizing complex topological
structures in Large Language Models, with support for various layer types
including conceptual, Riemannian, and custom geometries.
"""

__version__ = "0.1.0"

from braintop.core.topology import Topology
from braintop.core.visualizer import TopologyVisualizer
from braintop.layers.conceptual import ConceptualLayer
from braintop.layers.riemannian import RiemannianLayer

__all__ = [
    "Topology",
    "TopologyVisualizer",
    "ConceptualLayer",
    "RiemannianLayer",
]
