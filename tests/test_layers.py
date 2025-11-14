"""Tests for layer implementations."""

import pytest
import numpy as np

from braintop.layers.conceptual import ConceptualLayer
from braintop.layers.riemannian import RiemannianLayer
from braintop.core.topology import LayerType


def test_conceptual_layer_creation():
    """Test creating a ConceptualLayer."""
    layer = ConceptualLayer("test_layer", "Test Conceptual Layer")
    
    assert layer.layer_id == "test_layer"
    assert layer.name == "Test Conceptual Layer"
    assert layer.layer_type == LayerType.CONCEPTUAL


def test_conceptual_layer_generate_nodes():
    """Test generating nodes in conceptual layer."""
    layer = ConceptualLayer("test_layer")
    concepts = ["Concept A", "Concept B", "Concept C"]
    
    nodes = layer.generate_nodes(num_nodes=3, concepts=concepts)
    
    assert len(nodes) == 3
    assert nodes[0].label == "Concept A"
    assert nodes[1].label == "Concept B"
    assert nodes[2].label == "Concept C"
    assert all(len(node.position) == 3 for node in nodes)


def test_conceptual_layer_pca_projection():
    """Test PCA projection in conceptual layer."""
    layer = ConceptualLayer("test_layer")
    
    # Create synthetic high-dimensional data
    data = np.random.randn(10, 50)
    
    positions = layer.compute_positions(data, method="pca")
    
    assert len(positions) == 10
    assert all(len(pos) == 3 for pos in positions)
    assert all(isinstance(pos, tuple) for pos in positions)


def test_conceptual_layer_with_embeddings():
    """Test creating layer with custom embeddings."""
    layer = ConceptualLayer("test_layer")
    
    embeddings = np.random.randn(5, 100)
    concepts = ["A", "B", "C", "D", "E"]
    
    layer_model = layer.create_layer_with_nodes(
        num_nodes=5,
        concepts=concepts,
        embeddings=embeddings
    )
    
    assert len(layer_model.nodes) == 5
    assert layer_model.layer_type == LayerType.CONCEPTUAL


def test_riemannian_layer_creation():
    """Test creating a RiemannianLayer."""
    layer = RiemannianLayer("test_layer", "Test Riemannian Layer")
    
    assert layer.layer_id == "test_layer"
    assert layer.name == "Test Riemannian Layer"
    assert layer.layer_type == LayerType.RIEMANNIAN


def test_riemannian_layer_sphere():
    """Test sphere manifold generation."""
    layer = RiemannianLayer("sphere_layer")
    layer.set_config(manifold_type="sphere", radius=1.0)
    
    nodes = layer.generate_nodes(num_nodes=10)
    
    assert len(nodes) == 10
    
    # Check that points are approximately on sphere
    for node in nodes:
        x, y, z = node.position
        distance = np.sqrt(x**2 + y**2 + z**2)
        assert abs(distance - 1.0) < 0.1  # Allow small numerical error


def test_riemannian_layer_torus():
    """Test torus manifold generation."""
    layer = RiemannianLayer("torus_layer")
    layer.set_config(manifold_type="torus", radius=1.0)
    
    nodes = layer.generate_nodes(num_nodes=15)
    
    assert len(nodes) == 15
    assert all(len(node.position) == 3 for node in nodes)


def test_riemannian_layer_hyperbolic():
    """Test hyperbolic manifold generation."""
    layer = RiemannianLayer("hyperbolic_layer")
    layer.set_config(manifold_type="hyperbolic", curvature=-1.0)
    
    nodes = layer.generate_nodes(num_nodes=8)
    
    assert len(nodes) == 8
    assert all(len(node.position) == 3 for node in nodes)


def test_riemannian_create_layer_with_nodes():
    """Test creating complete layer with nodes."""
    layer = RiemannianLayer("test_layer")
    layer.set_config(manifold_type="sphere", radius=2.0)
    
    labels = [f"Node {i}" for i in range(5)]
    layer_model = layer.create_layer_with_nodes(num_nodes=5, labels=labels)
    
    assert len(layer_model.nodes) == 5
    assert layer_model.layer_type == LayerType.RIEMANNIAN
    assert layer_model.nodes[0].label == "Node 0"
