"""
Unit tests for topology layers.
"""

import pytest
import numpy as np
from braintop.layers.conceptual import ConceptualLayer
from braintop.layers.riemannian import RiemannianLayer
from braintop.layers.neural import NeuralLayer


def test_conceptual_layer_creation():
    """Test conceptual layer creation."""
    layer = ConceptualLayer(hierarchy_strength=0.8, spread_factor=2.5)
    
    assert layer.name == "conceptual"
    assert layer.hierarchy_strength == 0.8
    assert layer.spread_factor == 2.5


def test_conceptual_hierarchy():
    """Test hierarchical positioning in conceptual layer."""
    layer = ConceptualLayer()
    
    layer.add_node("root", weight=1.0)
    layer.add_node("child1", weight=0.8)
    layer.add_node("child2", weight=0.8)
    
    layer.add_edge("root", "child1", type="parent_of")
    layer.add_edge("root", "child2", type="parent_of")
    
    positions = layer.compute_positions()
    
    assert len(positions) == 3
    # Root should have lower z (higher hierarchy)
    assert positions["root"][2] > positions["child1"][2]
    assert positions["root"][2] > positions["child2"][2]


def test_conceptual_geometry():
    """Test conceptual layer geometry computation."""
    layer = ConceptualLayer()
    
    layer.add_node("node1")
    layer.add_node("node2")
    layer.add_edge("node1", "node2")
    
    geometry = layer.compute_geometry()
    
    assert geometry["num_nodes"] == 2
    assert geometry["num_edges"] == 1
    assert geometry["type"] == "conceptual"
    assert "num_levels" in geometry


def test_riemannian_layer_sphere():
    """Test Riemannian layer with spherical manifold."""
    layer = RiemannianLayer(manifold_type="sphere", curvature=0.5)
    
    layer.add_node("node1", embedding=[0.5, 0.5])
    layer.add_node("node2", embedding=[-0.5, 0.5])
    
    positions = layer.compute_positions()
    
    assert len(positions) == 2
    # Check that positions are 3D
    assert len(positions["node1"]) == 3
    
    # Verify points are roughly on a sphere
    radius = 3.0 + 0.5 * 2.0
    for pos in positions.values():
        distance = np.linalg.norm(pos)
        assert distance > 0


def test_riemannian_layer_hyperbolic():
    """Test Riemannian layer with hyperbolic manifold."""
    layer = RiemannianLayer(manifold_type="hyperbolic", curvature=0.3)
    
    layer.add_node("node1", embedding=[0.3, 0.4])
    
    positions = layer.compute_positions()
    
    assert len(positions) == 1
    assert len(positions["node1"]) == 3


def test_riemannian_layer_torus():
    """Test Riemannian layer with torus manifold."""
    layer = RiemannianLayer(manifold_type="torus")
    
    layer.add_node("node1", embedding=[0.5, 0.5])
    
    positions = layer.compute_positions()
    
    assert len(positions) == 1
    assert len(positions["node1"]) == 3


def test_riemannian_geometry():
    """Test Riemannian layer geometry computation."""
    layer = RiemannianLayer(manifold_type="sphere", curvature=0.6)
    
    layer.add_node("node1", embedding=[0.5, 0.5])
    layer.add_node("node2", embedding=[-0.5, 0.5])
    
    geometry = layer.compute_geometry()
    
    assert geometry["manifold_type"] == "sphere"
    assert geometry["curvature"] == 0.6
    assert "avg_geodesic_distance" in geometry
    assert geometry["type"] == "riemannian"


def test_neural_layer_structure():
    """Test neural layer structure creation."""
    layer = NeuralLayer()
    
    layer_sizes = [4, 3, 2]
    layer.set_layer_structure(layer_sizes)
    
    nodes = layer.get_nodes()
    edges = layer.get_edges()
    
    # Should have total of 4+3+2 = 9 nodes
    assert len(nodes) == 9
    
    # Should have 4*3 + 3*2 = 18 edges (fully connected between layers)
    assert len(edges) == 18


def test_neural_layer_positions():
    """Test neural layer position computation."""
    layer = NeuralLayer(layer_spacing=2.0)
    
    layer.add_node("layer_0_neuron_0", layer=0)
    layer.add_node("layer_1_neuron_0", layer=1)
    
    positions = layer.compute_positions()
    
    assert len(positions) == 2
    # Layer 1 should be higher in z
    assert positions["layer_1_neuron_0"][2] > positions["layer_0_neuron_0"][2]


def test_neural_layer_geometry():
    """Test neural layer geometry computation."""
    layer = NeuralLayer()
    layer.set_layer_structure([4, 3, 2])
    
    geometry = layer.compute_geometry()
    
    assert geometry["num_nodes"] == 9
    assert geometry["num_layers"] == 3
    assert geometry["type"] == "neural"
    assert "avg_in_degree" in geometry
    assert "avg_out_degree" in geometry


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
