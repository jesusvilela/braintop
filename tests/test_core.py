"""
Unit tests for core topology components.
"""

import pytest
import numpy as np
from braintop.core.topology import TopologyLayer, TopologyGraph


class SimpleLayer(TopologyLayer):
    """Simple implementation for testing."""
    
    def compute_positions(self):
        positions = {}
        for i, node_id in enumerate(self.get_nodes()):
            positions[node_id] = np.array([i, i, i])
        return positions
    
    def compute_geometry(self):
        return {
            "num_nodes": len(self.get_nodes()),
            "num_edges": len(self.get_edges()),
            "type": "simple",
        }


def test_topology_layer_creation():
    """Test basic layer creation."""
    layer = SimpleLayer("test_layer")
    assert layer.name == "test_layer"
    assert len(layer.get_nodes()) == 0
    assert len(layer.get_edges()) == 0


def test_add_nodes():
    """Test adding nodes to a layer."""
    layer = SimpleLayer("test")
    
    layer.add_node("node1", weight=1.0, type="test")
    layer.add_node("node2", weight=0.5)
    
    nodes = layer.get_nodes()
    assert len(nodes) == 2
    assert "node1" in nodes
    assert "node2" in nodes
    assert nodes["node1"]["weight"] == 1.0
    assert nodes["node1"]["type"] == "test"


def test_add_edges():
    """Test adding edges to a layer."""
    layer = SimpleLayer("test")
    
    layer.add_node("node1")
    layer.add_node("node2")
    layer.add_edge("node1", "node2", weight=0.8)
    
    edges = layer.get_edges()
    assert len(edges) == 1
    assert edges[0][0] == "node1"
    assert edges[0][1] == "node2"
    assert edges[0][2]["weight"] == 0.8


def test_compute_positions():
    """Test position computation."""
    layer = SimpleLayer("test")
    
    layer.add_node("node1")
    layer.add_node("node2")
    
    positions = layer.compute_positions()
    assert len(positions) == 2
    assert "node1" in positions
    assert "node2" in positions
    assert isinstance(positions["node1"], np.ndarray)
    assert len(positions["node1"]) == 3


def test_topology_graph():
    """Test TopologyGraph functionality."""
    graph = TopologyGraph()
    
    layer1 = SimpleLayer("layer1")
    layer1.add_node("node1")
    
    layer2 = SimpleLayer("layer2")
    layer2.add_node("node2")
    
    graph.add_layer(layer1)
    graph.add_layer(layer2)
    
    assert len(graph.get_all_layers()) == 2
    assert graph.get_layer("layer1") is layer1
    assert graph.get_layer("layer2") is layer2


def test_graph_metadata():
    """Test graph metadata functionality."""
    graph = TopologyGraph()
    
    graph.set_metadata("system", "test_system")
    graph.set_metadata("version", 1.0)
    
    assert graph.get_metadata("system") == "test_system"
    assert graph.get_metadata("version") == 1.0
    assert graph.get_metadata("nonexistent") is None


def test_to_networkx():
    """Test conversion to NetworkX graph."""
    layer = SimpleLayer("test")
    layer.add_node("node1", weight=1.0)
    layer.add_node("node2", weight=0.5)
    layer.add_edge("node1", "node2", strength=0.8)
    
    graph = TopologyGraph()
    graph.add_layer(layer)
    
    nx_graph = graph.to_networkx("test")
    assert nx_graph.number_of_nodes() == 2
    assert nx_graph.number_of_edges() == 1
    assert nx_graph.nodes["node1"]["weight"] == 1.0


def test_clear_layer():
    """Test clearing layer data."""
    layer = SimpleLayer("test")
    layer.add_node("node1")
    layer.add_node("node2")
    layer.add_edge("node1", "node2")
    
    assert len(layer.get_nodes()) == 2
    assert len(layer.get_edges()) == 1
    
    layer.clear()
    
    assert len(layer.get_nodes()) == 0
    assert len(layer.get_edges()) == 0


def test_remove_layer():
    """Test removing a layer from graph."""
    graph = TopologyGraph()
    layer = SimpleLayer("test")
    
    graph.add_layer(layer)
    assert "test" in graph.get_all_layers()
    
    graph.remove_layer("test")
    assert "test" not in graph.get_all_layers()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
