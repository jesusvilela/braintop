"""Tests for core topology module."""

import pytest
import numpy as np

from braintop.core.topology import (
    Topology, LayerModel, NodeModel, EdgeModel, LayerType
)


def test_node_model_creation():
    """Test creating a NodeModel."""
    node = NodeModel(
        id="node_1",
        position=(1.0, 2.0, 3.0),
        label="Test Node",
        layer_id="layer_1"
    )
    
    assert node.id == "node_1"
    assert node.position == (1.0, 2.0, 3.0)
    assert node.label == "Test Node"
    assert node.layer_id == "layer_1"


def test_edge_model_creation():
    """Test creating an EdgeModel."""
    edge = EdgeModel(
        id="edge_1",
        source="node_1",
        target="node_2",
        weight=0.8
    )
    
    assert edge.id == "edge_1"
    assert edge.source == "node_1"
    assert edge.target == "node_2"
    assert edge.weight == 0.8


def test_layer_model_creation():
    """Test creating a LayerModel."""
    nodes = [
        NodeModel(id="n1", position=(0, 0, 0), layer_id="l1"),
        NodeModel(id="n2", position=(1, 1, 1), layer_id="l1")
    ]
    
    layer = LayerModel(
        id="layer_1",
        name="Test Layer",
        layer_type=LayerType.CONCEPTUAL,
        nodes=nodes
    )
    
    assert layer.id == "layer_1"
    assert layer.name == "Test Layer"
    assert layer.layer_type == LayerType.CONCEPTUAL
    assert len(layer.nodes) == 2


def test_topology_creation():
    """Test creating a Topology."""
    topology = Topology(
        id="topo_1",
        name="Test Topology"
    )
    
    assert topology.id == "topo_1"
    assert topology.name == "Test Topology"
    assert len(topology.layers) == 0
    assert len(topology.inter_layer_edges) == 0


def test_topology_add_layer():
    """Test adding a layer to topology."""
    topology = Topology(id="topo_1", name="Test")
    
    layer = LayerModel(
        id="layer_1",
        name="Layer 1",
        layer_type=LayerType.CONCEPTUAL
    )
    
    topology.add_layer(layer)
    
    assert len(topology.layers) == 1
    assert topology.layers[0].id == "layer_1"


def test_topology_get_layer():
    """Test getting a layer by ID."""
    topology = Topology(id="topo_1", name="Test")
    
    layer1 = LayerModel(id="l1", name="Layer 1", layer_type=LayerType.CONCEPTUAL)
    layer2 = LayerModel(id="l2", name="Layer 2", layer_type=LayerType.RIEMANNIAN)
    
    topology.add_layer(layer1)
    topology.add_layer(layer2)
    
    retrieved = topology.get_layer("l1")
    assert retrieved is not None
    assert retrieved.id == "l1"
    
    not_found = topology.get_layer("l3")
    assert not_found is None


def test_topology_compute_statistics():
    """Test computing topology statistics."""
    topology = Topology(id="topo_1", name="Test")
    
    # Add layers with nodes
    layer1 = LayerModel(
        id="l1",
        name="Layer 1",
        layer_type=LayerType.CONCEPTUAL,
        nodes=[
            NodeModel(id="n1", position=(0, 0, 0), layer_id="l1"),
            NodeModel(id="n2", position=(1, 1, 1), layer_id="l1")
        ]
    )
    
    layer2 = LayerModel(
        id="l2",
        name="Layer 2",
        layer_type=LayerType.RIEMANNIAN,
        nodes=[
            NodeModel(id="n3", position=(2, 2, 2), layer_id="l2")
        ]
    )
    
    topology.add_layer(layer1)
    topology.add_layer(layer2)
    
    # Add edge
    edge = EdgeModel(id="e1", source="n1", target="n3", weight=1.0)
    topology.add_inter_layer_edge(edge)
    
    stats = topology.compute_statistics()
    
    assert stats["num_layers"] == 2
    assert stats["total_nodes"] == 3
    assert stats["total_edges"] == 1
    assert LayerType.CONCEPTUAL in stats["layer_types"]
    assert LayerType.RIEMANNIAN in stats["layer_types"]


def test_topology_get_all_nodes():
    """Test getting all nodes from topology."""
    topology = Topology(id="topo_1", name="Test")
    
    layer1 = LayerModel(
        id="l1",
        name="Layer 1",
        layer_type=LayerType.CONCEPTUAL,
        nodes=[
            NodeModel(id="n1", position=(0, 0, 0), layer_id="l1"),
            NodeModel(id="n2", position=(1, 1, 1), layer_id="l1")
        ]
    )
    
    layer2 = LayerModel(
        id="l2",
        name="Layer 2",
        layer_type=LayerType.RIEMANNIAN,
        nodes=[
            NodeModel(id="n3", position=(2, 2, 2), layer_id="l2")
        ]
    )
    
    topology.add_layer(layer1)
    topology.add_layer(layer2)
    
    all_nodes = topology.get_all_nodes()
    
    assert len(all_nodes) == 3
    assert any(n.id == "n1" for n in all_nodes)
    assert any(n.id == "n2" for n in all_nodes)
    assert any(n.id == "n3" for n in all_nodes)
