"""
Unit tests for integration components.
"""

import pytest
from braintop.integration import (
    AgentConnector,
    OrchestratorConnector,
    TransformerConnector,
    DataTransformer,
)
from braintop.core.topology import TopologyGraph


def test_agent_connector():
    """Test agent connector creation and metadata."""
    connector = AgentConnector(name="test_agent")
    
    assert connector.name == "test_agent"
    
    metadata = connector.get_metadata()
    assert metadata["connector_type"] == "agent"
    assert metadata["name"] == "test_agent"
    
    # Test extraction (returns empty graph in base implementation)
    graph = connector.extract_topology()
    assert isinstance(graph, TopologyGraph)


def test_orchestrator_connector():
    """Test orchestrator connector creation and metadata."""
    connector = OrchestratorConnector(name="test_orch")
    
    assert connector.name == "test_orch"
    
    metadata = connector.get_metadata()
    assert metadata["connector_type"] == "orchestrator"
    
    graph = connector.extract_topology()
    assert isinstance(graph, TopologyGraph)


def test_transformer_connector():
    """Test transformer connector creation."""
    connector = TransformerConnector(name="test_transformer")
    
    assert connector.name == "test_transformer"
    
    metadata = connector.get_metadata()
    assert metadata["connector_type"] == "transformer"
    
    graph = connector.extract_topology()
    assert isinstance(graph, TopologyGraph)


def test_data_transformer_from_dict():
    """Test transforming dictionary to topology graph."""
    data = {
        "nodes": [
            {"id": "node1", "weight": 1.0},
            {"id": "node2", "weight": 0.5},
        ],
        "edges": [
            {"source": "node1", "target": "node2", "weight": 0.8},
        ],
    }
    
    graph = DataTransformer.from_dict(data, layer_type="conceptual")
    
    layers = graph.get_all_layers()
    assert len(layers) == 1
    
    layer = list(layers.values())[0]
    nodes = layer.get_nodes()
    edges = layer.get_edges()
    
    assert len(nodes) == 2
    assert len(edges) == 1
    assert "node1" in nodes
    assert nodes["node1"]["weight"] == 1.0


def test_data_transformer_different_layers():
    """Test creating different layer types from data."""
    data = {
        "nodes": [{"id": "node1"}],
        "edges": [],
    }
    
    # Test conceptual
    graph1 = DataTransformer.from_dict(data, layer_type="conceptual")
    assert len(graph1.get_all_layers()) == 1
    
    # Test riemannian
    graph2 = DataTransformer.from_dict(data, layer_type="riemannian")
    assert len(graph2.get_all_layers()) == 1
    
    # Test neural
    graph3 = DataTransformer.from_dict(data, layer_type="neural")
    assert len(graph3.get_all_layers()) == 1


def test_data_transformer_from_networkx():
    """Test transforming NetworkX graph to topology graph."""
    try:
        import networkx as nx
    except ImportError:
        pytest.skip("NetworkX not available")
    
    # Create a simple NetworkX graph
    nx_graph = nx.Graph()
    nx_graph.add_node("node1", weight=1.0)
    nx_graph.add_node("node2", weight=0.5)
    nx_graph.add_edge("node1", "node2", strength=0.8)
    
    graph = DataTransformer.from_networkx(nx_graph, layer_type="conceptual")
    
    layers = graph.get_all_layers()
    assert len(layers) == 1
    
    layer = list(layers.values())[0]
    nodes = layer.get_nodes()
    edges = layer.get_edges()
    
    assert len(nodes) == 2
    assert len(edges) == 1
    assert "node1" in nodes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
