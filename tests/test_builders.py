"""Tests for TopologyBuilder."""


from braintop.utils.builders import TopologyBuilder
from braintop.integrations.llm_connector import LLMConnector


def test_builder_creation():
    """Test creating a TopologyBuilder."""
    builder = TopologyBuilder("test_topo", "Test Topology")
    
    assert builder.topology.id == "test_topo"
    assert builder.topology.name == "Test Topology"


def test_builder_add_conceptual_layer():
    """Test adding conceptual layer."""
    builder = TopologyBuilder("test_topo", "Test")
    
    concepts = ["A", "B", "C"]
    builder.add_conceptual_layer("layer1", num_nodes=3, concepts=concepts)
    
    topology = builder.build()
    
    assert len(topology.layers) == 1
    assert topology.layers[0].id == "layer1"
    assert len(topology.layers[0].nodes) == 3


def test_builder_add_riemannian_layer():
    """Test adding Riemannian layer."""
    builder = TopologyBuilder("test_topo", "Test")
    
    builder.add_riemannian_layer("sphere", num_nodes=5, manifold_type="sphere")
    
    topology = builder.build()
    
    assert len(topology.layers) == 1
    assert topology.layers[0].id == "sphere"
    assert len(topology.layers[0].nodes) == 5


def test_builder_connect_layers():
    """Test connecting layers."""
    builder = TopologyBuilder("test_topo", "Test")
    
    builder.add_conceptual_layer("layer1", num_nodes=3, concepts=["A", "B", "C"])
    builder.add_riemannian_layer("layer2", num_nodes=5, manifold_type="sphere")
    builder.connect_layers("layer1", "layer2", "nearest", num_connections=2)
    
    topology = builder.build()
    
    assert len(topology.inter_layer_edges) > 0


def test_builder_add_layer_from_llm():
    """Test adding layer from LLM."""
    connector = LLMConnector(provider="mock")
    connector.connect()
    
    builder = TopologyBuilder("test_topo", "Test")
    texts = ["Text A", "Text B", "Text C"]
    
    builder.add_layer_from_llm("llm_layer", connector, texts)
    
    topology = builder.build()
    
    assert len(topology.layers) == 1
    assert topology.layers[0].id == "llm_layer"
    assert len(topology.layers[0].nodes) == 3


def test_builder_set_metadata():
    """Test setting metadata."""
    builder = TopologyBuilder("test_topo", "Test")
    
    builder.set_metadata(version="1.0", author="Test")
    
    topology = builder.build()
    
    assert topology.metadata["version"] == "1.0"
    assert topology.metadata["author"] == "Test"


def test_builder_chaining():
    """Test fluent interface chaining."""
    topology = (TopologyBuilder("test_topo", "Test")
        .add_conceptual_layer("l1", num_nodes=3, concepts=["A", "B", "C"])
        .add_riemannian_layer("l2", num_nodes=5, manifold_type="sphere")
        .connect_layers("l1", "l2", "random", num_connections=2)
        .set_metadata(version="1.0")
        .build())
    
    assert len(topology.layers) == 2
    assert len(topology.inter_layer_edges) > 0
    assert topology.metadata["version"] == "1.0"
