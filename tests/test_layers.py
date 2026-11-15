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


def test_conceptual_layer_pca_handles_single_sample():
    """Test PCA projection returns finite 3D coordinates for one sample."""
    layer = ConceptualLayer("test_layer")
    data = np.array([[1.0, 2.0]])

    positions = layer.compute_positions(data, method="pca")

    assert positions == [(0.0, 0.0, 0.0)]


def test_conceptual_layer_pca_pads_low_dimensional_data():
    """Test PCA projection pads missing components for low-dimensional inputs."""
    layer = ConceptualLayer("test_layer")
    data = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])

    positions = layer.compute_positions(data, method="pca")

    assert len(positions) == 3
    assert all(len(pos) == 3 for pos in positions)
    assert all(np.isfinite(coordinate) for pos in positions for coordinate in pos)


def test_conceptual_layer_with_embeddings():
    """Test creating layer with custom embeddings."""
    layer = ConceptualLayer("test_layer")

    embeddings = np.random.randn(5, 100)
    concepts = ["A", "B", "C", "D", "E"]

    layer_model = layer.create_layer_with_nodes(
        num_nodes=5, concepts=concepts, embeddings=embeddings
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


def test_riemannian_torus_rejects_invalid_radii():
    """Test torus validation rejects self-intersecting radius configuration."""
    layer = RiemannianLayer("torus_layer")
    layer.set_config(manifold_type="torus", radius=1.0, minor_radius=1.0)

    with pytest.raises(ValueError, match="minor_radius"):
        layer.generate_nodes(num_nodes=1)


def test_riemannian_hyperbolic_rejects_nonnegative_curvature():
    """Test hyperbolic validation requires negative curvature."""
    layer = RiemannianLayer("hyperbolic_layer")
    layer.set_config(manifold_type="hyperbolic", curvature=0.0)

    with pytest.raises(ValueError, match="negative"):
        layer.generate_nodes(num_nodes=1)


def test_riemannian_create_layer_with_nodes():
    """Test creating complete layer with nodes."""
    layer = RiemannianLayer("test_layer")
    layer.set_config(manifold_type="sphere", radius=2.0)

    labels = [f"Node {i}" for i in range(5)]
    layer_model = layer.create_layer_with_nodes(num_nodes=5, labels=labels)

    assert len(layer_model.nodes) == 5
    assert layer_model.layer_type == LayerType.RIEMANNIAN
    assert layer_model.nodes[0].label == "Node 0"


def test_conceptual_layer_rejects_embedding_row_mismatch():
    """Review regression: node count and embedding rows must agree."""
    layer = ConceptualLayer("test_layer")
    embeddings = np.ones((2, 3))

    with pytest.raises(ValueError, match="row count"):
        layer.generate_nodes(num_nodes=3, embeddings=embeddings)


def test_conceptual_layer_rejects_nonfinite_embeddings():
    """Review regression: PCA inputs must be finite."""
    layer = ConceptualLayer("test_layer")
    embeddings = np.array([[0.0, np.inf], [1.0, 2.0]])

    with pytest.raises(ValueError, match="finite"):
        layer.compute_positions(embeddings, method="pca")


def test_riemannian_layer_rejects_parameter_row_mismatch():
    """Review regression: parameter rows must match requested node count."""
    layer = RiemannianLayer("sphere_layer")
    parameters = np.zeros((2, 2))

    with pytest.raises(ValueError, match="row count"):
        layer.generate_nodes(num_nodes=3, parameters=parameters)


def test_riemannian_sphere_rejects_nonpositive_radius():
    """Review regression: sphere radius must be geometrically valid."""
    layer = RiemannianLayer("sphere_layer")
    layer.set_config(manifold_type="sphere", radius=0.0)

    with pytest.raises(ValueError, match="positive"):
        layer.generate_nodes(num_nodes=1)


def test_layers_generate_zero_nodes_without_reduction_errors():
    """Review regression: empty generated layers should stay empty."""
    conceptual = ConceptualLayer("conceptual")
    riemannian = RiemannianLayer("riemannian")

    assert conceptual.generate_nodes(num_nodes=0) == []
    assert riemannian.generate_nodes(num_nodes=0) == []


def test_riemannian_hyperbolic_parameters_are_area_uniform_by_cdf():
    """Review regression: hyperbolic rho sampling follows area-uniform CDF."""
    layer = RiemannianLayer("hyperbolic_layer")
    layer.set_config(manifold_type="hyperbolic", curvature=-1.0, max_radius=4.0)

    np.random.seed(7)
    parameters = layer._generate_uniform_parameters(5000)
    rho = parameters[:, 0]
    transformed = (np.cosh(rho) - 1.0) / (np.cosh(4.0) - 1.0)

    assert abs(float(np.mean(transformed)) - 0.5) < 0.02
    assert np.mean(rho > 3.0) > np.mean(rho < 1.0)


def test_riemannian_hyperbolic_rejects_invalid_max_radius():
    """Review regression: finite geodesic disk radius is required for sampling."""
    layer = RiemannianLayer("hyperbolic_layer")
    layer.set_config(manifold_type="hyperbolic", curvature=-1.0, max_radius=np.inf)

    with pytest.raises(ValueError, match="max_radius"):
        layer.generate_nodes(num_nodes=1)


def test_conceptual_layer_rejects_unknown_projection_method():
    """A misspelled projection must not silently become a random projection."""
    layer = ConceptualLayer("test_layer")

    with pytest.raises(ValueError, match="Unsupported projection method"):
        layer.compute_positions(np.ones((3, 4)), method="pca-ish")


def test_riemannian_layer_rejects_unknown_manifold_type():
    """A misspelled manifold must not silently render as a sphere."""
    layer = RiemannianLayer("test_layer")
    layer.set_config(manifold_type="hyperbollic")

    with pytest.raises(ValueError, match="Unsupported manifold type"):
        layer.generate_nodes(num_nodes=3)
