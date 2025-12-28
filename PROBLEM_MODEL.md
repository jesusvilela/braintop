# Braintop Problem Model Definition

## Problem Statement

**Core Problem**: Visualizing and analyzing the complex, high-dimensional relationships between semantic entities in Large Language Model (LLM) systems by mapping them into comprehensible 3D geometric spaces that preserve meaningful structural properties.

---

## 1. Relevant Entities

### 1.1 Primary Entities

- **Topology**: The complete visualization structure containing all layers and their relationships
  - Properties: unique identifier, name, metadata
  - Contains: multiple layers, inter-layer connections

- **Layer**: A geometric space containing a collection of related nodes
  - Properties: unique identifier, name, type, configuration parameters
  - Types: Conceptual, Riemannian (sphere/torus/hyperbolic), Euclidean, Custom
  - Contains: nodes positioned within the layer's geometric space

- **Node**: A discrete entity/concept positioned in 3D space
  - Properties: unique identifier, 3D position (x, y, z), label, metadata, parent layer reference
  - Represents: concepts, embeddings, semantic entities, or LLM internal states

- **Edge**: A connection/relationship between two nodes
  - Properties: unique identifier, source node reference, target node reference, weight, metadata
  - Types: intra-layer edges (within same layer), inter-layer edges (across different layers)

### 1.2 Supporting Entities

- **Embeddings**: High-dimensional vector representations of semantic entities
  - Properties: dimensionality, numerical values
  - Source: LLM providers, pre-computed vectors, or synthetic data

- **Geometric Manifold**: The underlying mathematical space for a layer
  - Types: Euclidean (flat), Spherical (positive curvature), Hyperbolic (negative curvature), Toroidal (periodic)
  - Properties: curvature, radius, periodicity parameters

- **LLM Connector**: Interface to external language model systems
  - Properties: provider type, model name, connection status
  - Capabilities: extract embeddings, compute relationships, analyze semantic similarity

- **Visualization**: The rendered 3D graphical representation
  - Properties: rendering format (HTML/interactive), visual styling, camera position
  - Output: interactive plots, static images, exportable graphs

---

## 2. State Variables

### 2.1 Topology State

- **`topology.id`**: String - unique identifier
- **`topology.name`**: String - human-readable name
- **`topology.layers`**: List[Layer] - collection of all layers (ordered)
- **`topology.inter_layer_edges`**: List[Edge] - connections between layers
- **`topology.metadata`**: Dictionary - arbitrary key-value properties

### 2.2 Layer State

- **`layer.id`**: String - unique identifier within topology
- **`layer.name`**: String - descriptive name
- **`layer.type`**: Enum[LayerType] - geometric type (conceptual, riemannian, etc.)
- **`layer.nodes`**: List[Node] - nodes contained in this layer
- **`layer.config`**: Dictionary - layer-specific configuration
  - For Conceptual: `projection_method`, `spacing_factor`, `dimensions`
  - For Riemannian: `manifold_type`, `radius`, `curvature`, `periodicity`

### 2.3 Node State

- **`node.id`**: String - unique identifier
- **`node.position`**: Tuple[float, float, float] - 3D coordinates
- **`node.label`**: String - display name/concept
- **`node.layer_id`**: String - parent layer reference
- **`node.metadata`**: Dictionary - additional properties (embeddings, features, etc.)

### 2.4 Edge State

- **`edge.id`**: String - unique identifier
- **`edge.source`**: String - source node ID
- **`edge.target`**: String - target node ID
- **`edge.weight`**: Float - connection strength [0, ∞)
- **`edge.metadata`**: Dictionary - additional properties

### 2.5 Integration State

- **`llm_connector.provider`**: String - LLM provider name
- **`llm_connector.model_name`**: String - specific model identifier
- **`llm_connector.is_connected`**: Boolean - connection status
- **`llm_connector.config`**: Dictionary - provider-specific settings

### 2.6 Builder State

- **`builder.topology`**: Topology - topology being constructed
- **`builder.edge_counter`**: Integer - monotonic edge ID generator

---

## 3. Possible Actions (with Preconditions and Effects)

### 3.1 Topology Construction Actions

#### Action: `CREATE_TOPOLOGY(id, name)`
**Preconditions**:
- `id` is unique
- `name` is non-empty string

**Effects**:
- New Topology entity created with empty layers and edges
- Topology state initialized

**Postconditions**:
- `topology.layers = []`
- `topology.inter_layer_edges = []`

---

#### Action: `ADD_CONCEPTUAL_LAYER(topology, layer_id, num_nodes, concepts?, embeddings?)`
**Preconditions**:
- `topology` exists and is mutable
- `layer_id` is unique within topology
- `num_nodes > 0`
- If `concepts` provided: `len(concepts) = num_nodes`
- If `embeddings` provided: `embeddings.shape[0] = num_nodes` and `embeddings.ndim >= 2`

**Effects**:
- New ConceptualLayer created with specified configuration
- `num_nodes` nodes generated with positions computed via dimensionality reduction
- If embeddings provided: positions computed from embeddings via PCA/projection
- If no embeddings: random high-dimensional vectors generated
- Layer added to `topology.layers`

**Postconditions**:
- `len(topology.layers)` increased by 1
- New layer contains `num_nodes` nodes with 3D positions
- All nodes have `layer_id` matching the new layer

---

#### Action: `ADD_RIEMANNIAN_LAYER(topology, layer_id, num_nodes, manifold_type, labels?)`
**Preconditions**:
- `topology` exists and is mutable
- `layer_id` is unique within topology
- `num_nodes > 0`
- `manifold_type ∈ {"sphere", "torus", "hyperbolic"}`
- If `labels` provided: `len(labels) = num_nodes`

**Effects**:
- New RiemannianLayer created with specified manifold geometry
- `num_nodes` nodes positioned on the manifold surface
- For sphere: nodes distributed on spherical surface
- For torus: nodes distributed on toroidal surface
- For hyperbolic: nodes positioned in Poincaré disk
- Layer added to `topology.layers`

**Postconditions**:
- `len(topology.layers)` increased by 1
- All node positions satisfy manifold constraints
- Node positions respect geometric properties (distances, curvature)

---

#### Action: `ADD_LAYER_FROM_LLM(topology, layer_id, llm_connector, texts)`
**Preconditions**:
- `topology` exists and is mutable
- `layer_id` is unique within topology
- `llm_connector.is_connected = true`
- `len(texts) > 0`

**Effects**:
- LLM connector processes texts to extract embeddings
- Embeddings analyzed to compute semantic relationships
- ConceptualLayer created with embeddings as input
- Nodes created with labels from texts
- Edges created based on semantic similarity/relationships
- Layer and edges added to topology

**Postconditions**:
- New layer added with `num_nodes = len(texts)`
- Edges may be added to `topology.inter_layer_edges` if relationships detected
- Node metadata contains LLM-derived information

---

#### Action: `CONNECT_LAYERS(topology, source_layer_id, target_layer_id, connection_type, num_connections)`
**Preconditions**:
- `topology` contains layers with IDs `source_layer_id` and `target_layer_id`
- `source_layer_id ≠ target_layer_id`
- `connection_type ∈ {"nearest", "random", "all"}`
- If connection_type is "nearest" or "random": `num_connections > 0`
- Source and target layers have at least 1 node each

**Effects**:
- Edges created between nodes in source and target layers according to strategy:
  - **nearest**: Each source node connects to `k` nearest target nodes (by Euclidean distance)
  - **random**: Each source node connects to `k` random target nodes
  - **all**: Each source node connects to all target nodes
- Edge weights computed based on connection type:
  - nearest: weight = 1/(1 + distance)
  - random/all: weight = 1.0
- New edges added to `topology.inter_layer_edges`

**Postconditions**:
- `len(topology.inter_layer_edges)` increased
- For "nearest" and "random": edges added = `num_source_nodes × num_connections`
- For "all": edges added = `num_source_nodes × num_target_nodes`

---

### 3.2 Layer Generation Actions

#### Action: `GENERATE_NODES(layer, num_nodes, data?)`
**Preconditions**:
- `layer` exists
- `num_nodes > 0`
- If data provided: data format matches layer type requirements

**Effects**:
- Nodes created with IDs: `{layer_id}_node_{0..num_nodes-1}`
- Positions computed based on layer type:
  - Conceptual: projection of embeddings or random distribution
  - Riemannian: geometric sampling on manifold
- Nodes added to `layer.nodes`

**Postconditions**:
- `len(layer.nodes) = num_nodes`
- All positions are valid 3D tuples
- All nodes reference parent layer

---

#### Action: `COMPUTE_POSITIONS(layer, data)`
**Preconditions**:
- `layer` exists and has configuration set
- `data` format appropriate for layer type:
  - Conceptual: embeddings matrix (N × D)
  - Riemannian: optional positioning hints

**Effects**:
- For Conceptual layers:
  - High-dimensional embeddings projected to 3D via PCA or configured method
  - Positions scaled by `spacing_factor`
- For Riemannian layers:
  - Positions sampled uniformly on manifold surface
  - Geometric constraints enforced

**Postconditions**:
- Each data point mapped to unique 3D position
- Positions preserve structural properties (proximity, clustering)
- Positions satisfy layer geometric constraints

---

### 3.3 Visualization Actions

#### Action: `CREATE_VISUALIZATION(topology)`
**Preconditions**:
- `topology` exists
- `len(topology.layers) > 0`
- At least one layer has nodes

**Effects**:
- 3D scatter plot created with:
  - Points for each node (colored by layer)
  - Lines for each edge
  - Labels and hover information
  - Interactive camera controls
- Visualization object created with Plotly figure

**Postconditions**:
- Visualization contains all nodes and edges from topology
- Visual properties distinguish different layers
- Interactive features enabled

---

#### Action: `EXPORT_VISUALIZATION(visualization, filepath, format)`
**Preconditions**:
- `visualization` exists
- `filepath` is valid writable path
- `format ∈ {"html", "png", "svg", "json"}`

**Effects**:
- Visualization data serialized to specified format
- File written to filesystem

**Postconditions**:
- File exists at `filepath`
- File contains complete visualization data

---

#### Action: `EXPORT_TO_NETWORKX(topology)`
**Preconditions**:
- `topology` exists
- NetworkX library available

**Effects**:
- NetworkX graph object created
- Nodes added with position attributes
- Edges added with weight attributes

**Postconditions**:
- Graph contains all topology nodes and edges
- Graph analyzable with NetworkX algorithms

---

### 3.4 LLM Integration Actions

#### Action: `CONNECT_LLM(provider, model_name, config)`
**Preconditions**:
- `provider` is supported LLM provider name
- `model_name` is valid for provider
- Credentials/API keys available in config

**Effects**:
- Connection established to LLM provider
- Connector state set to connected

**Postconditions**:
- `llm_connector.is_connected = true`
- Connector ready to process requests

---

#### Action: `EXTRACT_EMBEDDINGS(llm_connector, texts)`
**Preconditions**:
- `llm_connector.is_connected = true`
- `texts` is non-empty list of strings

**Effects**:
- Each text processed through LLM to obtain embedding vector
- Embeddings matrix created (N × D)

**Postconditions**:
- Embeddings returned with shape `(len(texts), embedding_dim)`
- Embeddings are normalized vectors

---

#### Action: `COMPUTE_RELATIONSHIPS(llm_connector, texts)`
**Preconditions**:
- `llm_connector.is_connected = true`
- `len(texts) > 1`

**Effects**:
- Semantic similarity computed between text pairs
- Relationships above threshold converted to edges
- Edge weights reflect similarity scores

**Postconditions**:
- Edge list returned with source, target, weight
- Weights in range [0, 1]

---

## 4. Constraints

### 4.1 Structural Constraints

1. **Unique Identifiers**:
   - All topology IDs must be unique within a session
   - All layer IDs must be unique within a topology
   - All node IDs must be unique within a topology
   - All edge IDs must be unique within a topology

2. **Reference Integrity**:
   - Every `node.layer_id` must reference an existing layer in the topology
   - Every `edge.source` and `edge.target` must reference existing nodes
   - Inter-layer edges must connect nodes from different layers

3. **Cardinality Constraints**:
   - A topology must have at least 1 layer to be visualizable
   - A layer must have at least 1 node to be meaningful
   - `num_nodes ≥ 1` for all layer creation operations
   - `num_connections ≥ 1` for nearest/random connection strategies

4. **Type Constraints**:
   - Node positions must be 3-tuples of floats: `(x, y, z)`
   - Edge weights must be non-negative: `weight ≥ 0`
   - Layer types must be from predefined enum values

### 4.2 Geometric Constraints

5. **Manifold Constraints**:
   - **Sphere**: All node positions must satisfy `x² + y² + z² = radius²`
   - **Torus**: Positions must lie on toroidal surface defined by major/minor radii
   - **Hyperbolic**: Positions must lie within Poincaré disk: `x² + y² < 1`
   - **Conceptual**: No geometric constraints (Euclidean space)

6. **Dimensionality Constraints**:
   - Input embeddings can be D-dimensional where D ≥ 3
   - Output positions must be exactly 3-dimensional
   - Projection must reduce dimensions: `output_dim = 3 ≤ input_dim`

7. **Distance Preservation** (soft constraint):
   - Projection should approximately preserve relative distances
   - Nearby points in high-D space should be nearby in 3D
   - Clustering structure should be maintained where possible

### 4.3 Computational Constraints

8. **Scalability Constraints**:
   - Visualization performance degrades with large node counts (>10,000 nodes)
   - Browser rendering limited by WebGL capabilities
   - Memory usage grows O(N²) for all-to-all connections

9. **Numerical Constraints**:
   - Position coordinates must be finite (not NaN or Inf)
   - Embedding values must be finite
   - Edge weights must be finite and non-negative

10. **Configuration Constraints**:
    - Projection method must be valid: `{"pca", "tsne", "umap", "random"}`
    - Manifold type must be valid: `{"sphere", "torus", "hyperbolic"}`
    - Connection type must be valid: `{"nearest", "random", "all"}`

### 4.4 Logical Constraints

11. **Layer Ordering** (soft constraint):
    - Layers maintain insertion order for visualization stacking
    - No circular dependencies in layer relationships

12. **Edge Consistency**:
    - An edge cannot have identical source and target (no self-loops by default)
    - Duplicate edges (same source-target pair) should be avoided or merged

13. **Metadata Consistency**:
    - Metadata fields must be JSON-serializable for export
    - Reserved metadata keys should not be overwritten

14. **LLM Integration Constraints**:
    - LLM must be connected before embedding extraction
    - Text inputs must be non-empty strings
    - Embedding dimensions must be consistent for all texts in a batch

### 4.5 Visualization Constraints

15. **Rendering Constraints**:
    - At least one layer must exist before visualization
    - Node colors must be distinguishable for different layers
    - Edge opacity may be reduced for dense connections

16. **Export Format Constraints**:
    - HTML export requires web browser for viewing
    - Static image export requires kaleido library
    - NetworkX export requires all nodes to have positions

17. **Interactive Constraints**:
    - Camera controls require 3D rendering context
    - Hover labels require node metadata to be set
    - Animations require temporal data in metadata

---

## 5. Success Criteria (Goal State)

A successful topology visualization satisfies:

1. **Completeness**: All input entities (concepts, embeddings, relationships) are represented
2. **Visual Clarity**: Nodes and edges are distinguishable and interpretable
3. **Structural Fidelity**: Important relationships and patterns are preserved from high-D to 3D
4. **Interactivity**: Users can explore, rotate, zoom, and query the visualization
5. **Exportability**: Visualization can be saved and shared in standard formats
6. **Extensibility**: New layer types and integrations can be added without modifying core

---

## 6. Problem Complexity Analysis

### Computational Complexity:
- **Node generation**: O(N) where N = num_nodes
- **PCA projection**: O(N × D²) where D = embedding dimension
- **Nearest neighbor connections**: O(N₁ × N₂ × log(N₂)) with spatial indexing
- **All-to-all connections**: O(N₁ × N₂)
- **Visualization rendering**: O(N + E) where E = num_edges

### Space Complexity:
- **Topology storage**: O(N + E)
- **Embedding storage**: O(N × D)
- **Visualization data**: O(N + E)

### Key Challenges:
1. **Dimensionality Reduction**: Mapping high-D embeddings to 3D while preserving semantic structure
2. **Manifold Embedding**: Correctly positioning nodes on curved surfaces
3. **Visual Clutter**: Managing edge density in dense graphs
4. **Semantic Fidelity**: Ensuring geometric proximity reflects semantic similarity
5. **Scalability**: Handling large-scale topologies (thousands of nodes)

---

## 7. Open Questions

1. **Projection Methods**: Which dimensionality reduction technique best preserves semantic relationships?
2. **Layer Interactions**: How should cross-layer relationships be weighted and visualized?
3. **Dynamic Updates**: How to handle streaming updates to topology without full recomputation?
4. **Temporal Evolution**: How to represent topology changes over time?
5. **Multi-Modal Data**: How to integrate different data types (text, images, audio) in one topology?
6. **Optimization**: What are optimal positions for nodes to minimize overlap while preserving structure?

---

## Summary

This problem model defines the challenge of creating interactive 3D visualizations of complex, high-dimensional relationships in LLM systems. The core difficulty lies in:

- **Dimensionality reduction** from high-D embeddings to 3D space
- **Multi-geometry support** across different mathematical manifolds
- **Relationship preservation** during projection
- **Scalable visualization** of potentially large graphs
- **Flexible integration** with various LLM providers

The model provides a foundation for reasoning about valid topology configurations, operations, and constraints without prescribing specific algorithmic solutions.
