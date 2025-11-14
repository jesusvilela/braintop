# Architecture Guide

## Overview

Braintop is designed with a modular, loosely coupled architecture that allows easy extension and integration with various LLM systems, agents, and orchestrators.

## Core Components

### 1. Core Module (`braintop.core`)

**Purpose**: Provides the fundamental data structures and visualization engine.

**Key Classes**:
- `Topology`: Main container for the entire topology structure
- `LayerModel`: Represents a single layer in the topology
- `NodeModel`: Represents a node/vertex in the topology
- `EdgeModel`: Represents an edge/connection between nodes
- `TopologyVisualizer`: Renders topologies as interactive 3D visualizations

### 2. Layers Module (`braintop.layers`)

**Purpose**: Implements different types of geometric layers.

**Key Classes**:
- `BaseLayer`: Abstract base class for all layer types
- `ConceptualLayer`: Projects high-dimensional semantic data to 3D
- `RiemannianLayer`: Embeds data on curved manifolds

**Extensibility**: Create custom layers by extending `BaseLayer`.

### 3. Integrations Module (`braintop.integrations`)

**Purpose**: Connects braintop to external systems like LLMs and agents.

**Key Classes**:
- `BaseIntegration`: Abstract interface for integrations
- `LLMConnector`: Connects to LLM providers to extract embeddings

**Extensibility**: Create custom integrations by extending `BaseIntegration`.

### 4. Utils Module (`braintop.utils`)

**Purpose**: Provides helper utilities and builder patterns.

**Key Classes**:
- `TopologyBuilder`: Fluent interface for constructing complex topologies

## Design Principles

### 1. Modularity
Each component is self-contained with clear interfaces. Layers, integrations, and visualizers can be swapped or extended independently.

### 2. Loose Coupling
Components communicate through well-defined interfaces (Pydantic models and abstract base classes), minimizing dependencies.

### 3. Extensibility
Base classes provide extension points for custom implementations.

### 4. Type Safety
Uses Pydantic for data validation and type checking, ensuring data consistency.
