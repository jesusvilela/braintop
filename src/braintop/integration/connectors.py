"""
LLM Integration - Interface for connecting to LLMs, agents, and orchestrators.

This module provides a loosely coupled interface for integrating with
various LLM systems and frameworks.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from ..core.topology import TopologyGraph, TopologyLayer


class LLMConnector(ABC):
    """
    Abstract base class for LLM connectors.
    
    Defines the interface for extracting topology information
    from different LLM systems.
    """
    
    def __init__(self, name: str):
        """
        Initialize the connector.
        
        Args:
            name: Unique identifier for this connector
        """
        self.name = name
        
    @abstractmethod
    def extract_topology(self) -> TopologyGraph:
        """
        Extract topology information from the LLM system.
        
        Returns:
            A TopologyGraph representing the system structure
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the connected LLM system.
        
        Returns:
            Dictionary with system information
        """
        pass


class AgentConnector(LLMConnector):
    """
    Connector for LLM agents.
    
    Extracts topology showing agent interactions and state.
    """
    
    def __init__(self, name: str = "agent", **kwargs):
        """
        Initialize the agent connector.
        
        Args:
            name: Connector identifier
            **kwargs: Configuration options
                - agent_config: Agent configuration dictionary
                - state_provider: Function to get current agent state
        """
        super().__init__(name)
        self.config = kwargs
        
    def extract_topology(self) -> TopologyGraph:
        """
        Extract topology from agent system.
        
        Returns:
            TopologyGraph with agent nodes and interactions
        """
        graph = TopologyGraph()
        
        # This is a placeholder - in real use, would connect to actual agent system
        # Users should subclass and implement for their specific agent framework
        
        return graph
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get agent system metadata."""
        return {
            "connector_type": "agent",
            "name": self.name,
            "config": self.config,
        }
    
    def add_agent_node(
        self,
        graph: TopologyGraph,
        layer_name: str,
        agent_id: str,
        **attributes
    ):
        """
        Helper method to add an agent node to a layer.
        
        Args:
            graph: The topology graph
            layer_name: Name of the layer to add to
            agent_id: Unique identifier for the agent
            **attributes: Agent attributes
        """
        layer = graph.get_layer(layer_name)
        if layer:
            layer.add_node(agent_id, **attributes)


class OrchestratorConnector(LLMConnector):
    """
    Connector for LLM orchestrators.
    
    Extracts topology showing orchestration flow and dependencies.
    """
    
    def __init__(self, name: str = "orchestrator", **kwargs):
        """
        Initialize the orchestrator connector.
        
        Args:
            name: Connector identifier
            **kwargs: Configuration options
                - orchestrator_config: Orchestrator configuration
                - workflow_provider: Function to get workflow state
        """
        super().__init__(name)
        self.config = kwargs
        
    def extract_topology(self) -> TopologyGraph:
        """
        Extract topology from orchestrator system.
        
        Returns:
            TopologyGraph with workflow nodes and dependencies
        """
        graph = TopologyGraph()
        
        # This is a placeholder - in real use, would connect to actual orchestrator
        # Users should subclass and implement for their specific orchestrator
        
        return graph
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get orchestrator system metadata."""
        return {
            "connector_type": "orchestrator",
            "name": self.name,
            "config": self.config,
        }
    
    def add_task_node(
        self,
        graph: TopologyGraph,
        layer_name: str,
        task_id: str,
        **attributes
    ):
        """
        Helper method to add a task node to a layer.
        
        Args:
            graph: The topology graph
            layer_name: Name of the layer to add to
            task_id: Unique identifier for the task
            **attributes: Task attributes
        """
        layer = graph.get_layer(layer_name)
        if layer:
            layer.add_node(task_id, **attributes)


class TransformerConnector(LLMConnector):
    """
    Connector for transformer-based LLMs.
    
    Extracts architecture topology from transformer models.
    """
    
    def __init__(self, name: str = "transformer", **kwargs):
        """
        Initialize the transformer connector.
        
        Args:
            name: Connector identifier
            **kwargs: Configuration options
                - model: The transformer model object
                - attention_weights: Whether to include attention weights
        """
        super().__init__(name)
        self.config = kwargs
        self.model = kwargs.get("model")
        
    def extract_topology(self) -> TopologyGraph:
        """
        Extract topology from transformer model.
        
        Returns:
            TopologyGraph with model architecture
        """
        from ..layers.neural import NeuralLayer
        
        graph = TopologyGraph()
        
        # Create a neural layer for the model architecture
        neural_layer = NeuralLayer("transformer_architecture")
        
        # This is a placeholder - in real use, would introspect actual model
        # Users should subclass and implement for their specific model
        
        graph.add_layer(neural_layer)
        return graph
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get transformer model metadata."""
        metadata = {
            "connector_type": "transformer",
            "name": self.name,
        }
        
        if self.model is not None:
            # Try to extract model info if available
            try:
                metadata["model_type"] = type(self.model).__name__
            except:
                pass
        
        return metadata


class DataTransformer:
    """
    Utility class for transforming data into topology representations.
    
    Provides methods to convert various data formats into topology graphs.
    """
    
    @staticmethod
    def from_dict(
        data: Dict[str, Any],
        layer_type: str = "conceptual"
    ) -> TopologyGraph:
        """
        Create a topology graph from a dictionary structure.
        
        Args:
            data: Dictionary with 'nodes' and 'edges' keys
            layer_type: Type of layer to create
            
        Returns:
            TopologyGraph with the data
        """
        from ..layers.conceptual import ConceptualLayer
        from ..layers.riemannian import RiemannianLayer
        from ..layers.neural import NeuralLayer
        
        graph = TopologyGraph()
        
        # Select layer class
        if layer_type == "conceptual":
            layer = ConceptualLayer()
        elif layer_type == "riemannian":
            layer = RiemannianLayer()
        elif layer_type == "neural":
            layer = NeuralLayer()
        else:
            layer = ConceptualLayer()
        
        # Add nodes
        if "nodes" in data:
            for node_data in data["nodes"]:
                node_id = node_data.get("id", str(node_data))
                attrs = {k: v for k, v in node_data.items() if k != "id"}
                layer.add_node(node_id, **attrs)
        
        # Add edges
        if "edges" in data:
            for edge_data in data["edges"]:
                source = edge_data.get("source")
                target = edge_data.get("target")
                attrs = {k: v for k, v in edge_data.items() 
                        if k not in ("source", "target")}
                if source and target:
                    layer.add_edge(source, target, **attrs)
        
        graph.add_layer(layer)
        return graph
    
    @staticmethod
    def from_networkx(nx_graph, layer_type: str = "conceptual") -> TopologyGraph:
        """
        Create a topology graph from a NetworkX graph.
        
        Args:
            nx_graph: NetworkX graph object
            layer_type: Type of layer to create
            
        Returns:
            TopologyGraph with the NetworkX data
        """
        from ..layers.conceptual import ConceptualLayer
        from ..layers.riemannian import RiemannianLayer
        from ..layers.neural import NeuralLayer
        
        graph = TopologyGraph()
        
        # Select layer class
        if layer_type == "conceptual":
            layer = ConceptualLayer()
        elif layer_type == "riemannian":
            layer = RiemannianLayer()
        elif layer_type == "neural":
            layer = NeuralLayer()
        else:
            layer = ConceptualLayer()
        
        # Add nodes
        for node_id, attrs in nx_graph.nodes(data=True):
            layer.add_node(str(node_id), **attrs)
        
        # Add edges
        for source, target, attrs in nx_graph.edges(data=True):
            layer.add_edge(str(source), str(target), **attrs)
        
        graph.add_layer(layer)
        return graph
