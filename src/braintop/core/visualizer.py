"""3D Topology visualizer using Plotly and Dash."""

from typing import Optional, Dict, Any, List
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx

from braintop.core.topology import Topology, NodeModel, EdgeModel


class TopologyVisualizer:
    """Visualizer for 3D topologies with rich interactive features.
    
    This class provides methods to create beautiful 3D visualizations of
    topology structures using Plotly for rendering.
    """
    
    def __init__(self, topology: Topology, config: Optional[Dict[str, Any]] = None):
        """Initialize the visualizer.
        
        Args:
            topology: The topology to visualize
            config: Optional configuration dictionary
        """
        self.topology = topology
        self.config = config or {}
        self.figure: Optional[go.Figure] = None
    
    def create_figure(self) -> go.Figure:
        """Create a Plotly figure for the topology.
        
        Returns:
            A Plotly Figure object with the 3D visualization
        """
        fig = go.Figure()
        
        # Add nodes for each layer
        for layer in self.topology.layers:
            self._add_layer_nodes(fig, layer)
        
        # Add inter-layer edges
        self._add_edges(fig, self.topology.get_all_edges())
        
        # Configure layout
        fig.update_layout(
            title=dict(
                text=f"Topology: {self.topology.name}",
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis=dict(title='X', backgroundcolor="rgb(230, 230,230)"),
                yaxis=dict(title='Y', backgroundcolor="rgb(230, 230,230)"),
                zaxis=dict(title='Z', backgroundcolor="rgb(230, 230,230)"),
                aspectmode='cube'
            ),
            showlegend=True,
            hovermode='closest',
            width=1200,
            height=800,
        )
        
        self.figure = fig
        return fig
    
    def _add_layer_nodes(self, fig: go.Figure, layer: Any) -> None:
        """Add nodes from a layer to the figure.
        
        Args:
            fig: Plotly figure to add nodes to
            layer: Layer containing nodes to visualize
        """
        if not layer.nodes:
            return
        
        # Extract positions
        x_coords = [node.position[0] for node in layer.nodes]
        y_coords = [node.position[1] for node in layer.nodes]
        z_coords = [node.position[2] for node in layer.nodes]
        
        # Create hover text
        hover_texts = [
            f"ID: {node.id}<br>Label: {node.label}<br>Layer: {layer.name}"
            for node in layer.nodes
        ]
        
        # Add scatter plot for nodes
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers+text',
            name=f"{layer.name} ({layer.layer_type})",
            marker=dict(
                size=8,
                opacity=0.8,
                line=dict(width=0.5, color='white')
            ),
            text=[node.label for node in layer.nodes],
            textposition="top center",
            hovertext=hover_texts,
            hoverinfo='text'
        ))
    
    def _add_edges(self, fig: go.Figure, edges: List[EdgeModel]) -> None:
        """Add edges to the figure.
        
        Args:
            fig: Plotly figure to add edges to
            edges: List of edges to visualize
        """
        # Get all nodes for lookup
        all_nodes = self.topology.get_all_nodes()
        node_dict = {node.id: node for node in all_nodes}
        
        # Create edge traces
        for edge in edges:
            source_node = node_dict.get(edge.source)
            target_node = node_dict.get(edge.target)
            
            if source_node and target_node:
                x_coords = [source_node.position[0], target_node.position[0], None]
                y_coords = [source_node.position[1], target_node.position[1], None]
                z_coords = [source_node.position[2], target_node.position[2], None]
                
                fig.add_trace(go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode='lines',
                    line=dict(color='gray', width=2),
                    hoverinfo='skip',
                    showlegend=False,
                    opacity=0.5
                ))
    
    def show(self) -> None:
        """Display the visualization in a browser."""
        if self.figure is None:
            self.create_figure()
        
        if self.figure:
            self.figure.show()
    
    def save(self, filename: str, format: str = 'html') -> None:
        """Save the visualization to a file.
        
        Args:
            filename: Output filename
            format: Output format ('html', 'png', 'jpg', 'svg')
        """
        if self.figure is None:
            self.create_figure()
        
        if self.figure:
            if format == 'html':
                self.figure.write_html(filename)
            else:
                self.figure.write_image(filename, format=format)
    
    def get_network_graph(self) -> nx.Graph:
        """Convert topology to NetworkX graph for analysis.
        
        Returns:
            NetworkX Graph object
        """
        G = nx.Graph()
        
        # Add nodes
        for node in self.topology.get_all_nodes():
            G.add_node(node.id, **node.metadata)
        
        # Add edges
        for edge in self.topology.get_all_edges():
            G.add_edge(edge.source, edge.target, weight=edge.weight)
        
        return G
