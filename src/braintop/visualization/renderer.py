"""
3D Visualization Renderer for Topology Graphs.

Provides beautiful, interactive 3D visualizations using Plotly.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Optional, Dict, Any, List
from ..core.topology import TopologyGraph, TopologyLayer


class Renderer3D:
    """
    High-quality 3D renderer for topology visualizations.
    
    Creates interactive, beautiful visualizations with support
    for multiple layers and real-time updates.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the 3D renderer.
        
        Args:
            **kwargs: Renderer configuration
                - title: Visualization title
                - width: Figure width in pixels
                - height: Figure height in pixels
                - theme: Color theme ('dark', 'light', 'scientific')
        """
        self.config = kwargs
        self.title = kwargs.get("title", "BrainTop 3D Topology Visualizer")
        self.width = kwargs.get("width", 1200)
        self.height = kwargs.get("height", 800)
        self.theme = kwargs.get("theme", "dark")
        
        # Color schemes for different themes
        self.color_schemes = {
            "dark": {
                "background": "#0f0f0f",
                "grid": "#2a2a2a",
                "node": "#00d4ff",
                "edge": "#666666",
                "text": "#ffffff",
            },
            "light": {
                "background": "#ffffff",
                "grid": "#e0e0e0",
                "node": "#0066cc",
                "edge": "#999999",
                "text": "#000000",
            },
            "scientific": {
                "background": "#1a1a2e",
                "grid": "#16213e",
                "node": "#0f3460",
                "edge": "#533483",
                "text": "#eaeaea",
            },
        }
        
    def render_layer(
        self,
        layer: TopologyLayer,
        show_edges: bool = True,
        node_size: float = 8,
        edge_width: float = 2,
    ) -> go.Figure:
        """
        Render a single topology layer in 3D.
        
        Args:
            layer: The topology layer to visualize
            show_edges: Whether to draw edges between nodes
            node_size: Size of node markers
            edge_width: Width of edge lines
            
        Returns:
            Plotly Figure object
        """
        fig = go.Figure()
        
        # Compute positions
        positions = layer.compute_positions()
        if not positions:
            return self._create_empty_figure()
        
        # Get color scheme
        colors = self.color_schemes.get(self.theme, self.color_schemes["dark"])
        
        # Draw edges first (so they appear behind nodes)
        if show_edges:
            edge_traces = self._create_edge_traces(layer, positions, colors, edge_width)
            for trace in edge_traces:
                fig.add_trace(trace)
        
        # Draw nodes
        node_trace = self._create_node_trace(layer, positions, colors, node_size)
        fig.add_trace(node_trace)
        
        # Update layout
        fig.update_layout(
            title=f"{self.title} - {layer.name.capitalize()} Layer",
            width=self.width,
            height=self.height,
            showlegend=True,
            hovermode="closest",
            scene=dict(
                xaxis=dict(
                    showbackground=True,
                    backgroundcolor=colors["background"],
                    gridcolor=colors["grid"],
                    title="X",
                ),
                yaxis=dict(
                    showbackground=True,
                    backgroundcolor=colors["background"],
                    gridcolor=colors["grid"],
                    title="Y",
                ),
                zaxis=dict(
                    showbackground=True,
                    backgroundcolor=colors["background"],
                    gridcolor=colors["grid"],
                    title="Z",
                ),
                bgcolor=colors["background"],
            ),
            paper_bgcolor=colors["background"],
            font=dict(color=colors["text"]),
        )
        
        return fig
    
    def render_graph(
        self,
        graph: TopologyGraph,
        layers: Optional[List[str]] = None,
        **kwargs
    ) -> go.Figure:
        """
        Render multiple layers of a topology graph.
        
        Args:
            graph: The topology graph to visualize
            layers: List of layer names to render (all if None)
            **kwargs: Additional rendering options
            
        Returns:
            Plotly Figure with subplots for each layer
        """
        all_layers = graph.get_all_layers()
        
        if layers is None:
            layers = list(all_layers.keys())
        
        if not layers:
            return self._create_empty_figure()
        
        # Create subplots if multiple layers
        if len(layers) == 1:
            layer = all_layers[layers[0]]
            return self.render_layer(layer, **kwargs)
        
        # Create subplot layout
        rows = int(np.ceil(np.sqrt(len(layers))))
        cols = int(np.ceil(len(layers) / rows))
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f"{name.capitalize()} Layer" for name in layers],
            specs=[[{"type": "scene"}] * cols for _ in range(rows)],
            vertical_spacing=0.1,
            horizontal_spacing=0.1,
        )
        
        colors = self.color_schemes.get(self.theme, self.color_schemes["dark"])
        
        for idx, layer_name in enumerate(layers):
            if layer_name not in all_layers:
                continue
                
            layer = all_layers[layer_name]
            positions = layer.compute_positions()
            
            row = idx // cols + 1
            col = idx % cols + 1
            
            # Add traces for this layer
            edge_traces = self._create_edge_traces(layer, positions, colors, 1)
            for trace in edge_traces:
                trace.showlegend = False
                fig.add_trace(trace, row=row, col=col)
            
            node_trace = self._create_node_trace(layer, positions, colors, 6)
            node_trace.showlegend = False
            fig.add_trace(node_trace, row=row, col=col)
        
        # Update layout
        fig.update_layout(
            title=self.title,
            width=self.width,
            height=self.height,
            showlegend=False,
            paper_bgcolor=colors["background"],
            font=dict(color=colors["text"]),
        )
        
        return fig
    
    def _create_node_trace(
        self,
        layer: TopologyLayer,
        positions: Dict[str, np.ndarray],
        colors: Dict[str, str],
        size: float,
    ) -> go.Scatter3d:
        """Create a 3D scatter trace for nodes."""
        nodes = layer.get_nodes()
        
        x_coords = []
        y_coords = []
        z_coords = []
        hover_texts = []
        node_colors = []
        
        for node_id, attrs in nodes.items():
            if node_id not in positions:
                continue
                
            pos = positions[node_id]
            x_coords.append(pos[0])
            y_coords.append(pos[1])
            z_coords.append(pos[2])
            
            # Create hover text
            hover_text = f"<b>{node_id}</b><br>"
            for key, value in attrs.items():
                if key != "embedding":  # Skip large embedding arrays
                    hover_text += f"{key}: {value}<br>"
            hover_texts.append(hover_text)
            
            # Color based on attributes
            weight = attrs.get("weight", 1.0)
            node_colors.append(weight)
        
        return go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode="markers",
            name="Nodes",
            marker=dict(
                size=size,
                color=node_colors,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Weight"),
                line=dict(color=colors["text"], width=0.5),
            ),
            text=hover_texts,
            hoverinfo="text",
        )
    
    def _create_edge_traces(
        self,
        layer: TopologyLayer,
        positions: Dict[str, np.ndarray],
        colors: Dict[str, str],
        width: float,
    ) -> List[go.Scatter3d]:
        """Create 3D line traces for edges."""
        edges = layer.get_edges()
        traces = []
        
        x_coords = []
        y_coords = []
        z_coords = []
        
        for source, target, attrs in edges:
            if source not in positions or target not in positions:
                continue
            
            pos_source = positions[source]
            pos_target = positions[target]
            
            x_coords.extend([pos_source[0], pos_target[0], None])
            y_coords.extend([pos_source[1], pos_target[1], None])
            z_coords.extend([pos_source[2], pos_target[2], None])
        
        if x_coords:
            trace = go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode="lines",
                name="Edges",
                line=dict(color=colors["edge"], width=width),
                hoverinfo="none",
            )
            traces.append(trace)
        
        return traces
    
    def _create_empty_figure(self) -> go.Figure:
        """Create an empty figure with a message."""
        fig = go.Figure()
        colors = self.color_schemes.get(self.theme, self.color_schemes["dark"])
        
        fig.add_annotation(
            text="No data to visualize",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color=colors["text"]),
        )
        
        fig.update_layout(
            title=self.title,
            width=self.width,
            height=self.height,
            paper_bgcolor=colors["background"],
        )
        
        return fig
    
    def save(self, fig: go.Figure, filename: str, format: str = "html"):
        """
        Save a figure to a file.
        
        Args:
            fig: The figure to save
            filename: Output filename
            format: Output format ('html', 'png', 'jpg', 'svg', 'pdf')
        """
        if format == "html":
            fig.write_html(filename)
        else:
            fig.write_image(filename)
