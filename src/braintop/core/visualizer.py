"""3D Topology visualizer using Plotly and Dash."""

from typing import Optional, Dict, Any, List, Iterable, Union
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
    
    def create_figure(
        self,
        layers: Optional[List[str]] = None,
        include_edges: bool = True,
        color_by: str = "layer",
        size_by: str = "degree",
    ) -> go.Figure:
        """Create a Plotly figure for the topology.
        
        Returns:
            A Plotly Figure object with the 3D visualization
        """
        fig = go.Figure()

        layers_to_plot = [
            layer
            for layer in self.topology.layers
            if layers is None or layer.id in layers
        ]
        nodes_to_plot = [node for layer in layers_to_plot for node in layer.nodes]
        node_ids = {node.id for node in nodes_to_plot}
        edges_to_plot = [
            edge
            for edge in self.topology.get_all_edges()
            if include_edges and edge.source in node_ids and edge.target in node_ids
        ]
        metrics = self._compute_node_metrics(nodes_to_plot, edges_to_plot)

        # Add nodes for each layer
        for idx, layer in enumerate(layers_to_plot):
            self._add_layer_nodes(
                fig,
                layer,
                metrics=metrics,
                color_by=color_by,
                size_by=size_by,
                layer_color=self._layer_color(idx),
            )

        # Add inter-layer edges
        if include_edges:
            self._add_edges(fig, edges_to_plot)
        
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
    
    def _add_layer_nodes(
        self,
        fig: go.Figure,
        layer: Any,
        metrics: Dict[str, Dict[str, float]],
        color_by: str,
        size_by: str,
        layer_color: str,
    ) -> None:
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
            self._build_hover_text(node, layer, metrics.get(node.id, {}))
            for node in layer.nodes
        ]

        marker_sizes = self._metric_values(
            layer.nodes,
            metrics,
            size_by,
            default_value=6.0,
            scale=6.0,
            offset=6.0,
        )
        marker_colors = self._metric_values(
            layer.nodes,
            metrics,
            color_by,
            default_value=layer_color,
            scale=1.0,
        )
        marker_color_kwargs = (
            dict(color=marker_colors, colorscale="Viridis", showscale=True)
            if isinstance(marker_colors, list)
            else dict(color=marker_colors)
        )
        
        # Add scatter plot for nodes
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers+text',
            name=f"{layer.name} ({getattr(layer.layer_type, 'value', layer.layer_type)})",
            marker=dict(
                size=marker_sizes,
                opacity=0.85,
                line=dict(width=0.5, color='white'),
                **marker_color_kwargs,
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
                
                weight = edge.weight
                fig.add_trace(go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode='lines',
                    line=dict(color='rgba(120,120,120,0.6)', width=1 + weight * 2),
                    hoverinfo='skip',
                    showlegend=False,
                    opacity=0.5
                ))

    def _compute_node_metrics(
        self,
        nodes: Iterable[NodeModel],
        edges: Iterable[EdgeModel],
    ) -> Dict[str, Dict[str, float]]:
        """Compute graph metrics for nodes to enrich visualization."""
        graph = nx.Graph()
        for node in nodes:
            graph.add_node(node.id)
        for edge in edges:
            graph.add_edge(edge.source, edge.target, weight=edge.weight)

        degree_map = dict(graph.degree())
        betweenness = nx.betweenness_centrality(graph) if graph.number_of_edges() else {}
        closeness = nx.closeness_centrality(graph) if graph.number_of_edges() else {}
        metrics: Dict[str, Dict[str, float]] = {}
        for node in nodes:
            metrics[node.id] = {
                "degree": float(degree_map.get(node.id, 0.0)),
                "betweenness": float(betweenness.get(node.id, 0.0)),
                "closeness": float(closeness.get(node.id, 0.0)),
            }
        return metrics

    @staticmethod
    def _layer_color(index: int) -> str:
        colors = [
            "#4C78A8",
            "#F58518",
            "#54A24B",
            "#E45756",
            "#72B7B2",
            "#EECA3B",
            "#B279A2",
            "#FF9DA6",
            "#9D755D",
            "#BAB0AC",
        ]
        return colors[index % len(colors)]

    def _metric_values(
        self,
        nodes: List[NodeModel],
        metrics: Dict[str, Dict[str, float]],
        metric_name: str,
        default_value: Union[float, str],
        scale: float = 1.0,
        offset: float = 0.0,
    ) -> Union[List[float], str]:
        if metric_name == "layer":
            return str(default_value)
        values = []
        for node in nodes:
            value = metrics.get(node.id, {}).get(metric_name)
            if value is None:
                value = node.metadata.get(metric_name, None)
            if value is None:
                value = 0.0
            values.append(float(value) * scale + offset)
        return values

    @staticmethod
    def _build_hover_text(
        node: NodeModel,
        layer: Any,
        metrics: Dict[str, float],
    ) -> str:
        metadata_lines = [f"{key}: {value}" for key, value in node.metadata.items()]
        metric_lines = [f"{key}: {value:.3f}" for key, value in metrics.items()]
        layer_type = (
            layer.layer_type.value if hasattr(layer.layer_type, "value") else layer.layer_type
        )
        lines = [f"ID: {node.id}"]
        if node.label:
            lines.append(f"Label: {node.label}")
        lines += [
            f"Layer: {layer.name}",
            f"Layer type: {layer_type}",
        ]
        lines.extend(metric_lines)
        lines.extend(metadata_lines)
        return "<br>".join(lines)

    def create_sectional_study(
        self,
        axis: str = "z",
        sections: int = 3,
    ) -> go.Figure:
        """Create sectional studies by slicing the topology along an axis."""
        nodes = self.topology.get_all_nodes()
        if not nodes:
            return go.Figure()

        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        axis_pairs = {
            "x": (1, 2),
            "y": (0, 2),
            "z": (0, 1),
        }
        x_index, y_index = axis_pairs[axis]
        coords = np.array([node.position[axis_index] for node in nodes])
        bins = np.linspace(coords.min(), coords.max(), sections + 1)
        fig = make_subplots(
            rows=1,
            cols=sections,
            subplot_titles=[
                f"{axis.upper()} slice {i + 1}: {bins[i]:.2f}–{bins[i + 1]:.2f}"
                for i in range(sections)
            ],
        )

        for idx in range(sections):
            lower = bins[idx]
            upper = bins[idx + 1]
            slice_nodes = [
                node for node in nodes if lower <= node.position[axis_index] <= upper
            ]
            if not slice_nodes:
                continue
            x_coords = [node.position[x_index] for node in slice_nodes]
            y_coords = [node.position[y_index] for node in slice_nodes]
            fig.add_trace(
                go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode="markers+text",
                    text=[node.label for node in slice_nodes],
                    textposition="top center",
                    marker=dict(size=8, opacity=0.8),
                    hovertext=[node.id for node in slice_nodes],
                    hoverinfo="text",
                    showlegend=False,
                ),
                row=1,
                col=idx + 1,
            )

        fig.update_layout(
            title=f"Sectional Study ({axis.upper()} axis)",
            height=450,
            width=1200,
        )
        return fig

    def create_ablation_figures(self) -> Dict[str, go.Figure]:
        """Create ablation figures by removing layers or edges."""
        figures: Dict[str, go.Figure] = {}
        figures["no_edges"] = self.create_figure(include_edges=False)

        for layer in self.topology.layers:
            remaining_layers = [l.id for l in self.topology.layers if l.id != layer.id]
            figures[f"without_{layer.id}"] = self.create_figure(
                layers=remaining_layers
            )
            figures[f"only_{layer.id}"] = self.create_figure(layers=[layer.id])

        return figures
    
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
