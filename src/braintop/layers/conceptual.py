"""
Conceptual Layer - Abstract concept topology visualization.

This layer represents abstract concepts and their relationships,
useful for visualizing semantic spaces and conceptual hierarchies.
"""

import numpy as np
from typing import Dict, Any
from ..core.topology import TopologyLayer


class ConceptualLayer(TopologyLayer):
    """
    Topology layer for conceptual/semantic relationships.
    
    Uses hierarchical and force-directed positioning to represent
    abstract concepts and their semantic distances.
    """
    
    def __init__(self, name: str = "conceptual", **kwargs):
        """
        Initialize the conceptual layer.
        
        Args:
            name: Layer identifier
            **kwargs: Additional configuration
                - hierarchy_strength: How strongly to enforce hierarchy (0-1)
                - spread_factor: Controls node spacing
        """
        super().__init__(name, **kwargs)
        self.hierarchy_strength = kwargs.get("hierarchy_strength", 0.7)
        self.spread_factor = kwargs.get("spread_factor", 2.0)
        
    def compute_positions(self) -> Dict[str, np.ndarray]:
        """
        Compute 3D positions based on conceptual relationships.
        
        Uses a combination of hierarchy level and semantic clustering.
        
        Returns:
            Dictionary mapping node IDs to 3D position arrays
        """
        positions = {}
        nodes = self.get_nodes()
        
        if not nodes:
            return positions
        
        # Compute hierarchy levels
        levels = self._compute_hierarchy_levels()
        
        # For each level, arrange nodes in a circular pattern
        for node_id, attrs in nodes.items():
            level = levels.get(node_id, 0)
            
            # Get node index within its level
            level_nodes = [n for n, l in levels.items() if l == level]
            node_idx = level_nodes.index(node_id)
            num_at_level = len(level_nodes)
            
            # Circular arrangement with hierarchy
            angle = 2 * np.pi * node_idx / max(num_at_level, 1)
            radius = self.spread_factor * (1 + level * 0.5)
            
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = -level * 1.5  # Hierarchy goes downward
            
            # Add some semantic clustering noise
            semantic_weight = attrs.get("weight", 1.0)
            x += np.random.randn() * 0.2 * semantic_weight
            y += np.random.randn() * 0.2 * semantic_weight
            
            positions[node_id] = np.array([x, y, z])
        
        return positions
    
    def compute_geometry(self) -> Dict[str, Any]:
        """
        Compute conceptual layer geometry metadata.
        
        Returns:
            Dictionary with geometric properties
        """
        nodes = self.get_nodes()
        edges = self.get_edges()
        levels = self._compute_hierarchy_levels()
        
        return {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "num_levels": len(set(levels.values())) if levels else 0,
            "max_depth": max(levels.values()) if levels else 0,
            "density": len(edges) / max(len(nodes) * (len(nodes) - 1) / 2, 1),
            "type": "conceptual",
        }
    
    def _compute_hierarchy_levels(self) -> Dict[str, int]:
        """
        Compute hierarchy level for each node.
        
        Returns:
            Dictionary mapping node IDs to hierarchy levels
        """
        levels = {}
        nodes = self.get_nodes()
        edges = self.get_edges()
        
        # Build adjacency for hierarchy traversal
        children = {node_id: [] for node_id in nodes}
        parents = {node_id: [] for node_id in nodes}
        
        for source, target, attrs in edges:
            edge_type = attrs.get("type", "related")
            if edge_type == "parent_of" or edge_type == "contains":
                children[source].append(target)
                parents[target].append(source)
            elif edge_type == "child_of":
                parents[source].append(target)
                children[target].append(source)
        
        # Find root nodes (no parents)
        roots = [node_id for node_id, p in parents.items() if not p]
        if not roots:
            # If no clear roots, use nodes with most connections as roots
            roots = [list(nodes.keys())[0]] if nodes else []
        
        # BFS to assign levels
        visited = set()
        queue = [(root, 0) for root in roots]
        
        while queue:
            node_id, level = queue.pop(0)
            if node_id in visited:
                continue
                
            visited.add(node_id)
            levels[node_id] = level
            
            for child in children.get(node_id, []):
                if child not in visited:
                    queue.append((child, level + 1))
        
        # Assign level 0 to any unvisited nodes
        for node_id in nodes:
            if node_id not in levels:
                levels[node_id] = 0
        
        return levels
