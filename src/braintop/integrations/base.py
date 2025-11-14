"""Base integration interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class BaseIntegration(ABC):
    """Abstract base class for all integrations.
    
    This class defines the interface for connecting braintop to external
    systems like LLMs, agents, and orchestrators.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the integration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.is_connected = False
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the external system.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the external system."""
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from the system.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the connected model.
        
        Returns:
            Dictionary with model information
        """
        pass
