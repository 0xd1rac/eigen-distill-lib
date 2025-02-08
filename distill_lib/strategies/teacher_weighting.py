from abc import ABC, abstractmethod
import torch
from typing import List
from ..items.teacher import Teacher

class TeacherWeightingStrategy(ABC):
    """Abstract base class for teacher weighting strategies."""
    
    @abstractmethod
    def compute_weights(self, 
                       teachers: List[Teacher], 
                       x: torch.Tensor,
                       temperature: float = 1.0) -> List[float]:
        """
        Compute weights for each teacher based on the strategy.
        
        Args:
            teachers: List of teacher models
            x: Input batch to compute weights for
            temperature: Temperature parameter for softening weights
            
        Returns:
            List of weights, one for each teacher
        """
        pass 