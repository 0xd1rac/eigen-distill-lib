from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Any, List
from ..items.student import Student
from ..items.teacher import Teacher
from typing import Callable, Optional, List

class BaseOfflineDistiller(ABC):
    def __init__(self, teachers: List[Teacher], students: List[Student]) -> None:
        """
        Initialize the base offline distiller with lists of Teacher and Student instances.
        
        Args:
            teachers: List of Teacher instances
            students: List of Student instances
        """
        self.teachers = teachers
        self.students = students


    @abstractmethod
    def distill_step(self, 
                     x: torch.Tensor, 
                     labels: torch.Tensor, 
                     device: str,
                     base_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None, 
                     distill_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]] = None, 
                     alpha: float = 0.5, 
                     temperature: float = 1.0,
                     teacher_weights: Optional[list] = None) -> list:
        """
        Perform one step of distillation.

        Args:
            x (Tensor): Input batch.
            labels (Tensor): Ground truth labels.
            device (str): The device on which to run the computations.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
            
        Returns:
            Tensor: The loss value or other metrics.
        """
        pass
