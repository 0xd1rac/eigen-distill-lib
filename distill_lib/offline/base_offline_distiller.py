from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Any, List, Callable, Optional
from ..items.student import Student
from ..items.teacher import Teacher
from ..strategies.teacher_weighting import TeacherWeightingStrategy
from ..strategies.weighting_strategies import UniformWeightingStrategy

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
            x: Input batch
            labels: Ground truth labels
            device: The device on which to run the computations
            base_loss_fn: Base loss function to override self.base_loss_fn
            distill_loss_fn: Distillation loss function to override self.distill_loss_fn
            alpha: Weighting factor for the distillation loss
            temperature: Temperature for softening outputs
            teacher_weights: Weights for averaging teacher outputs
            
        Returns:
            list: A list of losses, one for each student
        """
        pass

    @abstractmethod
    def distill(self, 
                num_epochs: int,
                device: str,
                base_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None, 
                distill_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]] = None, 
                alpha: float = 0.5, 
                temperature: float = 1.0) -> list:
        """
        Perform the complete distillation process over multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            device: The device on which to run the computations
            base_loss_fn: Base loss function to override self.base_loss_fn
            distill_loss_fn: Distillation loss function to override self.distill_loss_fn
            alpha: Weighting factor for the distillation loss
            temperature: Temperature for softening outputs
            
        Returns:
            list: The average losses over the training data for each student
        """
        pass
