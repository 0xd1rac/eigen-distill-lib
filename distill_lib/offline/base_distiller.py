# base_distiller.py
from abc import ABC, abstractmethod
from typing import List, Callable, Optional
import torch

class BaseDistiller(ABC):
    def __init__(self, teachers: List, students: List) -> None:
        """
        Initialize the distiller with lists of teacher and student instances.
        
        Args:
            teachers: List of teacher instances (each must have a .model attribute).
            students: List of student instances (each must have a .model and .optimizer attribute).
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
                     temperature: float = 1.0) -> list:
        """
        Perform one step of distillation.
        
        Returns:
            list: A list of losses (e.g. one per student) for this step.
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
        Run the complete distillation process over multiple epochs.
        
        Returns:
            list: Average losses (or other metrics) for each student.
        """
        pass

    def move_models_to_device(self, device: str) -> None:
        """
        Move teacher and student models to the specified device.
        """
        for teacher in self.teachers:
            teacher.model = teacher.model.to(device)
        for student in self.students:
            student.model = student.model.to(device)
