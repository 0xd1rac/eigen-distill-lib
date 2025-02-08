from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Any

class BaseOfflineDistiller(ABC):
    def __init__(self, teachers: list, students: list) -> None:
        self.teachers = teachers
        self.students = students

        # Freeze the teacher models: set to eval mode and disable gradient computation.
        for teacher in self.teachers:
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False

        # Ensure that the student models are trainable.
        for student in self.students:
            student.train()
            for param in student.parameters():
                param.requires_grad = True

    @abstractmethod
    def distill_step(self, 
                     x: torch.Tensor, 
                     labels: torch.Tensor, 
                     device: str, 
                     *args: Any, 
                     **kwargs: Any) -> torch.Tensor:
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
