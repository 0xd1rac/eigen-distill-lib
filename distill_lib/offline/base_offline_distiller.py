from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Any

class BaseOfflineDistiller(ABC):
    def __init__(self, teacher: nn.Module, student: nn.Module) -> None:
        self.teacher = teacher
        self.student = student

        # Freeze the teacher model: set to eval mode and disable gradient computation.
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Ensure that the student model is trainable.
        self.student.train()
        for param in self.student.parameters():
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
