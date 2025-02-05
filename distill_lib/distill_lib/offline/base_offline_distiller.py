from abc import ABC, abstractmethod
import torch.nn as nn

class BaseOfflineDistiller(ABC):
    def __init__(self, teacher: nn.Module, student: nn.Module):
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
    def distill_step(self, data):
        """
        Perform one step of distillation.
        
        Args:
            data: Input data required for the distillation step.
            
        Returns:
            The loss value or other metrics.
        """
        pass
