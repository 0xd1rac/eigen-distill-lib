import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .base_offline_distiller import BaseOfflineDistiller
import logging
from typing import Callable, Optional

def default_distill_loss_fn(student_log_probs: torch.Tensor, teacher_probs: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Default distillation loss function using F.kl_div.
    The loss is scaled by temperature^2.
    
    Args:
        student_log_probs (Tensor): Log-softmax output of the student.
        teacher_probs (Tensor): Softmax output of the teacher.
        temperature (float): Temperature scaling factor.
        
    Returns:
        Tensor: The KL divergence loss scaled by temperature^2.
    """
    kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    return kd_loss * (temperature ** 2)

class SoftTargetDistiller(BaseOfflineDistiller):
    def __init__(self, 
                 student: nn.Module, 
                 teacher: nn.Module,
                 optimizer: Optional[optim.Optimizer] = None,
                 learning_rate: float = 0.001) -> None:
        """
        Initializes the SoftTargetDistiller.

        Args:
            student (nn.Module): The student model.
            teacher (nn.Module): The teacher model.
            optimizer (optim.Optimizer): Optimizer for student parameters.
            learning_rate (float): Learning rate for the optimizer if not provided.
        """
        super().__init__(student, teacher)
        if optimizer is None:
            self.optimizer = optim.Adam(self.student.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer
    
        # Set default loss functions.
        self.base_loss_fn = nn.CrossEntropyLoss()
        self.distill_loss_fn = default_distill_loss_fn

    def distill_step(self, 
                     x: torch.Tensor, 
                     labels: torch.Tensor, 
                     device: str,
                     base_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None, 
                     distill_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]] = None, 
                     alpha: float = 0.5, 
                     temperature: float = 1.0) -> torch.Tensor:
        """
        Performs one step of distillation using the specified or default loss functions.

        Args:
            x (Tensor): Input batch.
            labels (Tensor): Ground truth labels.
            device (str): The device on which to run the computations.
            base_loss_fn (callable, optional): Base loss function to override self.base_loss_fn.
            distill_loss_fn (callable, optional): Distillation loss function to override self.distill_loss_fn.
            alpha (float): Weighting factor for the distillation loss.
            temperature (float): Temperature for softening outputs.
        
        Returns:
            Tensor: The combined loss.
        """
        # Move models to the specified device
        self.student.to(device)
        self.teacher.to(device)

        # Move inputs to the specified device
        x, labels = x.to(device), labels.to(device)
        
        # Choose the provided loss functions or fall back to the defaults.
        base_loss_fn = base_loss_fn if base_loss_fn is not None else self.base_loss_fn
        distill_loss_fn = distill_loss_fn if distill_loss_fn is not None else self.distill_loss_fn

        # Compute teacher outputs (without gradient computation).
        with torch.no_grad():
            teacher_outputs = self.teacher(x)
        
        # Compute student outputs.
        student_outputs = self.student(x)
        
        # Compute the base loss (e.g., classification loss).
        base_loss = base_loss_fn(student_outputs, labels)
        
        # Compute the distillation loss.
        student_log_probs = F.log_softmax(student_outputs / temperature, dim=1)
        teacher_probs = F.softmax(teacher_outputs / temperature, dim=1)
        kd_loss_value = distill_loss_fn(student_log_probs, teacher_probs, temperature)
        
        # Combine the losses.
        total_loss = (1 - alpha) * base_loss + alpha * kd_loss_value
        return total_loss

    def distill(self, 
                train_loader: torch.utils.data.DataLoader, 
                num_epochs: int, 
                device: str = 'cpu', 
                base_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None, 
                distill_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]] = None, 
                alpha: float = 0.5, 
                temperature: float = 1.0) -> float:
        """
        Performs the distillation process over multiple epochs.

        Args:
            train_loader (DataLoader): DataLoader for the training data.
            num_epochs (int): Number of epochs to train.
            device (str): The device on which to run the computations.
            base_loss_fn (callable, optional): Base loss function to override self.base_loss_fn.
            distill_loss_fn (callable, optional): Distillation loss function to override self.distill_loss_fn.
            alpha (float): Weighting factor for the distillation loss.
            temperature (float): Temperature for softening outputs.
        
        Returns:
            float: The average loss over the training data.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        for epoch in range(1, num_epochs + 1):
            running_loss = 0.0
            for batch_idx, (input, labels) in enumerate(train_loader):
                loss = self.distill_step(input, labels, device, base_loss_fn, distill_loss_fn, alpha, temperature)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if (batch_idx + 1) % 100 == 0:
                    logger.info(f"Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] Loss: {loss:.4f}")
            
            # Calculate average loss per epoch
            average_loss = running_loss / len(train_loader)
            logger.info(f"Epoch {epoch} Average Loss: {average_loss:.4f}")
        
        return average_loss
        