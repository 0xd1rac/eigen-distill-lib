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
                 students: list,  # Accept a list of student models
                 teachers: list,  # Accept a list of teacher models
                 optimizers: Optional[list] = None,
                 learning_rate: float = 0.001) -> None:
        """
        Initializes the SoftTargetDistiller.

        Args:
            students (list): A list of student models.
            teachers (list): A list of teacher models.
            optimizers (list, optional): List of optimizers for each student.
            learning_rate (float): Learning rate for the optimizer if not provided.
        """
        super().__init__(teachers, students)  # Pass lists of teachers and students to the base class
        self.students = students  # Store all students
        self.teachers = teachers  # Store all teachers

        if optimizers is None:
            self.optimizers = [optim.Adam(student.parameters(), lr=learning_rate) for student in students]
        else:
            self.optimizers = optimizers
    
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
                     temperature: float = 1.0,
                     teacher_weights: Optional[list] = None) -> list:
        """
        Performs one step of distillation for each student using the specified or default loss functions.

        Args:
            x (Tensor): Input batch.
            labels (Tensor): Ground truth labels.
            device (str): The device on which to run the computations.
            base_loss_fn (callable, optional): Base loss function to override self.base_loss_fn.
            distill_loss_fn (callable, optional): Distillation loss function to override self.distill_loss_fn.
            alpha (float): Weighting factor for the distillation loss.
            temperature (float): Temperature for softening outputs.
            teacher_weights (list, optional): Weights for averaging teacher outputs.
        
        Returns:
            list: A list of combined losses for each student.
        """
        # Move models to the specified device
        for student in self.students:
            student.to(device)
        for teacher in self.teachers:
            teacher.to(device)

        # Move inputs to the specified device
        x, labels = x.to(device), labels.to(device)
        
        # Choose the provided loss functions or fall back to the defaults.
        base_loss_fn = base_loss_fn if base_loss_fn is not None else self.base_loss_fn
        distill_loss_fn = distill_loss_fn if distill_loss_fn is not None else self.distill_loss_fn

        # Compute teacher outputs and average them
        teacher_outputs = []
        with torch.no_grad():
            for teacher in self.teachers:
                teacher_outputs.append(teacher(x))
        
        # Average teacher outputs
        if teacher_weights is None:
            teacher_weights = [1.0 / len(self.teachers)] * len(self.teachers)
        combined_teacher_output = sum(w * F.softmax(output / temperature, dim=1) for w, output in zip(teacher_weights, teacher_outputs))
        
        # Compute losses for each student
        total_losses = []
        for student, optimizer in zip(self.students, self.optimizers):
            # Compute student outputs.
            student_outputs = student(x)
            
            # Compute the base loss (e.g., classification loss).
            base_loss = base_loss_fn(student_outputs, labels)
            
            # Compute the distillation loss.
            student_log_probs = F.log_softmax(student_outputs / temperature, dim=1)
            kd_loss_value = distill_loss_fn(student_log_probs, combined_teacher_output, temperature)
            
            # Combine the losses.
            total_loss = (1 - alpha) * base_loss + alpha * kd_loss_value
            total_losses.append(total_loss)

            # Backpropagation and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        return total_losses

    def distill(self, 
                teacher_train_loader: torch.utils.data.DataLoader, 
                student_train_loader: torch.utils.data.DataLoader, 
                num_epochs: int, 
                device: str = 'cpu', 
                base_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None, 
                distill_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]] = None, 
                alpha: float = 0.5, 
                temperature: float = 1.0) -> list:
        """
        Performs the distillation process over multiple epochs.

        Args:
            teacher_train_loader (DataLoader): DataLoader for the teacher's training data.
            student_train_loader (DataLoader): DataLoader for the student's training data.
            num_epochs (int): Number of epochs to train.
            device (str): The device on which to run the computations.
            base_loss_fn (callable, optional): Base loss function to override self.base_loss_fn.
            distill_loss_fn (callable, optional): Distillation loss function to override self.distill_loss_fn.
            alpha (float): Weighting factor for the distillation loss.
            temperature (float): Temperature for softening outputs.
        
        Returns:
            list: The average loss over the training data for each student.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        average_losses = [0.0] * len(self.students)

        for epoch in range(1, num_epochs + 1):
            running_losses = [0.0] * len(self.students)
            for batch_idx, (teacher_batch, student_batch) in enumerate(zip(teacher_train_loader, student_train_loader)):
                if len(teacher_batch) != len(student_batch):
                    raise ValueError("Teacher and student data loaders must have the same length.")
                
                input, labels = student_batch
                losses = self.distill_step(input, labels, device, base_loss_fn, distill_loss_fn, alpha, temperature)
                for i, loss in enumerate(losses):
                    running_losses[i] += loss.item()
                if (batch_idx + 1) % 100 == 0:
                    logger.info(f"Epoch {epoch} [{batch_idx+1}/{len(student_train_loader)}] Losses: {[loss.item() for loss in losses]}")
            
            # Calculate average loss per epoch for each student
            for i in range(len(self.students)):
                average_losses[i] = running_losses[i] / len(student_train_loader)
                logger.info(f"Epoch {epoch} Student {i+1} Average Loss: {average_losses[i]:.4f}")
        
        return average_losses
        