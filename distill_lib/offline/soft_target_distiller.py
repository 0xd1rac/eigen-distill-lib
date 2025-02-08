import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .base_offline_distiller import BaseOfflineDistiller
import logging
from typing import Callable, Optional, List
from ..items.teacher import Teacher
from ..items.student import Student
from ..strategies.teacher_weighting import TeacherWeightingStrategy
from ..strategies.weighting_strategies import UniformWeightingStrategy
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
                 students: List[Student],
                 teachers: List[Teacher],
                 weighting_strategy: TeacherWeightingStrategy = None) -> None:
        """
        Initializes the SoftTargetDistiller.

        Args:
            students: List of Student instances
            teachers: List of Teacher instances
            weighting_strategy: Strategy for computing teacher weights (defaults to uniform weighting)
        """
        super().__init__(teachers, students)
        
        # Set default weighting strategy if none provided
        if weighting_strategy is None:
            self.weighting_strategy = UniformWeightingStrategy()
        else:
            self.weighting_strategy = weighting_strategy
    
        # Set default loss functions
        self.base_loss_fn = nn.CrossEntropyLoss()
        self.distill_loss_fn = default_distill_loss_fn

    def distill_step(self, 
                     x: torch.Tensor, 
                     labels: torch.Tensor, 
                     device: str,
                     base_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None, 
                     distill_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]] = None, 
                     alpha: float = 0.5, 
                     temperature: float = 1.0) -> list:
        """
        Performs one step of distillation for each student using the specified or default loss functions.

        Args:
            x: Input batch
            labels: Ground truth labels
            device: The device on which to run the computations
            base_loss_fn: Base loss function to override self.base_loss_fn
            distill_loss_fn: Distillation loss function to override self.distill_loss_fn
            alpha: Weighting factor for the distillation loss
            temperature: Temperature for softening outputs
        
        Returns:
            list: A list of combined losses for each student
        """
        # Move inputs to the specified device
        x, labels = x.to(device), labels.to(device)
        
        # Choose the provided loss functions or fall back to the defaults
        base_loss_fn = base_loss_fn if base_loss_fn is not None else self.base_loss_fn
        distill_loss_fn = distill_loss_fn if distill_loss_fn is not None else self.distill_loss_fn

        # Move teacher models to device and compute outputs
        teacher_outputs = []
        with torch.no_grad():
            for teacher in self.teachers:
                teacher.model.to(device)
                teacher_outputs.append(teacher.model(x))
        
        # Get teacher weights from the strategy
        teacher_weights = self.weighting_strategy.compute_weights(self.teachers, x, temperature)
        
        # Compute weighted average of teacher outputs
        combined_teacher_output = sum(w * F.softmax(output / temperature, dim=1) 
                                    for w, output in zip(teacher_weights, teacher_outputs))
        
        # Compute losses for each student
        total_losses = []
        for i, student in enumerate(self.students):
            # Move student model to device
            student.model.to(device)
            
            # Compute student outputs
            student_outputs = student.model(x)
            
            # Compute the base loss (e.g., classification loss)
            base_loss = base_loss_fn(student_outputs, labels)
            
            # Compute the distillation loss
            student_log_probs = F.log_softmax(student_outputs / temperature, dim=1)
            kd_loss_value = distill_loss_fn(student_log_probs, combined_teacher_output, temperature)
            
            # Combine the losses
            total_loss = (1 - alpha) * base_loss + alpha * kd_loss_value
            total_losses.append(total_loss)

            # Backpropagation and optimization
            student.optimizer.zero_grad()
            if i < len(self.students) - 1:
                total_loss.backward(retain_graph=True)
            else:
                total_loss.backward()
            student.optimizer.step()

        return total_losses

    def distill(self, 
                num_epochs: int, 
                device: str,
                base_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None, 
                distill_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]] = None, 
                alpha: float = 0.5, 
                temperature: float = 1.0) -> list:
        """
        Performs the distillation process over multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            device: The device on which to run the computations
            base_loss_fn: Base loss function to override self.base_loss_fn
            distill_loss_fn: Distillation loss function to override self.distill_loss_fn
            alpha: Weighting factor for the distillation loss
            temperature: Temperature for softening outputs
        
        Returns:
            list: The average loss over the training data for each student
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        average_losses = [0.0] * len(self.students)

        for epoch in range(1, num_epochs + 1):
            running_losses = [0.0] * len(self.students)
            
            # Use the first student's data loader length as reference
            num_batches = len(self.students[0].data_loader)
            
            for batch_idx, batch_data in enumerate(zip(*[teacher.data_loader for teacher in self.teachers], 
                                                      *[student.data_loader for student in self.students])):
                # Split batch data into teacher and student batches
                num_teachers = len(self.teachers)
                teacher_batches = batch_data[:num_teachers]
                student_batches = batch_data[num_teachers:]

                # Verify all batches have the same size
                batch_sizes = [len(batch[0]) for batch in batch_data]
                if len(set(batch_sizes)) != 1:
                    raise ValueError("All data loaders must have the same batch size")

                # Use first student batch
                input, labels = student_batches[0]
                losses = self.distill_step(input, labels, device, base_loss_fn, distill_loss_fn, alpha, temperature)
                
                for i, loss in enumerate(losses):
                    running_losses[i] += loss.item()
                
                if (batch_idx + 1) % 100 == 0:
                    logger.info(f"Epoch {epoch} [{batch_idx+1}/{num_batches}] Losses: {[loss.item() for loss in losses]}")
            
            # Calculate average loss per epoch for each student
            for i in range(len(self.students)):
                average_losses[i] = running_losses[i] / num_batches
                logger.info(f"Epoch {epoch} Student {i+1} Average Loss: {average_losses[i]:.4f}")
        
        return average_losses
        