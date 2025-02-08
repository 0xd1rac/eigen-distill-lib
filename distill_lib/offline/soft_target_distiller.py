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
from torch.utils.data import DataLoader

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
                 weighting_strategy: Optional[TeacherWeightingStrategy] = None
                 ) -> None:
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

    def compute_teacher_outputs(self, device: str, temperature: float) -> List[List[torch.Tensor]]:
        """
        Compute and combine teacher outputs for all batches using each teacher's dataloader.
        
        Args:
            device: The device to compute on
            temperature: Temperature for softening outputs
            
        Returns:
            List[List[Tensor]]: List of combined teacher outputs for each batch for each teacher
        """
        all_teacher_outputs = []
        
        # Move teacher models to device once, outside the batch loop
        for teacher in self.teachers:
            teacher.model.to(device)
        
        with torch.no_grad():
            # Iterate through all teachers' dataloaders in parallel
            for batch_data in zip(*[teacher.data_loader for teacher in self.teachers]):
                # Get inputs from each teacher's batch
                teacher_inputs = [batch[0].to(device) for batch in batch_data]
                
                # Compute outputs for each teacher
                teacher_outputs = []
                for teacher, inputs in zip(self.teachers, teacher_inputs):
                    teacher_outputs.append(teacher.model(inputs))
                
                # Get teacher weights for this batch
                teacher_weights = self.weighting_strategy.compute_weights(self.teachers, teacher_inputs[0], temperature)
                
                # Compute weighted average of teacher outputs for this batch
                batch_combined_output = sum(w * F.softmax(output / temperature, dim=1) 
                                         for w, output in zip(teacher_weights, teacher_outputs))
                
                all_teacher_outputs.append(batch_combined_output)
        
        return all_teacher_outputs

    def distill_step(self, 
                     x: torch.Tensor, 
                     labels: torch.Tensor, 
                     student_idx: int,
                     device: str,
                     combined_teacher_output: torch.Tensor,
                     base_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None, 
                     distill_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]] = None, 
                     alpha: float = 0.5, 
                     temperature: float = 1.0) -> torch.Tensor:
        """
        Performs one step of distillation for a single student using the specified or default loss functions.
        """
        # Move inputs to the specified device
        x, labels = x.to(device), labels.to(device)
        
        # Choose the provided loss functions or fall back to the defaults
        base_loss_fn = base_loss_fn if base_loss_fn is not None else self.base_loss_fn
        distill_loss_fn = distill_loss_fn if distill_loss_fn is not None else self.distill_loss_fn
        
        # Get the current student
        student = self.students[student_idx]
        
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

        # Backpropagation and optimization
        student.optimizer.zero_grad()
        total_loss.backward()
        student.optimizer.step()

        return total_loss

    def distill(self, 
                num_epochs: int, 
                device: str,
                base_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None, 
                distill_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]] = None, 
                alpha: float = 0.5, 
                temperature: float = 1.0) -> list:
        """
        Performs the distillation process over multiple epochs.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Compute teacher outputs for all batches
        teacher_outputs = self.compute_teacher_outputs(device, temperature)

        # Keep track of the average loss for each student
        average_losses = [0.0] * len(self.students)

        for epoch in range(1, num_epochs + 1):
            running_losses = [0.0] * len(self.students)
            batch_counts = [0] * len(self.students)
            
            for batch_idx, (teacher_output, *student_batches) in enumerate(zip(teacher_outputs, 
                                                  *[student.data_loader for student in self.students])):
                
                    # Process each student's batch
                for student_idx, (student_input, student_labels) in enumerate(student_batches):
                    loss = self.distill_step(
                        student_input, 
                        student_labels, 
                        student_idx, 
                        device,
                        teacher_output,
                        base_loss_fn, distill_loss_fn, 
                        alpha, temperature
                    )
                    
                    running_losses[student_idx] += loss.item()
                    batch_counts[student_idx] += 1
                

                if (batch_idx + 1) % 100 == 0:
                    current_losses = [loss / count if count > 0 else 0.0 
                                    for loss, count in zip(running_losses, batch_counts)]
                    logger.info(f"Epoch {epoch} [{batch_idx+1}] Losses: {current_losses}")
            
            # Calculate average loss per epoch for each student
            for i in range(len(self.students)):
                if batch_counts[i] > 0:
                    average_losses[i] = running_losses[i] / batch_counts[i]
                    logger.info(f"Epoch {epoch} Student {i+1} Average Loss: {average_losses[i]:.4f}")
                else:
                    logger.warning(f"No batches processed for Student {i+1} in epoch {epoch}")
        
        return average_losses

         