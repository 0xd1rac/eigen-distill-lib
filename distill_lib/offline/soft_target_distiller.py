# soft_target_distiller.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Callable, Optional, List
from torch.utils.data import DataLoader
from .base_offline_distiller import BaseOfflineDistiller
from ..items.teacher import Teacher
from ..items.student import Student
from ..strategies.teacher_weighting import TeacherWeightingStrategy
from ..strategies.weighting_strategies import UniformWeightingStrategy

def default_distill_loss_fn(student_log_probs: torch.Tensor, teacher_probs: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Default distillation loss function using F.kl_div.
    The loss is scaled by temperature^2.
    
    Args:
        student_log_probs: Log-softmax output of the student.
        teacher_probs: Softmax output of the teacher.
        temperature: Temperature scaling factor.
        
    Returns:
        The scaled KL divergence loss.
    """
    kd_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    return kd_loss * (temperature ** 2)

class SoftTargetDistiller(BaseOfflineDistiller):
    def __init__(self, 
                 students: List[Student],
                 teachers: List[Teacher],
                 weighting_strategy: Optional[TeacherWeightingStrategy] = None) -> None:
        """
        Initialize the SoftTargetDistiller.
        
        Args:
            students: List of Student instances.
            teachers: List of Teacher instances.
            weighting_strategy: Strategy for computing teacher weights (default: uniform).
        """
        super().__init__(teachers, students)
        self.weighting_strategy = weighting_strategy or UniformWeightingStrategy()
        self.base_loss_fn = nn.CrossEntropyLoss()
        self.distill_loss_fn = default_distill_loss_fn

    def compute_teacher_outputs(self, device: str, temperature: float) -> list:
        """
        Compute weighted teacher outputs for all batches using the teachers' data loaders.
        
        Returns:
            list: A list of combined teacher outputs (one per batch).
        """
        all_teacher_outputs = []
        
        with torch.no_grad():
            # Zip together batches from each teacher's data loader.
            for batch_data in zip(*[teacher.data_loader for teacher in self.teachers]):
                # Assume each teacher's batch is a tuple (inputs, labels) and we only need inputs.
                teacher_inputs = [batch[0].to(device) for batch in batch_data]
                teacher_outputs = []
                for teacher, inputs in zip(self.teachers, teacher_inputs):
                    teacher_outputs.append(teacher.model(inputs))
                
                # Compute teacher weights for this batch.
                teacher_weights = self.weighting_strategy.compute_weights(self.teachers, teacher_inputs[0], temperature)
                
                # Compute the weighted average of teacher outputs (after softening).
                batch_combined_output = sum(
                    w * F.softmax(output / temperature, dim=1)
                    for w, output in zip(teacher_weights, teacher_outputs)
                )
                all_teacher_outputs.append(batch_combined_output)
        
        return all_teacher_outputs

    def distill_step(self, 
                     x: torch.Tensor, 
                     labels: torch.Tensor, 
                     device: str,
                     base_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None, 
                     distill_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]] = None, 
                     alpha: float = 0.5, 
                     temperature: float = 1.0) -> torch.Tensor:
        """
        Perform one distillation step for a single student's batch.
        
        Args:
            x: Input batch.
            labels: Ground truth labels.
            device: Device on which to run.
            base_loss_fn: Optionally override the base loss function.
            distill_loss_fn: Optionally override the distillation loss function.
            alpha: Weighting factor between base and distillation loss.
            temperature: Temperature for softening outputs.
        
        Returns:
            The total loss (base + distillation) for the batch.
        """
        # Move inputs to device.
        x, labels = x.to(device), labels.to(device)
        
        # For the given student (assumed to be selected in the distill loop), use its model.
        # (In our design, distill() will call distill_step per student batch.)
        # Note: Here, we expect the caller to pass the appropriate teacher output.
        student = self.current_student  # Set by the caller in the distill loop.
 
        # Compute student outputs.
        student_outputs = student.model(x)
        
        # Compute the base classification loss.
        base_loss = (base_loss_fn or self.base_loss_fn)(student_outputs, labels)
        
        # Compute the distillation (KL divergence) loss.
        student_log_probs = F.log_softmax(student_outputs / temperature, dim=1)
        kd_loss_value = (distill_loss_fn or self.distill_loss_fn)(
            student_log_probs, self.current_teacher_output, temperature
        )
        
        # Combine the losses.
        total_loss = (1 - alpha) * base_loss + alpha * kd_loss_value
        
        # Backpropagation and optimizer step.
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
        Perform the complete distillation process over multiple epochs.
        
        Returns:
            list: Average loss per epoch for each student.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Move all models to the specified device.
        self.move_models_to_device(device)
        
        # Precompute teacher outputs for all batches (offline).
        teacher_outputs = self.compute_teacher_outputs(device, temperature)
        
        # Initialize storage for average losses.
        average_losses = [0.0] * len(self.students)
        
        # For each epoch:
        for epoch in range(1, num_epochs + 1):
            running_losses = [0.0] * len(self.students)
            batch_counts = [0] * len(self.students)
            
            # Zip together teacher outputs and student data loader batches.
            # We assume that each student's data_loader yields (inputs, labels).
            student_loaders = [student.data_loader for student in self.students]
            for batch_idx, (teacher_output, *student_batches) in enumerate(
                    zip(teacher_outputs, *student_loaders)):
                # Set the current teacher output for use in the distill_step.
                self.current_teacher_output = teacher_output.to(device)
                
                # Process each student's batch.
                for student_idx, (student_input, student_labels) in enumerate(student_batches):
                    # Set the current student (so distill_step can access it).
                    self.current_student = self.students[student_idx]
                    loss = self.distill_step(
                        student_input,
                        student_labels,
                        device,
                        base_loss_fn,
                        distill_loss_fn,
                        alpha,
                        temperature
                    )
                    running_losses[student_idx] += loss.item()
                    batch_counts[student_idx] += 1
                
                if (batch_idx + 1) % 100 == 0:
                    current_losses = [
                        running_losses[i] / batch_counts[i] if batch_counts[i] > 0 else 0.0
                        for i in range(len(self.students))
                    ]
                    logger.info(f"Epoch {epoch} Batch {batch_idx+1} Losses: {current_losses}")
            
            # Log average loss for the epoch.
            for i in range(len(self.students)):
                if batch_counts[i] > 0:
                    average_losses[i] = running_losses[i] / batch_counts[i]
                    logger.info(f"Epoch {epoch} Student {i+1} Average Loss: {average_losses[i]:.4f}")
                else:
                    logger.warning(f"No batches processed for Student {i+1} in epoch {epoch}")
        
        return average_losses
