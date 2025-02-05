import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from distill_lib.offline.base_offline_distiller import BaseOfflineDistiller

def default_distill_loss_fn(student_log_probs, teacher_probs, temperature):
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
                 optimizer: optim.Optimizer = None,
                 ):
        """
        Initializes the SoftTargetDistiller.

        Args:
            student (nn.Module): The student model.
            teacher (nn.Module): The teacher model.
            optimizer (optim.Optimizer): Optimizer for student parameters.
        """
        super().__init__(student, teacher)
        self.optimizer = optimizer

        # Freeze teacher parameters.
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Set default loss functions.
        self.base_loss_fn = nn.CrossEntropyLoss()
        self.distill_loss_fn = default_distill_loss_fn

    def distill_step(self, x, 
                     labels, 
                     device,
                     base_loss_fn=None, 
                     distill_loss_fn=None, 
                     alpha=0.5, 
                     temperature=1.0):
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
        # Choose the provided loss functions or fall back to the defaults.
        base_loss_fn = base_loss_fn if base_loss_fn is not None else self.base_loss_fn
        distill_loss_fn = distill_loss_fn if distill_loss_fn is not None else self.distill_loss_fn

        # Set student model to training mode and move inputs to device.
        self.student.train()
        x, labels = x.to(device), labels.to(device)
        
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

    def train_step(self, x, labels, 
                   base_loss_fn=None, 
                   distill_loss_fn=None, 
                   alpha=0.5, 
                   temperature=1.0):
        """
        Performs a full training step: forward pass, loss computation, backward pass, and optimizer update.

        Args:
            x (Tensor): Input batch.
            labels (Tensor): Ground truth labels.
            base_loss_fn (callable, optional): Base loss function.
            distill_loss_fn (callable, optional): Distillation loss function.
            alpha (float): Weighting factor for the distillation loss.
            temperature (float): Temperature for softening outputs.
        
        Returns:
            float: The loss value for this training step.
        """
        # NOTE: Here we assume that the device is passed via the caller of distill_step.
        # You could also store device as an attribute if preferred.
        # For this example, we extract the device from the student's parameters.
        for param in self.student.parameters():
            param.requires_grad = True
        
        device = next(self.student.parameters()).device
        
        loss = self.distill_step(x, labels, device, base_loss_fn, distill_loss_fn, alpha, temperature)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

# Example usage:
# if __name__ == "__main__":
#     # Define dummy teacher and student models (replace these with your actual models).
#     teacher_model = nn.Sequential(
#         nn.Conv2d(3, 16, kernel_size=3, padding=1),
#         nn.ReLU(),
#         nn.Flatten(),
#         nn.Linear(16 * 224 * 224, 10)
#     )
#     student_model = nn.Sequential(
#         nn.Conv2d(3, 8, kernel_size=3, padding=1),
#         nn.ReLU(),
#         nn.Flatten(),
#         nn.Linear(8 * 224 * 224, 10)
#     )
    
#     # Create an optimizer for the student.
#     optimizer = optim.SGD(student_model.parameters(), lr=0.01, momentum=0.9)
    
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     teacher_model.to(device)
#     student_model.to(device)
    
#     # Instantiate the distiller.
#     distiller = SoftTargetDistiller(student_model, teacher_model, optimizer=optimizer)
    
#     # Dummy data (replace with your actual data loader).
#     dummy_inputs = torch.randn(32, 3, 224, 224)
#     dummy_labels = torch.randint(0, 10, (32,))
    
#     # Option 1: Use the default loss functions.
#     loss_value_default = distiller.train_step(dummy_inputs, dummy_labels, alpha=0.5, temperature=2.0)
    
#     # Option 2: Specify custom loss functions on the fly.
#     # (For example, here we use a custom base loss function; you can do the same for distill_loss_fn.)
#     custom_base_loss = nn.CrossEntropyLoss()  # Replace with a different loss if desired.
#     loss_value_custom = distiller.train_step(dummy_inputs, dummy_labels, 
#                                              base_loss_fn=custom_base_loss,
#                                              distill_loss_fn=default_distill_loss_fn,
#                                              alpha=0.5, temperature=2.0)
    
#     print("Default loss value:", loss_value_default)
#     print("Custom loss value:", loss_value_custom)
