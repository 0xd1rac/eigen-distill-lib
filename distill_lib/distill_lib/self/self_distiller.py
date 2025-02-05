import torch
import torch.nn as nn
import torch.optim as optim
from distill_lib.core.base_distiller import BaseDistiller
from distill_lib.utils.losses import KD_loss

class SelfDistiller(BaseDistiller):
    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, device='cpu'):
        """
        Self distillation: the same model acts as teacher and student.
        One common approach is to use the model's deeper layers to teach its shallower layers,
        or use a previous snapshot of the model as the teacher.
        Here, we create a frozen copy as the teacher.
        """
        # Create a teacher copy by deep copying and freezing the model.
        import copy
        teacher = copy.deepcopy(model)
        for param in teacher.parameters():
            param.requires_grad = False
        
        super().__init__(model, teacher)
        self.optimizer = optimizer
        self.device = device
        
        self.base_loss_fn = nn.CrossEntropyLoss()
        self.distill_loss_fn = KD_loss

    def update_teacher(self):
        # Optionally, update the teacher periodically (e.g., every epoch)
        import copy
        self.teacher = copy.deepcopy(self.student)
        for param in self.teacher.parameters():
            param.requires_grad = False

    def train_step(self, x, labels, alpha=0.5, temperature=1.0):
        self.student.train()
        x, labels = x.to(self.device), labels.to(self.device)
        
        # Forward pass with teacher (frozen snapshot)
        with torch.no_grad():
            teacher_outputs = self.teacher(x)
        student_outputs = self.student(x)
        
        loss = self.compute_loss(student_outputs, teacher_outputs, labels,
                                 self.base_loss_fn, self.distill_loss_fn,
                                 alpha=alpha, temperature=temperature)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
