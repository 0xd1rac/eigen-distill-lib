import torch
import torch.nn as nn
import torch.optim as optim
from distill_lib.core.base_distiller import BaseDistiller
from distill_lib.utils.losses import KD_loss

class OnlineDistiller(BaseDistiller):
    def __init__(self, student: nn.Module, teacher: nn.Module, optimizer: optim.Optimizer, device='cpu'):
        """
        Online distillation where teacher and student are trained simultaneously.
        Teacher can be updated as a moving average or an ensemble.
        """
        super().__init__(student, teacher)
        self.optimizer = optimizer
        self.device = device
        
        self.base_loss_fn = nn.CrossEntropyLoss()
        self.distill_loss_fn = KD_loss

    def train_step(self, x, labels, alpha=0.5, temperature=1.0):
        self.student.train()
        self.teacher.train()  # teacher may be trainable too
        x, labels = x.to(self.device), labels.to(self.device)
        
        student_outputs = self.student(x)
        teacher_outputs = self.teacher(x)
        loss = self.compute_loss(student_outputs, teacher_outputs, labels,
                                 self.base_loss_fn, self.distill_loss_fn,
                                 alpha=alpha, temperature=temperature)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Optionally update teacher as a moving average of the student
        self._update_teacher()
        return loss.item()
    
    def _update_teacher(self, momentum=0.99):
        with torch.no_grad():
            for t_param, s_param in zip(self.teacher.parameters(), self.student.parameters()):
                t_param.data.mul_(momentum).add_(s_param.data, alpha=1 - momentum)
