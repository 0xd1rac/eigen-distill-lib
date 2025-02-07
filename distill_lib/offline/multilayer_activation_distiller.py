import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .base_offline_distiller import BaseOfflineDistiller
from typing import Callable, Optional, Dict, List
import logging

def default_base_loss_fn(student_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Default base loss (classification) using CrossEntropyLoss."""
    return nn.CrossEntropyLoss()(student_logits, labels)

def default_activation_loss_fn(student_act: torch.Tensor, teacher_act: torch.Tensor) -> torch.Tensor:
    """Default activation loss using Mean Squared Error."""
    return F.mse_loss(student_act, teacher_act)

def create_adapter(teacher_act: torch.Tensor, student_act: torch.Tensor) -> nn.Module:
    """
    Creates a projection adapter to convert teacher_act to the shape of student_act.
    Supports 4D activations (N,C,H,W) using a 1x1 conv or 2D activations using a Linear layer.
    """
    if teacher_act.dim() == 4 and student_act.dim() == 4:
        in_channels = teacher_act.size(1)
        out_channels = student_act.size(1)
        adapter = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    elif teacher_act.dim() == 2 and student_act.dim() == 2:
        in_features = teacher_act.size(1)
        out_features = student_act.size(1)
        adapter = nn.Linear(in_features, out_features)
    else:
        raise ValueError(f"Unsupported activation dimensions: teacher_act.dim()={teacher_act.dim()}, student_act.dim()={student_act.dim()}")
    return adapter

class MultiLayerActivationDistiller(BaseOfflineDistiller):
    def __init__(self, 
                 student: nn.Module,
                 teacher: nn.Module,
                 optimizer: Optional[optim.Optimizer] = None,
                 learning_rate: float = 0.001,
                 alignment_map: Optional[Dict[str, str]] = None,
                 student_layers: Optional[List[str]] = None,
                 teacher_layers: Optional[List[str]] = None) -> None:
        """
        Initializes the MultiLayerActivationDistiller.

        Args:
            student (nn.Module): The student model. Its forward should return (logits, activations)
                                 where activations is a dict of layer outputs.
            teacher (nn.Module): The teacher model (same format as student).
            optimizer (optim.Optimizer, optional): Optimizer for updating the student.
            learning_rate (float): Learning rate for the optimizer if not provided.
            alignment_map (dict, optional): A dictionary mapping student activation keys to teacher activation keys.
                                            If provided, these pairs will be used for computing activation loss.
                                            Otherwise, the distiller will automatically align layers by sorted key order,
                                            matching the first n teacher layers with the n student layers.
            student_layers (list, optional): List of student layer names to register hooks.
            teacher_layers (list, optional): List of teacher layer names to register hooks.
        """
        super().__init__(student, teacher)
        if optimizer is None:
            self.optimizer = optim.Adam(self.student.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer

        self.alignment_map = alignment_map
        self.auto_adapters = {}
        self.student_activations = {}
        self.teacher_activations = {}
        self.student_hooks = []
        self.teacher_hooks = []

        # Default loss functions.
        self.base_loss_fn = default_base_loss_fn
        self.activation_loss_fn = default_activation_loss_fn

        # Register hooks
        if student_layers and teacher_layers:
            self._register_hooks(student, student_layers, self.student_activations)
            self._register_hooks(teacher, teacher_layers, self.teacher_activations)

    def _register_hooks(self, model: nn.Module, layers: List[str], activations: Dict[str, torch.Tensor]):
        def create_hook_fn(layer_name):
            def hook_fn(module, input, output):
                activations[layer_name] = output
            return hook_fn

        for layer_name in layers:
            layer = dict([*model.named_modules()])[layer_name]
            hook = layer.register_forward_hook(create_hook_fn(layer_name))
            if model == self.student:
                self.student_hooks.append(hook)
            else:
                self.teacher_hooks.append(hook)

    def _remove_hooks(self):
        for hook in self.student_hooks + self.teacher_hooks:
            hook.remove()

    def _get_aligned_keys(self) -> list:
        if self.alignment_map is not None:
            aligned = [(s_key, t_key) for s_key, t_key in self.alignment_map.items() if s_key in self.student_activations and t_key in self.teacher_activations]
            return aligned
        else:
            student_keys = sorted(self.student_activations.keys())
            teacher_keys = sorted(self.teacher_activations.keys())
            n = min(len(student_keys), len(teacher_keys))
            return list(zip(student_keys[:n], teacher_keys[:n]))

    def _get_adapter_for_key(self, s_key: str, teacher_act: torch.Tensor, student_act: torch.Tensor, device: str) -> nn.Module:
        if s_key in self.auto_adapters:
            return self.auto_adapters[s_key]
        else:
            if teacher_act.shape != student_act.shape:
                adapter = create_adapter(teacher_act, student_act)
                adapter.to(device)
                self.auto_adapters[s_key] = adapter
                return adapter
            else:
                self.auto_adapters[s_key] = nn.Identity()
                return self.auto_adapters[s_key]

    def distill_step(self, 
                     x: torch.Tensor, 
                     labels: torch.Tensor, 
                     device: str,
                     base_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None, 
                     activation_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None, 
                     alpha: float = 0.5, 
                     layer_weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        base_loss_fn = base_loss_fn if base_loss_fn is not None else self.base_loss_fn
        activation_loss_fn = activation_loss_fn if activation_loss_fn is not None else self.activation_loss_fn

        self.student.train()
        x, labels = x.to(device), labels.to(device)

        with torch.no_grad():
            teacher_logits = self.teacher(x)
        student_logits = self.student(x)

        base_loss = base_loss_fn(student_logits, labels)

        aligned_keys = self._get_aligned_keys()
        total_act_loss = 0.0
        total_weight = 0.0

        for s_key, t_key in aligned_keys:
            s_act = self.student_activations[s_key]
            t_act = self.teacher_activations[t_key]
            adapter = self._get_adapter_for_key(s_key, t_act, s_act, device)
            t_act_proj = adapter(t_act)
            act_loss = activation_loss_fn(s_act, t_act_proj)
            weight = layer_weights.get(s_key, 1.0) if layer_weights is not None else 1.0
            total_act_loss += weight * act_loss
            total_weight += weight

        aggregated_act_loss = total_act_loss / total_weight if total_weight > 0 else 0.0

        total_loss = (1 - alpha) * base_loss + alpha * aggregated_act_loss
        return total_loss

    def distill(self, 
                train_loader: torch.utils.data.DataLoader, 
                num_epochs: int, 
                device: str = 'cpu', 
                base_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None, 
                activation_loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None, 
                alpha: float = 0.5, 
                layer_weights: Optional[Dict[str, float]] = None) -> float:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        for epoch in range(1, num_epochs + 1):
            running_loss = 0.0
            for batch_idx, (input, labels) in enumerate(train_loader):
                loss = self.distill_step(input, labels, device, base_loss_fn, activation_loss_fn, alpha, layer_weights)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if (batch_idx + 1) % 100 == 0:
                    logger.info(f"Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] Loss: {loss:.4f}")
            
            average_loss = running_loss / len(train_loader)
            logger.info(f"Epoch {epoch} Average Loss: {average_loss:.4f}")
        
        # Remove hooks after training
        self._remove_hooks()

        return average_loss

