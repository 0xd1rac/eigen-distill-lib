import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from distill_lib.offline.base_offline_distiller import BaseOfflineDistiller

def default_base_loss_fn(student_logits, labels):
    """Default base loss (classification) using CrossEntropyLoss."""
    return nn.CrossEntropyLoss()(student_logits, labels)

def default_activation_loss_fn(student_act, teacher_act):
    """Default activation loss using Mean Squared Error."""
    return F.mse_loss(student_act, teacher_act)

def create_adapter(teacher_act, student_act):
    """
    Creates a projection adapter to convert teacher_act to the shape of student_act.
    Supports 4D activations (N,C,H,W) using a 1x1 conv or 2D activations using a Linear layer.
    """
    if teacher_act.dim() == 4 and student_act.dim() == 4:
        in_channels = teacher_act.size(1)
        out_channels = student_act.size(1)
        # Use a 1x1 convolution to change channel dimensions.
        adapter = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    elif teacher_act.dim() == 2 and student_act.dim() == 2:
        in_features = teacher_act.size(1)
        out_features = student_act.size(1)
        adapter = nn.Linear(in_features, out_features)
    else:
        raise ValueError("Unsupported activation dimensions: teacher_act.dim()={}, student_act.dim()={}"
                         .format(teacher_act.dim(), student_act.dim()))
    return adapter

class MultiLayerActivationDistiller(BaseOfflineDistiller):
    def __init__(self, 
                 student: nn.Module,
                 teacher: nn.Module,
                 optimizer: optim.Optimizer = None,
                 alignment_map: dict = None):
        """
        Initializes the MultiLayerActivationDistiller.

        Args:
            student (nn.Module): The student model. Its forward should return (logits, activations)
                                 where activations is a dict of layer outputs.
            teacher (nn.Module): The teacher model (same format as student).
            optimizer (optim.Optimizer, optional): Optimizer for updating the student.
            alignment_map (dict, optional): A dictionary mapping student activation keys to teacher activation keys.
                                            If provided, these pairs will be used for computing activation loss.
                                            Otherwise, the distiller will automatically align layers by sorted key order,
                                            matching the first n teacher layers with the n student layers.
        """
        super().__init__(student, teacher)
        self.optimizer = optimizer
        self.alignment_map = alignment_map  # e.g., {"student_layer3": "teacher_layer2"}
        # This dictionary will store automatically created adapters.
        self.auto_adapters = {}  # key: student activation key, value: adapter module

        # Freeze teacher parameters.
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False

        # Default loss functions.
        self.base_loss_fn = default_base_loss_fn
        self.activation_loss_fn = default_activation_loss_fn

    def _get_aligned_keys(self, student_acts: dict, teacher_acts: dict):
        """
        Returns a list of tuples (s_key, t_key) representing aligned student and teacher layer keys.
        If an alignment_map is provided, use that; otherwise, align sorted keys.
        If teacher has more keys than student, match the first n.
        """
        if self.alignment_map is not None:
            aligned = []
            for s_key, t_key in self.alignment_map.items():
                if s_key in student_acts and t_key in teacher_acts:
                    aligned.append((s_key, t_key))
            return aligned
        else:
            student_keys = sorted(student_acts.keys())
            teacher_keys = sorted(teacher_acts.keys())
            n = min(len(student_keys), len(teacher_keys))
            return list(zip(student_keys[:n], teacher_keys[:n]))

    def _get_adapter_for_key(self, s_key, teacher_act, student_act, device):
        """
        Retrieves an adapter for the given student key. If an adapter has not been created yet
        and the shapes differ, create one automatically.
        """
        if s_key in self.auto_adapters:
            return self.auto_adapters[s_key]
        else:
            # If shapes differ, create an adapter.
            if teacher_act.shape != student_act.shape:
                adapter = create_adapter(teacher_act, student_act)
                adapter.to(device)
                self.auto_adapters[s_key] = adapter
                return adapter
            else:
                # No adapter is needed; return an identity module.
                self.auto_adapters[s_key] = nn.Identity()
                return self.auto_adapters[s_key]

    def distill_step(self, x, labels, device,
                     base_loss_fn=None, 
                     activation_loss_fn=None, 
                     alpha=0.5, 
                     layer_weights: dict = None):
        """
        Performs one distillation step combining the base loss on logits and the activation loss
        computed over multiple aligned layers.

        Args:
            x (Tensor): Input batch.
            labels (Tensor): Ground truth labels.
            device (str): The device to perform computations on.
            base_loss_fn (callable, optional): Loss function for logits (defaults to self.base_loss_fn).
            activation_loss_fn (callable, optional): Loss function for activations (defaults to self.activation_loss_fn).
            alpha (float): Weighting factor for the activation loss.
                           Total loss = (1 - alpha)*base_loss + alpha*aggregated_activation_loss.
            layer_weights (dict, optional): Dictionary mapping student layer keys to weight factors.
        
        Returns:
            Tensor: The combined loss.
        """
        base_loss_fn = base_loss_fn if base_loss_fn is not None else self.base_loss_fn
        activation_loss_fn = activation_loss_fn if activation_loss_fn is not None else self.activation_loss_fn

        self.student.train()
        x, labels = x.to(device), labels.to(device)

        # Forward passes.
        with torch.no_grad():
            teacher_logits, teacher_acts = self.teacher(x)
        student_logits, student_acts = self.student(x)

        # Compute base loss (classification loss on logits).
        base_loss = base_loss_fn(student_logits, labels)

        # Align activations.
        aligned_keys = self._get_aligned_keys(student_acts, teacher_acts)
        total_act_loss = 0.0
        total_weight = 0.0

        for s_key, t_key in aligned_keys:
            s_act = student_acts[s_key]
            t_act = teacher_acts[t_key]
            # Get (or create) an adapter if teacher and student activations differ.
            adapter = self._get_adapter_for_key(s_key, t_act, s_act, device)
            t_act_proj = adapter(t_act)
            act_loss = activation_loss_fn(s_act, t_act_proj)
            weight = layer_weights.get(s_key, 1.0) if layer_weights is not None else 1.0
            total_act_loss += weight * act_loss
            total_weight += weight

        if total_weight > 0:
            aggregated_act_loss = total_act_loss / total_weight
        else:
            aggregated_act_loss = 0.0

        total_loss = (1 - alpha) * base_loss + alpha * aggregated_act_loss
        return total_loss

    def train_step(self, x, labels, 
                   base_loss_fn=None, 
                   activation_loss_fn=None, 
                   alpha=0.5, 
                   layer_weights: dict = None):
        """
        Performs a full training step: computes the loss, backpropagates, and updates the student.

        Args:
            x (Tensor): Input batch.
            labels (Tensor): Ground truth labels.
            base_loss_fn (callable, optional): Overrides the default base loss function.
            activation_loss_fn (callable, optional): Overrides the default activation loss function.
            alpha (float): Weighting factor for activation loss.
            layer_weights (dict, optional): Optional weights per layer.
        
        Returns:
            float: The scalar loss value.
        """
        device = next(self.student.parameters()).device
        loss = self.distill_step(x, labels, device, base_loss_fn, activation_loss_fn, alpha, layer_weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

