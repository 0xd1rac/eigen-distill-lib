import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from .teacher_weighting import TeacherWeightingStrategy
from ..items.teacher import Teacher

class UniformWeightingStrategy(TeacherWeightingStrategy):
    """Assigns equal weights to all teachers."""
    
    def compute_weights(self, teachers: List[Teacher], x: torch.Tensor, temperature: float) -> List[float]:
        """
        Compute uniform weights for all teachers.
        
        Args:
            teachers: List of Teacher instances
            x: Input tensor (not used in this strategy)
            temperature: Temperature parameter (not used in this strategy)
            
        Returns:
            List of equal weights summing to 1.0
        """
        num_teachers = len(teachers)
        return [1.0 / num_teachers] * num_teachers

class ConfidenceBasedWeightingStrategy(TeacherWeightingStrategy):
    """Weights teachers based on their prediction confidence."""
    
    def compute_weights(self, teachers: List[Teacher], x: torch.Tensor, temperature: float) -> List[float]:
        """
        Compute weights based on the prediction confidence of each teacher.
        Teachers with higher confidence (lower entropy) get higher weights.
        
        Args:
            teachers: List of Teacher instances
            x: Input tensor to compute teacher predictions
            temperature: Temperature parameter for softening the confidence scores
            
        Returns:
            List of weights based on prediction confidence, summing to 1.0
        """
        confidences = []
        device = x.device
        
        with torch.no_grad():
            for teacher in teachers:
                # Move teacher to same device as input
                teacher.model.to(device)
                teacher.model.eval()
                
                # Get teacher predictions
                logits = teacher.model(x)
                probs = F.softmax(logits, dim=1)
                
                # Compute entropy as confidence measure (lower entropy = higher confidence)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
                confidence = 1.0 / (entropy + 1e-10)  # Add small epsilon to avoid division by zero
                confidences.append(confidence.item())
        
        # Convert confidences to weights using softmax with temperature
        confidences = torch.tensor(confidences) / temperature
        weights = F.softmax(confidences, dim=0).tolist()
        
        return weights

class GatingNetworkWeightingStrategy(TeacherWeightingStrategy):
    """Use a learned gating network to compute weights based on teacher features."""
    
    def __init__(self, feature_size: int, num_teachers: int):
        """
        Initialize the gating network.
        
        Args:
            feature_size: Size of the feature vector from each teacher
            num_teachers: Number of teachers to weight
        """
        super().__init__()
        self.gating_network = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_teachers),
            nn.Softmax(dim=1)
        )
    
    def extract_features(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from the penultimate layer of a model.
        
        Args:
            model: The teacher model
            x: Input tensor
            
        Returns:
            Feature tensor from the penultimate layer
        """
        # Forward pass until the penultimate layer
        if isinstance(model, nn.Sequential):
            # For sequential models, get features from second-to-last layer
            for layer in list(model.children())[:-1]:
                x = layer(x)
        else:
            # For other architectures (like ResNet), get features before the final FC layer
            if hasattr(model, 'avgpool'):  # ResNet-like
                x = model.conv1(x)
                x = model.bn1(x)
                x = model.relu(x)
                x = model.maxpool(x)
                x = model.layer1(x)
                x = model.layer2(x)
                x = model.layer3(x)
                x = model.layer4(x)
                x = model.avgpool(x)
            elif hasattr(model, 'features'):  # VGG/DenseNet-like
                x = model.features(x)
                x = F.adaptive_avg_pool2d(x, (1, 1))
        
        return x.view(x.size(0), -1)
    
    def compute_weights(self, teachers: List[Teacher], x: torch.Tensor, temperature: float = 1.0) -> List[float]:
        """
        Compute weights for each teacher using the gating network.
        
        Args:
            teachers: List of Teacher instances
            x: Input tensor
            temperature: Temperature for softening the weights
            
        Returns:
            List of weights for each teacher, summing to 1.0
        """
        device = x.device
        self.gating_network = self.gating_network.to(device)
        
        with torch.no_grad():
            # Extract features from the first teacher to get feature size
            first_teacher = teachers[0]
            first_teacher.model.to(device)
            features = self.extract_features(first_teacher.model, x)
            
            # Compute gating weights using the features
            weights = self.gating_network(features)
            # Average weights across batch
            avg_weights = weights.mean(dim=0)
            return avg_weights.tolist()

class SampleSpecificWeightingStrategy(TeacherWeightingStrategy):
    """Compute weights based on teacher performance on similar samples."""
    
    def __init__(self, memory_size: int = 1000):
        super().__init__()
        self.memory_size = memory_size
        self.sample_history = []
        self.performance_history = []
    
    def compute_weights(self, 
                       teachers: List[Teacher], 
                       x: torch.Tensor,
                       temperature: float = 1.0) -> List[float]:
        # This is a simplified version. In practice, you would:
        # 1. Find similar samples in history
        # 2. Check teacher performance on those samples
        # 3. Weight teachers based on their historical performance
        if not self.performance_history:
            return [1.0 / len(teachers)] * len(teachers)
            
        # For now, return uniform weights
        return [1.0 / len(teachers)] * len(teachers)
    
    def update_history(self, sample: torch.Tensor, teacher_performances: List[float]):
        """Update the history with new samples and performances."""
        self.sample_history.append(sample)
        self.performance_history.append(teacher_performances)
        
        # Keep only recent history
        if len(self.sample_history) > self.memory_size:
            self.sample_history.pop(0)
            self.performance_history.pop(0) 