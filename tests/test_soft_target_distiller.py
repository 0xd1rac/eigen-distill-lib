import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from distill_lib.offline.soft_target_distiller import SoftTargetDistiller

class SimpleModel(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

class TestSoftTargetDistiller(unittest.TestCase):
    def setUp(self) -> None:
        self.student = SimpleModel(10, 2)
        self.teacher = SimpleModel(10, 2)
        self.optimizer = optim.SGD(self.student.parameters(), lr=0.01)
        self.distiller = SoftTargetDistiller(self.student, self.teacher, self.optimizer)

    def test_distill_step(self) -> None:
        x = torch.randn(5, 10)
        labels = torch.randint(0, 2, (5,))
        device = 'cpu'
        loss = self.distiller.distill_step(x, labels, device)
        self.assertIsInstance(loss, torch.Tensor)

    def test_distill(self) -> None:
        data = torch.randn(10, 10)
        labels = torch.randint(0, 2, (10,))
        teacher_loader = DataLoader(TensorDataset(data, labels), batch_size=5)
        student_loader = DataLoader(TensorDataset(data, labels), batch_size=5)
        average_loss = self.distiller.distill(teacher_loader, student_loader, num_epochs=1)
        self.assertIsInstance(average_loss, float)

if __name__ == '__main__':
    unittest.main() 