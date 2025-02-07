import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from distill_lib.offline.multilayer_activation_distiller import MultiLayerActivationDistiller

class MockModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(MockModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

class TestMultiLayerActivationDistiller(unittest.TestCase):
    def setUp(self):
        self.student = MockModel(10, 2)
        self.teacher = MockModel(10, 2)
        self.optimizer = optim.SGD(self.student.parameters(), lr=0.01)
        self.student_layers = ['fc1']
        self.teacher_layers = ['fc1']
        self.distiller = MultiLayerActivationDistiller(self.student, self.teacher, self.optimizer,
                                                       student_layers=self.student_layers, teacher_layers=self.teacher_layers)

    def test_distill_step(self):
        x = torch.randn(5, 10)
        labels = torch.randint(0, 2, (5,))
        device = 'cpu'
        loss = self.distiller.distill_step(x, labels, device)
        self.assertIsInstance(loss, torch.Tensor)

    def test_distill(self):
        train_loader = [(torch.randn(5, 10), torch.randint(0, 2, (5,))) for _ in range(10)]
        average_loss = self.distiller.distill(train_loader, num_epochs=1)
        self.assertIsInstance(average_loss, float)

if __name__ == '__main__':
    unittest.main() 