# Eigen Knowledge Distillation Python Library

This repository provides an end-to-end implementation of Knowledge Distillation (KD) techniques (offline, online, self) for model compression and optimization. The goal is to democratize ML model inference through distillation.

## How Eigen Can Help You with Knowledge Distillation

### 1️⃣ Deploying AI on Mobile 📱  
**Have a new vision segmentation model but don't want it to drain memory or battery on mobile devices?**  
💡 Distill the model down to a smaller architecture using **Eigen's offline distillation**, keeping accuracy while reducing compute costs.  

### 2️⃣ Making LLMs Cheaper & Faster 🧠⚡  
**Have a powerful LLM but it's too slow and expensive to deploy in production?**  
💡 Use **Eigen's online distillation** to train a smaller student LLM in real-time while retaining knowledge from the original model.  

### 3️⃣ Optimizing Edge AI for IoT & Robotics 🤖🌍  
**Want to run an object detection model on an edge device but can't afford a massive YOLO or Faster R-CNN?**  
💡 Apply **feature-based distillation** with Eigen to compress the model while preserving detection accuracy.  

### 4️⃣ Speeding Up Vision Transformers (ViTs) 🖼️⚡  
**Training a ViT but need efficient inference without losing too much performance?**  
💡 Use **self-distillation** to refine the model's internal representations, reducing redundancy while improving feature extraction.  

### 5️⃣ Accelerating Generative AI 🎨💨  
**Want faster inference for a diffusion model or GAN without sacrificing image quality?**  
💡 Use **contrastive distillation** in Eigen to train a lightweight generative model that runs faster while keeping high visual fidelity.  

## Installation
To install the library, you can clone this repository and install the dependencies using pip:
```bash
git clone https://github.com/0xd1rac/eigen-distill-lib.git
cd eigen-distill-lib
pip install -r requirements.txt
```

## Usage 

### 1. Basic Offline Distillation
The simplest form of distillation using a single teacher and student:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from distill_lib.offline.soft_target_distiller import SoftTargetDistiller
from distill_lib.items.student import Student
from distill_lib.items.teacher import Teacher
from examples.utils import get_dataloaders, evaluate

# Set up the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data
teacher_train_loader, student_train_loader, teacher_test_loader, student_test_loader = get_dataloaders()

# Initialize teacher model
teacher_model = models.resnet50(pretrained=True)
teacher_model.fc = nn.Linear(teacher_model.fc.in_features, 10)
teacher = Teacher(model=teacher_model, data_loader=teacher_train_loader)

# Initialize student model
student_model = models.resnet18(pretrained=False)
student_model.fc = nn.Linear(student_model.fc.in_features, 10)
optimizer = optim.SGD(student_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
student = Student(model=student_model, optimizer=optimizer, data_loader=student_train_loader)

# Create distiller
distiller = SoftTargetDistiller(students=[student], teachers=[teacher])

# Distillation parameters
num_epochs = 1
alpha = 0.5  # Weight for distillation loss
temperature = 2.0  # Temperature for softening outputs

# Perform distillation
losses = distiller.distill(num_epochs, device, alpha=alpha, temperature=temperature)

# Evaluate the student model
test_acc = evaluate(student.model, student_test_loader, device)
print(f"Test Accuracy: {test_acc:.2f}%")
```

### 2. Multi-Teacher Distillation with Different Weighting Strategies

Eigen supports multiple strategies for weighting teacher contributions when using multiple teachers:

#### a) Uniform Weighting Strategy
Use this when you want equal contributions from all teachers:

```python
from distill_lib.strategies.weighting_strategies import UniformWeightingStrategy

# Initialize strategy
weighting_strategy = UniformWeightingStrategy()

# Create distiller with uniform weighting
distiller = SoftTargetDistiller(
    students=[student],
    teachers=teacher_models,
    weighting_strategy=weighting_strategy
)
```

#### b) Confidence-Based Weighting Strategy
Use this when you want to weight teachers based on their prediction confidence:

```python
from distill_lib.strategies.weighting_strategies import ConfidenceBasedWeightingStrategy

# Initialize strategy
weighting_strategy = ConfidenceBasedWeightingStrategy()

# Create distiller with confidence-based weighting
distiller = SoftTargetDistiller(
    students=[student],
    teachers=teacher_models,
    weighting_strategy=weighting_strategy
)
```

#### c) Gating Network Weighting Strategy
Use this when you want to learn dynamic weights based on input features:

```python
from distill_lib.strategies.weighting_strategies import GatingNetworkWeightingStrategy

# Get feature size from the teacher's penultimate layer
feature_size = teacher_models[0].model.fc.in_features

# Initialize strategy
weighting_strategy = GatingNetworkWeightingStrategy(
    feature_size=feature_size,
    num_teachers=len(teacher_models)
)

# Create optimizer for the gating network
gating_optimizer = optim.Adam(weighting_strategy.gating_network.parameters(), lr=0.001)

# Create distiller with gating network
distiller = SoftTargetDistiller(
    students=[student],
    teachers=teacher_models,
    weighting_strategy=weighting_strategy
)
```

#### d) Sample-Specific Weighting Strategy
Use this when you want to weight teachers based on their historical performance:

```python
from distill_lib.strategies.weighting_strategies import SampleSpecificWeightingStrategy

# Initialize strategy with memory size
weighting_strategy = SampleSpecificWeightingStrategy(memory_size=1000)

# Create distiller
distiller = SoftTargetDistiller(
    students=[student],
    teachers=teacher_models,
    weighting_strategy=weighting_strategy
)

# During training, update performance history
def update_teacher_history(inputs, labels):
    teacher_performances = []
    for teacher in teacher_models:
        outputs = teacher.model(inputs)
        performance = compute_teacher_performance(outputs, labels)
        teacher_performances.append(performance)
    weighting_strategy.update_history(inputs, teacher_performances)
```

### 3. Complete Examples

For complete working examples of each strategy, refer to the example files:
- `examples/soft_target_distillation_example.py`: Basic distillation
- `examples/confidence_based_distillation_example.py`: Confidence-based weighting
- `examples/gating_network_distillation_example.py`: Gating network weighting
- `examples/sample_specific_distillation_example.py`: Sample-specific weighting

To run an example:
```bash
python examples/confidence_based_distillation_example.py
```

## Features and Status
- [x] Offline: Soft Target Distiller
  - [x] Single teacher, Single student pipeline
  - [x] Single teacher, Many students pipeline
  - [x] Many teachers, Single student pipeline
    - [x] Uniform weighting
    - [x] Confidence-based weighting
    - [x] Gating network weighting
    - [x] Sample-specific weighting
  - [x] Many teachers, Many students pipeline

## Contributing
We welcome contributions! Please feel free to submit a Pull Request.

## Distillation with Eigen
Eigen provides a comprehensive suite of distillation techniques to optimize and compress machine learning models. Here's an overview of the available methods:

### Offline Distillation

Offline distillation involves training a student model using a pre-trained teacher model. This process is typically done in a batch setting, where the teacher's knowledge is transferred to the student through various strategies.

![Offline Distillation](resources/img/offline_distillation.png)


### Online Distillation

Online distillation allows the student model to learn from the teacher model in real-time during training. This approach is beneficial for scenarios where the teacher model is continuously updated or when training data is streamed.

![Online Distillation](resources/img/online_distillation.png)




### Self-Distillation

Self-distillation involves a single model learning from its own predictions or internal representations. This technique can improve model performance by refining its feature extraction capabilities.

![Self Distillation](resources/img/self_distillation.png)




## Usage 
### 1. Offline Distillation: Soft Target Strategy

Here's an example of how to use the `SoftTargetDistiller` for offline distillation:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from distill_lib.offline.soft_target_distiller import SoftTargetDistiller
from examples.utils import get_dataloaders, evaluate

# Set up the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data
train_loader, test_loader = get_dataloaders()

# Initialize teacher and student models
teacher = models.resnet50(pretrained=True)
teacher.fc = nn.Linear(teacher.fc.in_features, 10)
teacher = teacher.to(device)

student = models.resnet18(pretrained=False)
student.fc = nn.Linear(student.fc.in_features, 10)
student = student.to(device)

# Set up optimizer and distiller
optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
distiller = SoftTargetDistiller(student, teacher, optimizer=optimizer)

# Distillation parameters
num_epochs = 1
alpha = 0.5  # Weight for distillation loss
temperature = 2.0  # Temperature for softening outputs

# Perform distillation
distiller.distill(train_loader, num_epochs, device, alpha=alpha, temperature=temperature)

# Evaluate the student model
test_acc = evaluate(student, test_loader, device)
print(f"Test Accuracy: {test_acc:.2f}%")
```

This example demonstrates how to set up and run a soft target distillation process using a larger ResNet50 model as the teacher and a smaller ResNet18 model as the student. The `SoftTargetDistiller` uses the teacher's softened outputs to guide the student's learning process.
