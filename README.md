# Eigen.AI Knowledge Distillation Python Library
This repository provides an end-to-end implementation of Knowledge Distillation (KD) techniques (offline, online, self) for model compression and optimization. Check out .md for current goals of this prject.

## Features 
- [ ] Offline Distillation Pipeline
  - [ ] Knowledge Source: Probability Output - Student learns the teacher's soft logits 
  - [ ] Knowledge Source: Intermediate Feature Maps - Student mimics the teachers's activations
  - [ ] Knowledge Source: Feature Map Relations - Student captures the relationships between teacher's feature maps
  - [ ] Multi-Teacher Support
  - [ ] Customizable Loss Functions
- [ ] Online Distillation Pipeline
  - [ ] Real-Time Teacher-Student Training - Teacher generates knowledge on the fly
- [ ] Self Distillation Pipeline

## Installation
To install the library, you can clone this repository and install the dependencies using pip:
```bash
git clone https://github.com/0xd1rac/eigen-distill-lib.git
cd eigen-distill-lib
pip install -r requirements.txt
```

## Usage 
### 1. Offline Disillation: Soft Output Strategy 
```python

```
