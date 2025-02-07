# Eigen Knowledge Distillation Python Library
This repository is meant to to provide an end-to-end implementation of Knowledge Distillation (KD) techniques (offline, online, self) for model compression and optimization. The goal is to democratise ML model inference through distillation. 

## How Eigen Can Help You with Knowledge Distillation  

### 1️⃣ Deploying AI on Mobile 📱  
**Have a new vision segmentation model but don’t want it to drain memory or battery on mobile devices?**  
💡 Distill the model down to a smaller architecture using **Eigen’s offline distillation**, keeping accuracy while reducing compute costs.  

### 2️⃣ Making LLMs Cheaper & Faster 🧠⚡  
**Have a powerful LLM but it’s too slow and expensive to deploy in production?**  
💡 Use **Eigen’s online distillation** to train a smaller student LLM in real-time while retaining knowledge from the original model.  

### 3️⃣ Optimizing Edge AI for IoT & Robotics 🤖🌍  
**Want to run an object detection model on an edge device but can’t afford a massive YOLO or Faster R-CNN?**  
💡 Apply **feature-based distillation** with Eigen to compress the model while preserving detection accuracy.  

### 4️⃣ Speeding Up Vision Transformers (ViTs) 🖼️⚡  
**Training a ViT but need efficient inference without losing too much performance?**  
💡 Use **self-distillation** to refine the model’s internal representations, reducing redundancy while improving feature extraction.  

### 5️⃣ Accelerating Generative AI 🎨💨  
**Want faster inference for a diffusion model or GAN without sacrificing image quality?**  
💡 Use **contrastive distillation** in Eigen to train a lightweight generative model that runs faster while keeping high visual fidelity.  

## Features 
- [ ] Offline Distillation Pipeline
  - [ ] Probability Output - Student learns the teacher's soft logits 
  - [ ] Intermediate Feature Maps - Student mimics the teachers's activations
  - [ ] Feature Map Relations - Student captures the relationships between teacher's feature maps
  - [ ] Attention Transfer
  - [ ] ViT to CNN Transfer 
  - [ ] Multi-Teacher Support
  - [ ] Customizable Loss Functions

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
