# CIFAR-10 Object Recognition using ResNet-50

This project trains a deep learning model on the CIFAR-10 dataset using a
ResNet-50 transfer-learning architecture. The notebook was converted into a
production-ready Python script (`app.py`) that loads the CIFAR-10 images,
preprocesses them, trains a model, evaluates accuracy, and saves the final
trained model.

---

## Features
- CIFAR-10 image loading and preprocessing
- Label encoding
- Train/test split
- Image scaling and resizing for ResNet-50 input
- Transfer learning on ResNet-50 (ImageNet weights)
- Dense classifier head with batch normalization + dropout
- Training history plots (loss + accuracy)
- Model evaluation on test data
- Model export (`.h5`)

---

---

## Requirements
Install dependencies:
```bash
pip install tensorflow pandas numpy pillow scikit-learn matplotlib
```

---

---

## Model Architecture

- Pretrained **ResNet-50** as base (ImageNet weights)
- Custom classification head:
- Flatten  
- BatchNorm  
- Dense → 128 (ReLU)  
- Dropout  
- BatchNorm  
- Dense → 64 (ReLU)  
- Dropout  
- BatchNorm  
- Dense → 10 (softmax)

---

## Author
Piyush Sharma  
GitHub: https://github.com/sharma-piyush1


