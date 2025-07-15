# Convolutional Autoencoder and Classifier on Fashion-MNIST
This repository implements a deep learning pipeline using PyTorch to train a convolutional autoencoder on the Fashion-MNIST dataset, extract latent representations, and build a classifier on top of the learned encoder.
---
## Overview
The pipeline consists of two main stages:
1. **Autoencoder Pretraining**
    - A convolutional autoencoder is trained to reconstruct Fashion-MNIST images.
    - The encoder compresses the images into a low-dimensional latent space.
    - The decoder reconstructs the input from this representation.
2. **Classification**
    - A classifier is trained using the pre-trained encoder as a feature extractor.
    - The classifier predicts the clothing category among the 10 Fashion-MNIST classes.
    - Supports training with either:
        - The encoder **frozen** (only classifier weights updated), or
        - The encoder **fine-tuned** jointly with the classifier.
---
## Repository Structure
```
.
├── data_utils.py            # Data downloading, wrapping, and DataLoaders
├── datasets.py              # Custom PyTorch Dataset wrappers for autoencoder & classifier
├── models.py                # ConvAutoencoder and Classifier architectures
├── plots.py                 # Loss, accuracy, and confusion matrix plots
├── train.py                 # General training and evaluation loops
├── train_autoencoder.py     # Script to train autoencoder, supports grid search
├── train_classifier.py      # Script to train classifier, with optional encoder freezing
├── requirements.txt         # Python dependencies
├── Weights/                 # Directory to store trained weights
└── Images/                  # Directory to store plots
```
---
## Installation
1. Clone this repository:
```bash
git clone <repository-url>
cd cnn-autoencoder-classifier
```
2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
---
## Usage
### 1. Training the Autoencoder
Train a convolutional autoencoder on Fashion-MNIST:
```bash
python train_autoencoder.py
```
Optional hyperparameter tuning grid search:
```bash
python train_autoencoder.py --tune
```
This saves the encoder weights to `Weights/encoder_weights.pth` and plots training vs validation loss curves in `Images/`.
---
### 2. Training the Classifier
Train a classifier on top of the pre-trained encoder.  
By default, the encoder is **fine-tuned jointly** with the classifier head.
```bash
python train_classifier.py
```
To **freeze the encoder** and train only the classifier head:
```bash
python train_classifier.py --pretrained_enc False
```
This saves the classifier weights to `Weights/classifier_weights.pth` and generates plots for loss, accuracy, and the confusion matrix in `Images/`.
---
## Hyperparameters and CLI Options
### Autoencoder (`train_autoencoder.py`)
| Flag             | Description                             | Default |
|-------------------|----------------------------------------|---------|
| `--optimizer` / `-opt` | Optimizer (ADAM, SGD)             | ADAM    |
| `--learning_rate` / `-lr` | Learning rate                 | 1e-3    |
| `--epochs` / `-ep`      | Number of training epochs       | 20      |
| `--dropout` / `-dp`     | Dropout probability            | 0.2     |
| `--latent_dim` / `-ld`  | Dimension of latent space      | 64      |
| `--tune`                | Run hyperparameter grid search | False   |
---
### Classifier (`train_classifier.py`)
| Flag             | Description                             | Default |
|-------------------|----------------------------------------|---------|
| `--learning_rate` / `-lr` | Learning rate                 | 1e-3    |
| `--epochs` / `-ep`      | Number of training epochs       | 20      |
| `--pretrained_enc` / `-pe` | If `True`, fine-tune encoder; if `False`, freeze encoder | True |
---
## Results and Outputs
- Trained encoder weights are saved in `Weights/encoder_weights.pth`.
- Trained classifier weights are saved in `Weights/classifier_weights.pth`.
- Plots for:
    - Autoencoder loss (`Images/autoencoder_loss_curve.png`)
    - Classifier loss (`Images/classifier_loss.png`)
    - Classifier accuracy (`Images/classifier_accuracy.png`)
    - Confusion matrix (`Images/confusion_matrix.png`)
---
## Notes on Reproducibility
- Random seeds are fixed for `torch` and `numpy`.
- Device (CPU or CUDA) is automatically selected based on availability.