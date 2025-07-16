from typing import List, Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader


def train_encoder_loop(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """
    Run one epoch of training for the convolutional autoencoder.

    Args:
        model (nn.Module): The autoencoder model to be trained.
        train_loader (DataLoader): DataLoader for training dataset.
        criterion (nn.Module): Loss function (e.g., MSELoss).
        optimizer (Optimizer): Optimizer used for training (e.g., Adam).
        device (torch.device): Computation device (CPU or CUDA).

    Returns:
        float: Average training loss over the epoch.
    """
    model.train()
    train_loss = 0.0

    for data, target in train_loader:

        # Get sample and target
        data, target = data.to(device), target.to(device)

        # Compute model prediction and its loss
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = train_loss / len(train_loader)
    return avg_loss


def eval_encoder_loop(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Run one epoch of evaluation for the convolutional autoencoder.

    Args:
        model (nn.Module): The trained autoencoder model.
        test_loader (DataLoader): DataLoader for validation or test set.
        criterion (nn.Module): Loss function (e.g., MSELoss).
        device (torch.device): Computation device (CPU or CUDA).

    Returns:
        float: Average evaluation loss over the epoch.
    """
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            # Get sample and target
            data, target = data.to(device), target.to(device)

            # Compute model prediction and its loss
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

    avg_loss = test_loss / len(test_loader)

    return avg_loss


def autoencoder_training(
    train_loader,
    valid_loader,
    test_loader,
    optimizer_class,
    criterion,
    device,
    model,
    learning_rate: float = 1e-3,
    epochs: int = 20,
) -> Tuple[List[float], List[float], float, nn.Module]:
    """
    Complete training process for the autoencoder over multiple epochs.
    Includes training, validation, and final testing phases.

    Args:
        train_loader (DataLoader): Training data.
        valid_loader (DataLoader): Validation data.
        test_loader (DataLoader): Test data.
        optimizer_class (type): Optimizer class (e.g., torch.optim.Adam).
        criterion (nn.Module): Loss function (e.g., MSELoss).
        device (torch.device): Computation device.
        model (nn.Module): The autoencoder instance.
        learning_rate (float): Learning rate for optimizer.
        epochs (int): Number of training epochs.

    Returns:
        Tuple[List[float], List[float], float, nn.Module]:
            - List of training losses
            - List of validation losses
            - Final test loss
            - Trained model
    """
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    train_losses = []
    valid_losses = []

    for epoch in range(epochs):
        avg_valid_loss = eval_encoder_loop(
            model,
            valid_loader,
            criterion,
            device
        )
        avg_training_loss = train_encoder_loop(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        train_losses.append(avg_training_loss)
        valid_losses.append(avg_valid_loss)

        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(
                f"Epoch {epoch+1}/{epochs} - "
                f"Training Loss: {avg_training_loss:.5f} - "
                f"Valid Loss: {avg_valid_loss:.5f}"
            )

    avg_test_loss = eval_encoder_loop(
        model,
        test_loader,
        criterion,
        device
    )
    return train_losses, valid_losses, avg_test_loss, model


def train_loop_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """
    Train the classifier model for one epoch and compute loss and accuracy.

    Args:
        model (nn.Module): Classifier model with encoder + head.
        train_loader (DataLoader): DataLoader for training dataset.
        criterion (nn.Module): CrossEntropyLoss function.
        optimizer (Optimizer): Optimizer for training.
        device (torch.device): CPU or CUDA.

    Returns:
        Tuple[float, float]:
            - Average training loss
            - Training accuracy
    """
    model.train()
    train_loss, correct, total = 0.0, 0, 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = output.argmax(dim=1)
        correct += (preds == target).sum().item()
        total += target.size(0)
    avg_loss = train_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def eval_loop_classifier(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Evaluate the classifier model for one epoch and compute loss and accuracy.

    Args:
        model (nn.Module): Classifier model.
        val_loader (DataLoader): DataLoader for validation or test dataset.
        criterion (nn.Module): CrossEntropyLoss function.
        device (torch.device): CPU or CUDA.

    Returns:
        Tuple[float, float]:
            - Average evaluation loss
            - Evaluation accuracy
    """
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    avg_loss = val_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def classifier_training(
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    optimizer_class: type[torch.optim.Optimizer],
    criterion: nn.Module,
    device: torch.device,
    model: nn.Module,
    learning_rate: float = 1e-3,
    epochs: int = 20,
) -> Tuple[
    List[float], List[float],
    List[float], List[float],
    float, float, nn.Module
]:
    """
    Full training and evaluation loop for a CNN classifier using a pretrained encoder.

    Args:
        train_loader (DataLoader): Training set.
        valid_loader (DataLoader): Validation set.
        test_loader (DataLoader): Test set.
        optimizer_class (type): Optimizer class (e.g., torch.optim.Adam).
        criterion (nn.Module): CrossEntropyLoss.
        device (torch.device): Computation device.
        model (nn.Module): Classifier model to train.
        learning_rate (float): Learning rate for optimizer.
        epochs (int): Number of training epochs.

    Returns:
        Tuple[List[float], List[float], List[float], List[float], float, float, nn.Module]:
            - Training losses
            - Validation losses
            - Training accuracies
            - Validation accuracies
            - Final test loss
            - Final test accuracy
            - Trained model
    """
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []

    for epoch in range(epochs):
        valid_loss, valid_acc = eval_loop_classifier(
            model,
            valid_loader,
            criterion,
            device
        )
        train_loss, train_acc = train_loop_classifier(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)

        if (epoch + 1) % max(1, epochs // 10) == 0:
            print(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Valid Loss: {valid_loss:.4f}, Acc: {valid_acc:.4f}"
            )

    test_loss, test_acc = eval_loop_classifier(
        model,
        test_loader,
        criterion,
        device
    )

    return train_losses, valid_losses, train_accuracies, valid_accuracies, test_loss, test_acc, model
