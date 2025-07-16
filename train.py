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
    Run a single training epoch through the whole
    batch.
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
    Run a single evaluation epoch through the whole
    batch.
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
) -> Tuple[List[float], List[float], float]:
    
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


def train_loop_classifier(model, train_loader, criterion, optimizer, device):
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


def eval_loop_classifier(model, val_loader, criterion, device):
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
    train_loader,
    valid_loader,
    test_loader,
    optimizer_class,
    criterion,
    device,
    model,
    learning_rate: float = 1e-3,
    epochs: int = 20,
):
    """
    Entrena un clasificador CNN con un encoder preentrenado
    y registra las métricas de loss y accuracy para train/valid.

    Retorna:
        train_losses, valid_losses, train_accuracies, valid_accuracies, test_loss, test_accuracy, modelo_final
    """
    optimizer = optimizer_class(model.parameters(), lr=learning_rate)

    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []

    for epoch in range(epochs):
        # Validation antes del entrenamiento
        valid_loss, valid_acc = eval_loop_classifier(
            model,
            valid_loader,
            criterion,
            device
        )

        # Entrenamiento
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

    # Evaluación final en test set
    test_loss, test_acc = eval_loop_classifier(
        model,
        test_loader,
        criterion,
        device
    )

    return train_losses, valid_losses, train_accuracies, valid_accuracies, test_loss, test_acc, model
