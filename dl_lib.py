
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim



def train(model, criterion, optimizer, data, device):
    model.train()
    train_loss = 0
    for batch in data:
        X, y = batch
        X = X.to(device).float()
        y = y.to(device)
        y = y.reshape(-1)
        predictions = model(X)
        loss = criterion(predictions, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return (train_loss / len(data)), model


def validate(model, criterion, data, device):
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for batch in data:
            X, y = batch
            X = X.to(device).float()
            y = y.to(device)

            y = y.reshape(-1)

            predictions = model(X)
            loss = criterion(predictions, y)
            validation_loss += loss.item()

    return validation_loss / len(data)


def train_model(
    model,
    criterion,
    optimizer,
    train_data,
    validation_data,
    epochs=10,
    patience=5,
    device="cpu",
):
    model = model.to(device)
    criterion = criterion.to(device)
    best_loss = float("inf")
    epoch_no_improvement = 0
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        train_loss, model = train(model, criterion, optimizer, train_data, device)
        model = model.to(device)
        validation_loss = validate(model, criterion, validation_data, device)
        print(
            f"Epoch: {epoch+1}\nTraining Loss: {train_loss:.3f}\nValidation Loss: {validation_loss:.3f}"
        )

        # Save Losses:
        train_losses.append(train_loss)
        val_losses.append(validation_loss)
        # Early Stop
        if validation_loss < best_loss:
            best_loss = validation_loss
            epoch_no_improvement = 0
        else:
            epoch_no_improvement += 1

        if epoch_no_improvement > patience:
            break
    return model, (train_losses, val_losses)


if __name__ == "__main__":
    pass