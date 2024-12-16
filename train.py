import torch
from config import DEVICE, NUM_EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH
from model import get_model
from torch.amp.grad_scaler import GradScaler
from torch.amp import autocast

scaler = GradScaler()

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device, save_path=MODEL_SAVE_PATH):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Compute validation loss
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                val_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

        # Save model checkpoint
        if save_path:
            checkpoint_path = f"{save_path.rsplit('.', 1)[0]}_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved at {checkpoint_path}")

    print("Training complete!")
    # Save final model
    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"Final model saved at {save_path}")



