from config import *
from data import get_dataloaders
from model import get_model
from train import train_model

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(IMAGES_DIR, MASKS_DIR, BATCH_SIZE, IMAGE_SIZE)
    model = get_model(NUM_CLASSES)
    train_model(model, train_loader,val_loader, NUM_EPOCHS, LEARNING_RATE, DEVICE)





