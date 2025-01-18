from config import *
import tensorflow as tf
from tensorflow.keras import layers, models


def get_model():
    if MODEL_NAME == "U-Net":
        return unet_model()
    elif MODEL_NAME == "U-Net smaller":
        return unet_model_smaller()
    elif MODEL_NAME == "U-Net bigger":
        return unet_model_bigger()

# U-Net Model
def unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=NUM_CLASSES):
    inputs = layers.Input(input_size)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

    u1 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c4)
    u1 = layers.concatenate([u1, c3])
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u1)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u2 = layers.concatenate([u2, c2])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u2)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u3 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u3 = layers.concatenate([u3, c1])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u3)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c7)
    model = models.Model(inputs, outputs)
    
    return model

def unet_model_smaller(input_size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=NUM_CLASSES):
    inputs = layers.Input(input_size)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)

    u1 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c3)
    u1 = layers.concatenate([u1, c2])
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)

    u2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c4)
    u2 = layers.concatenate([u2, c1])
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c5)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(c5)

    model = models.Model(inputs, outputs)
    return model

def unet_model_bigger(input_size=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=NUM_CLASSES):
    inputs = layers.Input(input_size)

    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    p5 = layers.MaxPooling2D((2, 2))(c5)

    c6 = layers.Conv2D(2048, (3, 3), activation='relu', padding='same')(p5)
    c6 = layers.Conv2D(2048, (3, 3), activation='relu', padding='same')(c6)

    u1 = layers.Conv2DTranspose(1024, (2, 2), strides=(2, 2), padding='same')(c6)
    u1 = layers.concatenate([u1, c5])
    d1 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(u1)
    d1 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(d1)

    u2 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(d1)
    u2 = layers.concatenate([u2, c4])
    d2 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u2)
    d2 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(d2)

    u3 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(d2)
    u3 = layers.concatenate([u3, c3])
    d3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u3)
    d3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(d3)

    u4 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(d3)
    u4 = layers.concatenate([u4, c2])
    d4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u4)
    d4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(d4)

    u5 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(d4)
    u5 = layers.concatenate([u5, c1])
    d5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u5)
    d5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(d5)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(d5)

    model = models.Model(inputs, outputs)
    return model
