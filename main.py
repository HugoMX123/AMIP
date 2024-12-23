import tensorflow as tf
from config import *
from data import *
from model import unet_model



train_dataset, val_dataset, test_dataset = load_datasets()


model = unet_model()

# Save model after each epoch
checkpoint_path = "checkpoints/epoch_{epoch:02d}.h5"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False, save_best_only=False)

history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
