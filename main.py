import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

import pandas as pd
import torch
import random
import time

from config import *
from data import *
from model import get_model
from evaluation import save_results, get_metric, save_learning_curve, dice_loss


# fix the seeds
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.environ['PYTHONHASHSEED'] = str(SEED)

# split the dataset
train_dataset, val_dataset, test_dataset = load_datasets()

# get metric
chosen_metric, monitor_metric = get_metric()
# select the model
model = get_model()

if LOSS_FUNCTION == "dice":
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=dice_loss, metrics=[chosen_metric])
else:
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=LOSS_FUNCTION, metrics=[chosen_metric])

# Save model after each epoch
checkpoint_folder = SAVING_PATH + "checkpoints/"+ MODEL_NAME + SPECIALIZATION 
if not os.path.exists(checkpoint_folder):
      os.mkdir(checkpoint_folder)

checkpoint_callback = ModelCheckpoint(filepath=checkpoint_folder +"/epoch_{epoch:02d}.h5", 
                                      save_weights_only=False, save_best_only=True, mode='max', monitor = monitor_metric)
start_time = time.time()
history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=[checkpoint_callback], verbose = 1)
traning_time = time.time() - start_time
print(f"Training time: {time.time() - start_time} s")

save_learning_curve(history, monitor_metric)

best_model_path = max(
    [os.path.join(checkpoint_folder, f) for f in os.listdir(checkpoint_folder) if f.endswith(".h5")],
    key=os.path.getctime  # Sort by creation time to find the latest file
)

print(f"Best model saved at: {best_model_path}")

# Load the best model
best_model = tf.keras.models.load_model(best_model_path, custom_objects={METRIC: chosen_metric})

# Evaluate the best model on the test dataset
test_loss, test_acc = best_model.evaluate(test_dataset)

# Convert history.history to a DataFrame
history_df = pd.DataFrame(history.history)

# Save the DataFrame to a CSV file
history_df.to_csv( SAVING_PATH + 'training_history/' + MODEL_NAME + SPECIALIZATION + '.csv', index=False) 

save_results(history, test_acc, traning_time)
