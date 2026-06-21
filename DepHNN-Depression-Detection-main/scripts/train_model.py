import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --- Import our model builder ---
# We assume 'model.py' is in the same 'scripts/' folder
from model import build_dephnn_model, N_CHANNELS, N_TIMESTEPS

# --- Configuration ---
DATA_PATH = 'data/preprocessed/'
MODEL_SAVE_PATH = 'models/'
HISTORY_SAVE_PATH = 'models/'

# Create directories if they don't exist
if not os.path.exists(MODEL_SAVE_PATH):
    os.makedirs(MODEL_SAVE_PATH)

# --- 1. Load Data ---
print("Loading preprocessed data...")
try:
    X = np.load(os.path.join(DATA_PATH, 'X_data.npy'))
    y = np.load(os.path.join(DATA_PATH, 'y_labels.npy'))
except FileNotFoundError:
    print("\n--- ERROR ---")
    print(f"Could not find X_data.npy or y_labels.npy in '{DATA_PATH}'")
    print("Please run 'python scripts/preprocess_data.py' first.")
    exit()

print(f"Data (X) loaded with shape: {X.shape}")
print(f"Labels (y) loaded with shape: {y.shape}")

# --- 2. Format Labels ---
# Our model's output layer is 'softmax' with 2 units.
# It requires labels to be "one-hot encoded".
# e.g., 0 -> [1, 0] (Healthy)
#       1 -> [0, 1] (Depressed)
y_categorical = to_categorical(y, num_classes=2)
print(f"Labels (y) one-hot encoded to shape: {y_categorical.shape}")

# --- 3. Split Data (70% Train, 20% Val, 10% Test) ---
# This is a two-step split.
# First, split into 70% Train and 30% (Val + Test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_categorical, test_size=0.30, random_state=42, stratify=y_categorical
)

# Second, split the 30% temp data into 2/3 (20%) and 1/3 (10%)
# This gives 0.30 * 0.666 = 0.20 (20%) for Validation
# and 0.30 * 0.333 = 0.10 (10%) for Test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.333, random_state=42, stratify=y_temp
)

print("\nData splitting complete:")
print(f"  - Training samples:   {len(X_train)}")
print(f"  - Validation samples: {len(X_val)}")
print(f"  - Test samples:       {len(X_test)}")

# --- 4. Build and Compile Model ---
# Build the model we defined in 'model.py'
model = build_dephnn_model(input_shape=(N_CHANNELS, N_TIMESTEPS))
model.summary()

# Compile the model as specified in the paper
# Optimizer: ADAM
# Loss: Huber's Loss
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_function = tf.keras.losses.Huber()

# --- THIS BLOCK IS NOW CORRECTED ---
model.compile(
    loss=loss_function,
    optimizer=optimizer,
    metrics=['accuracy', 'mae'] # 'mae' is Mean Absolute Error
)
# --- END OF CORRECTION ---

print("\nModel compiled successfully.")

# --- 5. Define Callbacks ---
# These help us manage the training process.

# ModelCheckpoint: Save the *best* model (lowest val_loss)
# We use the .keras format, which is the modern standard
checkpoint_path = os.path.join(MODEL_SAVE_PATH, 'DepHNN_model_best.keras')
model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# EarlyStopping: Stop training if the model stops improving
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15, # Stop after 15 epochs of no improvement
    restore_best_weights=True,
    verbose=1
)

# --- 6. Train the Model ---
print("\nStarting model training...")

EPOCHS = 100
BATCH_SIZE = 64 # From the paper

history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=[model_checkpoint, early_stopping]
)

print("Model training complete.")

# --- 7. Save Final Model and History ---
# Save the final (best) trained model
final_model_path = os.path.join(MODEL_SAVE_PATH, 'DepHNN_model_final.keras')
model.save(final_model_path)
print(f"Final trained model saved to: {final_model_path}")

# Save the training history as a .npy file
history_path = os.path.join(HISTORY_SAVE_PATH, 'training_history.npy')
np.save(history_path, history.history)
print(f"Training history saved to: {history_path}")

# --- 8. Save the Test Set ---
# We save the 10% test set to be used *only* in Phase 5
print("Saving 10% test set for evaluation...")
np.save(os.path.join(DATA_PATH, 'X_test.npy'), X_test)
np.save(os.path.join(DATA_PATH, 'y_test.npy'), y_test)
print("Test set saved.")

print("\nPhase 4: Model Training is complete.")