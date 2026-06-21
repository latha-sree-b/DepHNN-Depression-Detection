import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# --- Configuration ---
MODEL_PATH = 'models/DepHNN_model_final.keras'
HISTORY_PATH = 'models/training_history.npy'
DATA_PATH = 'data/preprocessed/'
PLOTS_PATH = 'models/' # We'll save the plots in the models folder

# --- 1. Load Trained Model ---
print(f"Loading model from: {MODEL_PATH}")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except OSError:
    print("\n--- ERROR ---")
    print(f"Could not find model file at '{MODEL_PATH}'")
    print("Please run 'python scripts/train_model.py' first.")
    exit()

# --- 2. Load Test Data ---
print("Loading 10% test set...")
try:
    X_test = np.load(os.path.join(DATA_PATH, 'X_test.npy'))
    y_test_categorical = np.load(os.path.join(DATA_PATH, 'y_test.npy'))
except FileNotFoundError:
    print("\n--- ERROR ---")
    print(f"Could not find X_test.npy or y_test.npy in '{DATA_PATH}'")
    print("Please run 'python scripts/train_model.py' first.")
    exit()

# For metrics, we need the original 0 or 1 labels, not the one-hot encoded
# np.argmax converts [1, 0] -> 0 and [0, 1] -> 1
y_test_labels = np.argmax(y_test_categorical, axis=1)

print(f"Test data (X) loaded with shape: {X_test.shape}")
print(f"Test labels (y) loaded with shape: {y_test_labels.shape}")

# --- 3. Evaluate Model on Test Set ---
print("\nEvaluating model on the 10% test set...")
results = model.evaluate(X_test, y_test_categorical, verbose=1)

print("\n--- Test Set Evaluation ---")
print(f"Test Loss:     {results[0]:.4f}")
print(f"Test Accuracy: {results[1] * 100:.2f}%") # Paper's metric
print(f"Test MAE:      {results[2]:.4f}")      # Paper's metric

# --- 4. Generate Classification Report & Confusion Matrix ---
# Get model predictions
y_pred_probs = model.predict(X_test)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

# Classification Report (Precision, Recall, F1-Score)
print("\n--- Classification Report ---")
target_names = ['Class 0 (Healthy)', 'Class 1 (Depressed)']
print(classification_report(y_test_labels, y_pred_labels, target_names=target_names))

# Confusion Matrix
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_test_labels, y_pred_labels)
print(cm)

# Plot and save the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Healthy (0)', 'Depressed (1)'],
            yticklabels=['Healthy (0)', 'Depressed (1)'])
plt.title('Confusion Matrix on Test Data')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
cm_plot_path = os.path.join(PLOTS_PATH, 'confusion_matrix.png')
plt.savefig(cm_plot_path)
print(f"\nConfusion matrix plot saved to: {cm_plot_path}")
plt.close() # Close the plot to save memory

# --- 5. Plot Training & Validation History ---
print("Loading training history...")
try:
    history = np.load(HISTORY_PATH, allow_pickle=True).item()
except FileNotFoundError:
    print(f"Could not find '{HISTORY_PATH}'. Skipping history plots.")
    history = None

if history:
    # Plot Accuracy (like Fig. 9)
    plt.figure(figsize=(10, 5))
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    acc_plot_path = os.path.join(PLOTS_PATH, 'model_accuracy_plot.png')
    plt.savefig(acc_plot_path)
    print(f"Accuracy plot saved to: {acc_plot_path}")
    plt.close()

    # Plot Loss (like Fig. 8)
    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss (Huber)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    loss_plot_path = os.path.join(PLOTS_PATH, 'model_loss_plot.png')
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved to: {loss_plot_path}")
    plt.close()

print("\nPhase 5: Model Evaluation is complete.")