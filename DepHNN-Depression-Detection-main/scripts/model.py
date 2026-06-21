import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, 
    LSTM, 
    Dense, 
    Permute,
    Input,
    Dropout  # <-- Import the Dropout layer
)

# --- Model Parameters ---
N_CHANNELS = 19
N_TIMESTEPS = 256
N_CLASSES = 2

def build_dephnn_model(input_shape=(N_CHANNELS, N_TIMESTEPS)):
    """
    Builds the DepHNN model based on the research paper,
    now including Dropout layers for regularization to prevent overfitting.
    """
    
    print(f"Building model with input shape: {input_shape} and Dropout")
    
    model = Sequential(name="DepHNN_Model_with_Dropout")
    
    # --- Input Layer ---
    model.add(Input(shape=input_shape))
    
    # --- Layer 1: Convolutional Layer ---
    model.add(Conv1D(
        filters=64, 
        kernel_size=5, 
        strides=1, 
        activation='relu', 
        data_format='channels_first',
        name='Conv1D_Layer_1'
    ))
    
    # --- Reshaping for LSTM ---
    model.add(Permute((2, 1), name='Permute_Layer'))
    
    # --- Layer 2: LSTM Layer ---
    model.add(LSTM(
        units=64, 
        return_sequences=True,
        name='LSTM_Layer_2'
    ))
    
    # --- Layer 3: LSTM Layer ---
    model.add(LSTM(
        units=32, 
        return_sequences=False,
        name='LSTM_Layer_3'
    ))

    # --- NEW: Dropout Layer ---
    # Adds regularization after the LSTM block.
    # It will randomly "drop" 50% of the connections.
    model.add(Dropout(0.5, name='Dropout_1'))

    # --- Layer 4: Fully Connected (Dense) Layer ---
    model.add(Dense(
        units=16, 
        activation='relu', 
        name='Dense_Layer_4'
    ))
    
    # --- NEW: Dropout Layer ---
    model.add(Dropout(0.5, name='Dropout_2'))

    # --- Layer 5: Fully Connected (Dense) Layer ---
    model.add(Dense(
        units=16, 
        activation='relu', 
        name='Dense_Layer_5'
    ))
    
    # --- Layer 6: Output Layer ---
    model.add(Dense(
        units=N_CLASSES, 
        activation='softmax', 
        name='Output_Layer_6'
    ))

    return model

if __name__ == "__main__":
    model = build_dephnn_model()
    
    print("\n--- Model Summary (with Dropout) ---")
    model.summary()