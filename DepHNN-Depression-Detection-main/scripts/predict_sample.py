import numpy as np
import os
import tensorflow as tf
import mne
from sklearn.preprocessing import StandardScaler
import warnings

# --- 1. Configuration & Constants (Must match preprocess_data.py) ---
MODEL_PATH = 'models/DepHNN_model_final.keras'

# --- Preprocessing Constants ---
SAMPLING_FREQ = 256
FINAL_CHANNEL_NAMES_19 = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz'
]
CHANNEL_RENAME_MAP = {
    'EEG Fp1-LE': 'Fp1', 'EEG Fp2-LE': 'Fp2',
    'EEG F3-LE':  'F3',  'EEG F4-LE':  'F4',
    'EEG C3-LE':  'C3',  'EEG C4-LE':  'C4',
    'EEG P3-LE':  'P3',  'EEG P4-LE':  'P4',
    'EEG O1-LE':  'O1',  'EEG O2-LE':  'O2',
    'EEG F7-LE':  'F7',  'EEG F8-LE':  'F8',
    'EEG T3-LE':  'T7',  'EEG T4-LE':  'T8', # T3->T7, T4->T8
    'EEG T5-LE':  'P7',  'EEG T6-LE':  'P8', # T5->P7, T6->P8
    'EEG Fz-LE':  'Fz',  'EEG Cz-LE':  'Cz',
    'EEG Pz-LE':  'Pz'
}
FILTER_L_FREQ = 0.5
FILTER_H_FREQ = 45
WINDOW_DURATION = 1.0


# --- 2. Re-implement Preprocessing Function ---
# A new sample must go through the *exact* same steps as the training data.
def preprocess_single_file(filepath):
    """
    Applies the full preprocessing pipeline to a single new .edf file.
    """
    try:
        # Suppress MNE info messages for a cleaner prediction output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        
        # Rename and pick channels
        raw.rename_channels(CHANNEL_RENAME_MAP)
        available_channels = raw.info['ch_names']
        channels_to_pick = [ch for ch in FINAL_CHANNEL_NAMES_19 if ch in available_channels]
        
        if len(channels_to_pick) < len(FINAL_CHANNEL_NAMES_19):
            print(f"Warning: File is missing {len(FINAL_CHANNEL_NAMES_19) - len(channels_to_pick)} channels.")
        
        raw.pick(channels_to_pick)

        # Resample if necessary
        if raw.info['sfreq'] != SAMPLING_FREQ:
            raw.resample(SAMPLING_FREQ, verbose=False)
            
        # Filter
        raw.filter(l_freq=FILTER_L_FREQ, h_freq=FILTER_H_FREQ, fir_design='firwin', verbose=False)
        
        # ICA
        n_components = len(raw.info['ch_names'])
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=97, max_iter=800)
        ica.fit(raw)
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['Fp1', 'Fp2'])
        ica.exclude.extend(eog_indices)
        muscle_indices, muscle_scores = ica.find_bads_muscle(raw)
        if muscle_indices:
            ica.exclude.extend(muscle_indices)
        ica.apply(raw, verbose=False)
        
        # Normalize
        raw_data = raw.get_data()
        scaler = StandardScaler()
        raw_data_normalized = scaler.fit_transform(raw_data.T).T
        
        # Create new Raw object
        info = mne.create_info(ch_names=raw.info['ch_names'], sfreq=SAMPLING_FREQ, ch_types='eeg')
        raw_normalized = mne.io.RawArray(raw_data_normalized, info, verbose=False)
        
        # Windowing
        epochs = mne.make_fixed_length_epochs(raw_normalized, duration=WINDOW_DURATION, overlap=0.0, preload=True, verbose=False)
        epoch_data = epochs.get_data()
        
        # Ensure data is float32 for the model
        return epoch_data.astype(np.float32)
        
    except Exception as e:
        print(f"  --- Error preprocessing file {filepath}: {e} ---")
        return None

# --- 3. Main Prediction Function ---
def predict_from_sample(filepath, model):
    """
    Loads a raw EEG file, preprocesses it, and returns a final diagnosis.
    """
    print(f"\nProcessing new sample: {os.path.basename(filepath)}")
    
    # Preprocess the file to get windows
    windows = preprocess_single_file(filepath)
    
    if windows is None or len(windows) == 0:
        print("Could not process file or no windows were created.")
        return

    print(f"File segmented into {len(windows)} windows.")
    
    # Get model predictions for *all* windows
    predictions = model.predict(windows)
    
    # Get the predicted class (0 or 1) for each window
    predicted_classes = np.argmax(predictions, axis=1)
    
    # --- Aggregate Results ---
    # We take the "majority vote" of all windows.
    healthy_votes = np.sum(predicted_classes == 0)
    depressed_votes = np.sum(predicted_classes == 1)
    
    # Calculate percentages
    total_votes = len(predicted_classes)
    healthy_percent = (healthy_votes / total_votes) * 100
    depressed_percent = (depressed_votes / total_votes) * 100
    
    print("\n--- Prediction Results ---")
    print(f"Healthy votes:   {healthy_votes} ({healthy_percent:.1f}%)")
    print(f"Depressed votes: {depressed_votes} ({depressed_percent:.1f}%)")
    
    # Final diagnosis
    if depressed_votes > healthy_votes:
        print("\nDiagnosis: 🔴 Depressed")
    else:
        print("\nDiagnosis: 🟢 Healthy (Control)")
    
    return

# --- 4. Run the Script ---
if __name__ == "__main__":
    # --- Load the Model ---
    print("Loading trained model...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except OSError:
        print("\n--- ERROR ---")
        print(f"Could not find model file at '{MODEL_PATH}'")
        print("Please run 'python scripts/train_model.py' first.")
        exit()
    
    # --- !! IMPORTANT: SET THIS PATH !! ---
    # To test, provide a path to *one* raw .edf file.
    # I've used a file from the "Depressed" group as an example.
    
    # **REPLACE THIS** with a path to one of your .edf files:
    # EXAMPLE_FILE_PATH = r"C:\Users\chara\Desktop\DepHNN_EEG_Project\Data\Raw_Data\4244171\MDD S1 EC.edf"
    # EXAMPLE_FILE_PATH = r"C:\Users\chara\Desktop\DepHNN_EEG_Project\Data\Raw_Data\4244171\H S1 EC.edf"
    
    # Please un-comment one of the lines above and set the correct path
    EXAMPLE_FILE_PATH = r"C:\Users\chara\Desktop\DepHNN_EEG_Project\Data\Raw_Data\4244171\H S8 EC.edf" # <-- SET THIS
    
    if not EXAMPLE_FILE_PATH or not os.path.exists(EXAMPLE_FILE_PATH):
        print("\n--- PLEASE SET A VALID FILE PATH ---")
        print("Open 'scripts/predict_sample.py' and set the 'EXAMPLE_FILE_PATH' variable.")
    else:
        predict_from_sample(EXAMPLE_FILE_PATH, model)

    print("\nPhase 6: Deployment & Testing is complete.")