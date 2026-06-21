import os
import glob
import numpy as np
import mne
from sklearn.preprocessing import StandardScaler

# --- Configuration ---

# --- Use your direct path ---
DATA_PATH = r"C:\Users\chara\Desktop\DepHNN_EEG_Project\Data\Raw_Data\4244171"
# ---

SAVE_PATH = 'data/preprocessed/'

# Check if save directory exists, if not, create it
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Add a check to make sure the path is valid
if not os.path.exists(DATA_PATH):
    print(f"Error: The hard-coded path does not exist:")
    print(f"{DATA_PATH}")
    print("Please double-check the path is correct.")
    exit() # Stop the script if the path is wrong

# --- Dataset-specific parameters ---
SAMPLING_FREQ = 256  # 256 Hz

# --- FINALIZED 19-CHANNEL LIST ---
# This is the list of *standard* names we want to end up with.
FINAL_CHANNEL_NAMES_19 = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz'
]

# --- CHANNEL RENAME MAP ---
# This map converts the file's names to our standard names.
# (e.g., 'EEG Fp1-LE' -> 'Fp1', 'EEG T3-LE' -> 'T7')
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
    # 'EEG A2-A1' is a reference, we will drop it
}

# Preprocessing parameters
FILTER_L_FREQ = 0.5
FILTER_H_FREQ = 45
WINDOW_DURATION = 1.0
N_COMPONENTS_ICA = 19  # We have 19 channels

def preprocess_edf_file(filepath):
    """
    Applies the full preprocessing pipeline to a single .edf file.
    """
    try:
        # 1. Load EDF file
        raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        
        # --- FIX: Rename channels ---
        raw.rename_channels(CHANNEL_RENAME_MAP)
        
        # --- FIX: Pick channels in a compatible way ---
        # Get the channels that exist in the file *after* renaming
        available_channels = raw.info['ch_names']
        
        # Find the intersection of channels we want and channels we have
        channels_to_pick = [ch for ch in FINAL_CHANNEL_NAMES_19 if ch in available_channels]
        
        # Check if we found all channels
        if len(channels_to_pick) != len(FINAL_CHANNEL_NAMES_19):
            missing = [ch for ch in FINAL_CHANNEL_NAMES_19 if ch not in available_channels]
            print(f"  Warning: File {os.path.basename(filepath)} is missing channels: {missing}")
            
        # Pick the channels that are available
        raw.pick(channels_to_pick)
        
        # --- END OF FIX ---

        # Verify sampling rate
        if raw.info['sfreq'] != SAMPLING_FREQ:
            # print(f"  Warning: Resampling {filepath} from {raw.info['sfreq']}Hz to {SAMPLING_FREQ}Hz")
            raw.resample(SAMPLING_FREQ)
            
        # 2. Apply bandpass filter (0.5–45 Hz)
        raw.filter(l_freq=FILTER_L_FREQ, h_freq=FILTER_H_FREQ, fir_design='firwin', verbose=False)
        
        # 3. Apply ICA for artifact removal
        # --- FIX: Set n_components dynamically ---
        n_components = len(raw.info['ch_names'])
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=97, max_iter=800)
        # --- END OF FIX ---
        
        ica.fit(raw)
        
        # Find EOG artifacts (blinks)
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['Fp1', 'Fp2'])
        ica.exclude.extend(eog_indices)
        
        # Find muscle artifacts
        muscle_indices, muscle_scores = ica.find_bads_muscle(raw)
        if muscle_indices:
            ica.exclude.extend(muscle_indices)
        
        # Apply the ICA to remove the bad components
        ica.apply(raw, verbose=False)
        
        # 4. Normalization (Z-score normalization per channel)
        raw_data = raw.get_data()
        scaler = StandardScaler()
        raw_data_normalized = scaler.fit_transform(raw_data.T).T
        
        # Create a new Raw object with the normalized data
        info = mne.create_info(ch_names=raw.info['ch_names'], sfreq=SAMPLING_FREQ, ch_types='eeg')
        raw_normalized = mne.io.RawArray(raw_data_normalized, info)
        
        # 5. Segment into 1-second windows (Epochs)
        epochs = mne.make_fixed_length_epochs(raw_normalized, duration=WINDOW_DURATION, overlap=0.0, preload=True, verbose=False)
        
        # Get the data as a 3D NumPy array (n_epochs, n_channels, n_samples)
        epoch_data = epochs.get_data()
        
        return epoch_data
        
    except Exception as e:
        print(f"  --- Error processing {filepath}: {e} ---")
        return None
    
def main():
    """
    Main function to run the preprocessing pipeline.
    """
    print(f"Loading data from: {DATA_PATH}")
    
    # Find all "Eyes Closed" (EC) files
    healthy_files = glob.glob(os.path.join(DATA_PATH, "H * EC.edf"))
    depressed_files = glob.glob(os.path.join(DATA_PATH, "MDD * EC.edf"))
    
    print(f"Found {len(healthy_files)} healthy (EC) files and {len(depressed_files)} depressed (EC) files.")

    if len(healthy_files) == 0 and len(depressed_files) == 0:
        print("\nError: No 'H * EC.edf' or 'MDD * EC.edf' files found.")
        print("Please ensure your DATA_PATH is correct and the files are named as expected.")
        return

    all_windows = []
    all_labels = []

    # Process healthy files (Label 0)
    print("\nProcessing Healthy (Control) files...")
    for i, f in enumerate(healthy_files):
        print(f"  - Processing {os.path.basename(f)} ({i+1}/{len(healthy_files)})")
        windows = preprocess_edf_file(f)
        if windows is not None:
            all_windows.append(windows)
            all_labels.append(np.zeros(len(windows))) # Label 0 for healthy

    # Process depressed files (Label 1)
    print("\nProcessing Depressed (MDD) files...")
    for i, f in enumerate(depressed_files):
        print(f"  - Processing {os.path.basename(f)} ({i+1}/{len(depressed_files)})")
        windows = preprocess_edf_file(f)
        if windows is not None:
            all_windows.append(windows)
            all_labels.append(np.ones(len(windows))) # Label 1 for depressed

    # Check if any data was actually processed
    if not all_windows:
        print("\nError: No data was successfully processed. Stopping.")
        return

    # Combine all windows and labels into single NumPy arrays
    X_data = np.concatenate(all_windows, axis=0)
    y_labels = np.concatenate(all_labels, axis=0)
    
    # --- CRITICAL: SHUFFLE THE DATA ---
    print("\nShuffling data...")
    indices = np.random.permutation(len(X_data))
    X_data = X_data[indices]
    y_labels = y_labels[indices]
    print("Data shuffling complete.")
    
    # Final check of the data shape
    print(f"\nTotal windows created: {X_data.shape[0]}")
    # Shape should be (num_windows, 19_channels, 256_samples)
    print(f"Data shape (X): {X_data.shape}")
    print(f"Labels shape (y): {y_labels.shape}")
    
    # Show class balance
    unique, counts = np.unique(y_labels, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    print("\nClass Balance:")
    print(f"  - Class 0 (Healthy):   {class_counts.get(0.0, 0)} windows")
    print(f"  - Class 1 (Depressed): {class_counts.get(1.0, 0)} windows")

    # 6. Save preprocessed data in .npy format
    save_file_x = os.path.join(SAVE_PATH, 'X_data.npy')
    save_file_y = os.path.join(SAVE_PATH, 'y_labels.npy')
    
    np.save(save_file_x, X_data)
    np.save(save_file_y, y_labels)
    
    print(f"\nSuccessfully saved preprocessed data to:")
    print(f"  - {save_file_x}")
    print(f"  - {save_file_y}")
    print("\nPhase 2: Data Preprocessing is complete.")

if __name__ == "__main__":
    main()