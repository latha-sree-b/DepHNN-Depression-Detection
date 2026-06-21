import mne
import numpy as np
import os

# 1. SETUP: Point to a real Depressed file you already have
# Replace this with the name of any MDD file from your dataset
original_file = r"C:\Users\chara\Desktop\dephnn_django\Data\Raw_Data\4244171\MDD S1  EO.edf" 

# 2. OUTPUT: Name of the new file you want to create
new_filename = "synthetic_test_patient.edf"

def create_synthetic_file():
    print(f"Loading original file: {original_file}...")
    
    # Read the file
    raw = mne.io.read_raw_edf(original_file, preload=True, verbose=False)
    
    # Get the data (Voltage values)
    data = raw.get_data()
    
    # 3. THE TRICK: Add tiny random noise (0.5% variation)
    # This changes the values so it's technically a "new" file, 
    # but keeps the "Depressed" pattern intact.
    noise = np.random.normal(0, 0.05 * np.std(data), data.shape)
    new_data = data + noise
    
    # Create a new Raw object with the modified data
    new_raw = mne.io.RawArray(new_data, raw.info)
    
    # 4. SAVE: Export it as a new .edf file
    print(f"Creating new file: {new_filename}...")
    mne.export.export_raw(new_filename, new_raw, fmt='edf', overwrite=True)
    
    print("Success! You can now upload 'synthetic_test_patient.edf' to your website.")

if __name__ == "__main__":
    # Check if file exists before running
    if os.path.exists(original_file):
        create_synthetic_file()
    else:
        print(f"Error: Could not find {original_file}. Please put a real .edf file in this folder first.")