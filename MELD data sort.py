import os
import shutil
import pandas as pd

# 1. Define paths (Change these if your files are in different folders)
AUDIO_DIR = "/Users/cici/Downloads/MELD.Raw"  # Where your .wav files are
CSV_PATH = "/Users/cici/Downloads/MELD.Raw/train_sent_emo.csv"  # Can change to dev_sent_emo.csv or test_sent_emo.csv
OUTPUT_DIR = "/Users/cici/Downloads/labeled_audio"  # Where sorted folders will be created

# 2. Load the CSV metadata
if not os.path.exists(CSV_PATH):
    print(f"Error: Could not find {CSV_PATH}. Place this script next to your CSV.")
    exit()

df = pd.read_csv(CSV_PATH)

# 3. Process each row in the dataset
moved_count = 0
missing_count = 0

for idx, row in df.iterrows():
    # MELD naming convention: dia[Dialogue_ID]_utt[Utterance_ID].wav
    filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.wav"
    emotion = row["Emotion"].lower().strip()  # e.g., 'joy', 'anger', 'neutral'
    
    source_file = os.path.join(AUDIO_DIR, filename)
    target_folder = os.path.join(OUTPUT_DIR, emotion)
    target_file = os.path.join(target_folder, filename)
    
    # Check if the audio file actually exists on your drive
    if os.path.exists(source_file):
        # Create the emotion folder if it doesn't exist yet
        os.makedirs(target_folder, exist_ok=True)
        # Copy the file to the new folder
        shutil.copy(source_file, target_file)
        moved_count += 1
    else:
        missing_count += 1

print(f"Sorting complete!")
print(f"Successfully copied: {moved_count} files into '{OUTPUT_DIR}'")
if missing_count > 0:
    print(f"Skipped {missing_count} clips because their audio files weren't found.")
