# import libraries
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

import os

# Define the path to your Google Drive directory
ser_folder = "/Users/cici/Documents/VS Code/CodingBara/SER/"

# Create the directory if it doesn't exist
if not os.path.exists(ser_folder):
    os.makedirs(ser_folder)


# Define the base directory for your dataset in Google Drive
base_dir = "/Users/cici/Documents/VS Code/CodingBara/SER/"  # Adjust the path according to your directory structure

# Define paths to train and test directories
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Verify the paths
print("Train directory:", train_dir)
print("Test directory:", test_dir)

# create the folders

# --train--
folder = "/Users/cici/Documents/VS Code/CodingBara/SER_std/train/angry"
if not os.path.exists(folder):
  os.makedirs(folder)

folder = "/Users/cici/Documents/VS Code/CodingBara/SER_std/train/disgust"
if not os.path.exists(folder):
  os.makedirs(folder)

folder = "/Users/cici/Documents/VS Code/CodingBara/SER_std/train/happy"
if not os.path.exists(folder):
  os.makedirs(folder)

folder = "/Users/cici/Documents/VS Code/CodingBara/SER_std/train/fear"
if not os.path.exists(folder):
  os.makedirs(folder)

folder = "/Users/cici/Documents/VS Code/CodingBara/SER_std/train/neutral"
if not os.path.exists(folder):
  os.makedirs(folder)

folder = "/Users/cici/Documents/VS Code/CodingBara/SER_std/train/sad"
if not os.path.exists(folder):
  os.makedirs(folder)

folder = "/Users/cici/Documents/VS Code/CodingBara/SER_std/train/surprise"
if not os.path.exists(folder):
  os.makedirs(folder)

# --test--

folder = "/Users/cici/Documents/VS Code/CodingBara/SER_std/test/angry"
if not os.path.exists(folder):
  os.makedirs(folder)

folder = "/Users/cici/Documents/VS Code/CodingBara/SER_std/test/disgust"
if not os.path.exists(folder):
  os.makedirs(folder)

folder = "/Users/cici/Documents/VS Code/CodingBara/SER_std/test/happy"
if not os.path.exists(folder):
  os.makedirs(folder)

folder = "/Users/cici/Documents/VS Code/CodingBara/SER_std/test/fear"
if not os.path.exists(folder):
  os.makedirs(folder)

folder = "/Users/cici/Documents/VS Code/CodingBara/SER_std/test/neutral"
if not os.path.exists(folder):
  os.makedirs(folder)

folder = "/Users/cici/Documents/VS Code/CodingBara/SER_std/test/sad"
if not os.path.exists(folder):
  os.makedirs(folder)

folder = "/Users/cici/Documents/VS Code/CodingBara/SER_std/test/surprise"
if not os.path.exists(folder):
  os.makedirs(folder)


""" Prepare the files """

# Unified audio segmentation
import os
from pydub import AudioSegment
import librosa
import librosa.display
import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

print(f"Is Apple MPS available? {torch.backends.mps.is_available()}")

""" standardize audio duration by cutting or padding and convert from 1d audio to 2d mel spectrogram tensor """
def prep_files(input_dir, output_dir, target_duration_s=3, n_mels=64, emo_label="None"):
    """
    Cuts or pads with silence all audio files in a directory to a target duration
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Supported extensions (pydub handles these if ffmpeg is installed)
    valid_extensions = ('.wav', '.m4a')
    i = 1

    for file_name in os.listdir(input_dir):
        if not file_name.lower().endswith(valid_extensions):
            continue

        file_path = os.path.join(input_dir, file_name)
        こんにちわ= emo_label+str(i)+".png"
        output_path = os.path.join(output_dir, こんにちわ)
        i += 1

        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=22050)
            current_duration = len(audio) / sr
            target_duration_s = 3

            if current_duration > target_duration_s:
                # Cut the audio if it's too long
                processed_audio = audio[:target_duration_s]
            elif current_duration < target_duration_s:
                # Pad with silence if it's too short
                processed_audio = np.pad(audio, (0, max(0, int(target_duration_s - len(audio)))), 'constant')
                # silence_needed = target_duration_s - current_duration
                # silence = AudioSegment.silent(duration=silence_needed, frame_rate=sr)
                # processed_audio = audio + silence
            else:
                processed_audio = audio

            # mel filter banks
            filter_banks = librosa.filters.mel(sr=sr, n_fft=1024, n_mels=n_mels)

            # filter_banks.shape 
            plt.figure(figsize=(25, 10))
            librosa.display.specshow(filter_banks, 
                                    sr=sr, 
                                    x_axis="linear")
            plt.colorbar(format="%+2.f")
            plt.show()

            # Convert 1D wave signal to 2D Mel Spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(
                y=processed_audio, sr=sr, n_mels=n_mels, n_fft=1024, hop_length=512
            )            
            
            # Convert power to decibels (log scale matches human hearing)
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

            plt.figure(figsize=(25, 10))
            librosa.display.specshow(log_mel_spectrogram, 
                                    x_axis="time", 
                                    y_axis="mel", 
                                    sr=sr)
            plt.colorbar(format="%+2.f") 
            plt.show()
            

            # Add a Channel dimension (1, n_mels, time_steps) to match CNN expectation
            # PyTorch CNNs expect: (Batch, Channel, Height, Width)
            audio_tensor = torch.tensor(log_mel_spectrogram, dtype=torch.float32).unsqueeze(0)

            torch.save(audio_tensor, output_path)

        except Exception as e:
            print(f"Failed to process {file_name}: {e}")

# --- Configuration ---
TARGET_DURATION = 3

# --train--
INPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER/train/angry"
OUTPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER_std/train/angry"
prep_files(INPUT_DATASET, OUTPUT_DATASET, TARGET_DURATION, "angry")

INPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER/train/disgust"
OUTPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER_std/train/disgust"
prep_files(INPUT_DATASET, OUTPUT_DATASET, TARGET_DURATION, "disgust")

INPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER/train/happy"
OUTPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER_std/train/happy"
prep_files(INPUT_DATASET, OUTPUT_DATASET, TARGET_DURATION, "happy")

INPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER/train/fear"
OUTPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER_std/train/fear"
prep_files(INPUT_DATASET, OUTPUT_DATASET, TARGET_DURATION, "fear")

INPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER/train/neutral"
OUTPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER_std/train/neutral"
prep_files(INPUT_DATASET, OUTPUT_DATASET, TARGET_DURATION, "neutral")

INPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER/train/sad"
OUTPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER_std/train/sad"
prep_files(INPUT_DATASET, OUTPUT_DATASET, TARGET_DURATION, "sad")

INPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER/train/surprise"
OUTPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER_std/train/surprise"
prep_files(INPUT_DATASET, OUTPUT_DATASET, TARGET_DURATION, "surprise")

# --test--

INPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER/test/angry"
OUTPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER_std/test/angry"
prep_files(INPUT_DATASET, OUTPUT_DATASET, TARGET_DURATION, "angry")

INPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER/test/disgust"
OUTPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER_std/test/disgust"
prep_files(INPUT_DATASET, OUTPUT_DATASET, TARGET_DURATION, "disgust")

INPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER/test/happy"
OUTPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER_std/test/happy"
prep_files(INPUT_DATASET, OUTPUT_DATASET, TARGET_DURATION, "happy")

INPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER/test/fear"
OUTPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER_std/test/fear"
prep_files(INPUT_DATASET, OUTPUT_DATASET, TARGET_DURATION, "fear")

INPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER/test/neutral"
OUTPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER_std/test/neutral"
prep_files(INPUT_DATASET, OUTPUT_DATASET, TARGET_DURATION, "neutral")

INPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER/test/sad"
OUTPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER_std/test/sad"
prep_files(INPUT_DATASET, OUTPUT_DATASET, TARGET_DURATION, "sad")

INPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER/test/surprise"
OUTPUT_DATASET = "/Users/cici/Documents/VS Code/CodingBara/SER_std/test/surprise"
prep_files(INPUT_DATASET, OUTPUT_DATASET, TARGET_DURATION, "surprise")