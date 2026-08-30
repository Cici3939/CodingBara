import os
import time
import wave
import keyboard
import keras
import librosa
import numpy as np
import pyaudio
import torch

from CNN import SpecAugment

# Audio Recording Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

# File Paths
recordings_dir = "/Users/cici/Documents/VS Code/CodingBara/recordings/"
melspects_dir = "/Users/cici/Documents/VS Code/CodingBara/melspects/"

os.makedirs(recordings_dir, exist_ok=True)
os.makedirs(melspects_dir, exist_ok=True)

OUTPUT_FILENAME = "output.wav"
out_path = os.path.join(recordings_dir, OUTPUT_FILENAME)
output_pt_path = os.path.join(melspects_dir, "output.pt")
model_path = "/Users/cici/Documents/VS Code/CodingBara/best_model.keras"

# Emotion Dictionary Map
EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Happy",
    3: "Fear",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}


def record_audio(out_filepath):
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    frames = []
    print("Press SPACE to start recording.")
    keyboard.wait("space")
    print("Recording... Press SPACE again to stop.")
    time.sleep(0.2)

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        except KeyboardInterrupt:
            break
        if keyboard.is_pressed("space"):
            print("Stopping recording...")
            time.sleep(0.2)
            break

    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(out_filepath, "wb")
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b"".join(frames))
    waveFile.close()


def prep_files(input_filepath, output_pt_file, target_duration_s=3, n_mels=64):
    """Standardizes audio duration and converts 1D wave signal into a 2D Mel Spectrogram."""
    # Ensure the parent directory exists, NOT creating output_pt_file as a directory!
    os.makedirs(os.path.dirname(output_pt_file), exist_ok=True)

    # Load audio file (resampled to 22,050 Hz)
    audio, sr = librosa.load(input_filepath, sr=22050)
    target_samples = int(target_duration_s * sr)  # 3s * 22050 = 66150 samples

    if len(audio) > target_samples:
        processed_audio = audio[:target_samples]
    elif len(audio) < target_samples:
        pad_amount = target_samples - len(audio)
        processed_audio = np.pad(audio, (0, pad_amount), mode="constant")
    else:
        processed_audio = audio

    # Convert 1D wave signal to 2D Mel Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=processed_audio, sr=sr, n_mels=n_mels, n_fft=1024, hop_length=512
    )

    # Convert power to decibels
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    data = torch.tensor(log_mel_spectrogram, dtype=torch.float32).numpy()

    # Force time dimension to exactly 130
    if data.shape[1] < 130:
        pad_width = 130 - data.shape[1]
        data = np.pad(data, ((0, 0), (0, pad_width)), mode="constant")
    elif data.shape[1] > 130:
        data = data[:, :130]

    # Reshape (64, 130) -> (64, 130, 1)
    data = np.expand_dims(data, axis=-1)

    # Save output PyTorch tensor file
    torch.save(data, output_pt_file)


# 1. Record Audio
OUTPUT_FILENAME = "output.wav"
out_path = os.path.join(
    "/Users/cici/Documents/VS Code/CodingBara/recordings/", OUTPUT_FILENAME
)
record_audio(out_path)

# 2. Preprocess Recording
output_pt_path = "/Users/cici/Documents/VS Code/CodingBara/melspects/output.pt"
prep_files(out_path, output_pt_path)

# 3. Load Model
model_path = "/Users/cici/Documents/VS Code/CodingBara/best_model.keras"
model = keras.models.load_model(
    model_path, custom_objects={"SpecAugment": SpecAugment}
)

# 4. Perform Inference (FIXED: pass output_pt_path instead of out_path!)
tensor_data = torch.load(output_pt_path, weights_only=False)
batch_data = np.expand_dims(tensor_data, axis=0)  # Shape: (1, 64, 130, 1)

predictions = model.predict(batch_data, verbose=0)
predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class] * 100

# Emotion Label Map
EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Happy",
    3: "Fear",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}

print("\n--- Emotion Prediction Results ---")
print(f"Predicted Emotion : {EMOTION_LABELS.get(predicted_class, 'Unknown')}")
print(f"Confidence        : {confidence:.2f}%")
