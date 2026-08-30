import pyaudio
import wave
import keyboard
import os
import time
import librosa
import torch

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
OUTPUT_FILENAME = "output.wav"

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

frames = []
print("Press SPACE to start recording.")
keyboard.wait('space')
print("Recording... Press SPACE again to stop.")
time.sleep(0.2)

while True:
    try:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    except KeyboardInterrupt:
        break
    if keyboard.is_pressed('space'):
        print("Stopping recording after a brief delay...")
        time.sleep(0.2)
        break

stream.stop_stream()
stream.close()
audio.terminate()

waveFile = wave.open('/Users/cici/Documents/VS Code/CodingBara/recordings' + OUTPUT_FILENAME + '/', 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

input_dir = '/Users/cici/Documents/VS Code/CodingBara/recordings' + OUTPUT_FILENAME + '/'
output_dir = '/Users/cici/Documents/VS Code/CodingBara/melspects/output.pt'
model = os.open('best_model.keras')

""" standardize audio duration by cutting or padding and convert from 1d audio to 2d mel spectrogram tensor """
def prep_files(input_dir, output_dir, target_duration_s=3, n_mels=64):
    """
    Cuts or pads with silence all audio files in a directory to a target duration
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Supported extensions (pydub handles these if ffmpeg is installed)
    valid_extensions = ('.wav')

    try:
        # Load audio file
        audio, sr = librosa.load(input_dir, sr=22050)
        current_duration = len(audio) / sr
        print(f"Current duration: {current_duration}")
        target_duration_s = 3

        if current_duration > target_duration_s:
            # Cut the audio if it's too long
            processed_audio = audio[:66150]
        elif current_duration < target_duration_s:
            # Pad with silence if it's too short
            processed_audio = np.pad(audio, (0, max(0, int(target_duration_s - current_duration))), 'constant')
            # silence_needed = target_duration_s - current_duration
            # silence = AudioSegment.silent(duration=silence_needed, frame_rate=sr)
            # processed_audio = audio + silence
        else:
            processed_audio = audio

        # mel filter banks
        filter_banks = librosa.filters.mel(sr=sr, n_fft=1024, n_mels=n_mels)

        # Convert 1D wave signal to 2D Mel Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(
            y=processed_audio, sr=sr, n_mels=n_mels, n_fft=1024, hop_length=512
        )            
            
        # Convert power to decibels (log scale matches human hearing)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

        """
        plt.figure(figsize=(25, 10))
        librosa.display.specshow(log_mel_spectrogram, 
                                x_axis="time", 
                                y_axis="mel", 
                                sr=sr)
        plt.colorbar(format="%+2.f") 
        plt.show()
        """

        # Add a Channel dimension (1, n_mels, time_steps) to match CNN expectation
        # PyTorch CNNs expect: (Batch, Channel, Height, Width)
        audio_tensor = torch.tensor(log_mel_spectrogram, dtype=torch.float32).unsqueeze(0)

        torch.save(audio_tensor, output_path)

    except Exception as e:
        print(f"Failed to process {file_name}: {e}")

prep_files(input_dir, output_dir)
