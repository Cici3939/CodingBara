import asyncio
import os
import sys
import cv2
import numpy as np
import wave
import time
import threading
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo
from scipy.signal import resample_poly

import board
import digitalio
import adafruit_character_lcd.character_lcd as characterlcd

from picamera2 import Picamera2
from google import genai
from google.genai import types

import keras
import librosa
import torch

sys.path.insert(0, "/home/cosmos")
from CNN import SpecAugment


# ============================================================
# CONFIGURATION
# ============================================================

MODEL = "gemini-3.1-flash-live-preview"

# Use 16 kHz directly from the microphone.
# This avoids realtime resampling and keeps the microphone loop fast.
MIC_RATE = 16000
GEMINI_RATE = 16000
SER_RATE = 22050
SPEAKER_RATE = 24000

# Gemini audio can arrive in bursts.  Buffer this much audio before
# starting ALSA so short network/SDK gaps do not cause playback underruns.
SPEAKER_PREBUFFER_SECONDS = 0.80
SPEAKER_PREBUFFER_BYTES = int(SPEAKER_RATE * 2 * SPEAKER_PREBUFFER_SECONDS)

MIC_DEVICE = "plughw:3,0"
SPEAKER_DEVICE = "plughw:CARD=Headphones,DEV=0"

# 100 ms chunks.
MIC_CHUNK = 1600

AUDIO_FOLDER = "/home/cosmos/CBAudio"
PIC_FOLDER = "/home/cosmos/CBPic"
MELSPEC_FOLDER = "/home/cosmos/MelSpec"

AUDIO_FILE = os.path.join(AUDIO_FOLDER, "user_speaking.wav")
PIC_FILE = os.path.join(PIC_FOLDER, "user_speaking.jpg")
MELSPEC_FILE = os.path.join(MELSPEC_FOLDER, "user_speaking.pt")

SER_MODEL_PATH = "/home/cosmos/Downloads/best_model.keras"
FER_MODEL_PATH = "/home/cosmos/Downloads/best_FER_model.keras"

EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Happy",
    3: "Fear",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}

# SER uses the first ~3 seconds of the user's turn.
AUDIO_CLIP_SECONDS = 3
AUDIO_CLIP_SAMPLES = GEMINI_RATE * AUDIO_CLIP_SECONDS
AUDIO_CLIP_BYTES = AUDIO_CLIP_SAMPLES * 2

# Manual VAD.
VAD_MIN_THRESHOLD = 500
VAD_NOISE_MULTIPLIER = 2.5
VAD_SILENCE_SECONDS = 0.45
VAD_PREROLL_CHUNKS = 5

# We do not want a slow SER/FER calculation to hold up Gemini.
# It is allowed to contribute emotion data for a short time before
# ActivityEnd. If it is not ready, Gemini responds immediately.
SER_FER_WAIT_SECONDS = 1.00

RECONNECT_DELAY = 2

# Latest API key already used by the project.
client = genai.Client(api_key="API key here")

running = True

# ============================================================
# LCD CLOCK
#
# 16x2 parallel character LCD, no backpack.
#
# LCD pin -> Raspberry Pi GPIO
#   4  RS -> GPIO22
#   6  E  -> GPIO17
#  11  D4 -> GPIO25
#  12  D5 -> GPIO24
#  13  D6 -> GPIO23
#  14  D7 -> GPIO18
#
# LCD pins 1/5/16 -> GND
# LCD pins 2/15    -> power
# LCD pin 3        -> GND through contrast resistor
# ============================================================

LCD_COLUMNS = 16
LCD_ROWS = 2

lcd_rs = digitalio.DigitalInOut(board.D22)
lcd_en = digitalio.DigitalInOut(board.D17)
lcd_d4 = digitalio.DigitalInOut(board.D25)
lcd_d5 = digitalio.DigitalInOut(board.D24)
lcd_d6 = digitalio.DigitalInOut(board.D23)
lcd_d7 = digitalio.DigitalInOut(board.D18)

lcd = None
lcd_thread = None


def lcd_clock_worker():
    global lcd

    try:
        lcd = characterlcd.Character_LCD_Mono(
            lcd_rs,
            lcd_en,
            lcd_d4,
            lcd_d5,
            lcd_d6,
            lcd_d7,
            LCD_COLUMNS,
            LCD_ROWS
        )

        lcd.clear()
        print("LCD CLOCK ACTIVE", flush=True)

        pacific = ZoneInfo("America/Los_Angeles")

        while running:
            now = datetime.now(pacific)

            # 16 characters maximum per line.
            line1 = now.strftime("%I:%M:%S %p %Z").strip()
            line2 = now.strftime("%b %d, %Y")

            lcd.message = (
                line1[:LCD_COLUMNS]
                + "\n"
                + line2[:LCD_COLUMNS]
            )

            time.sleep(1.0)

    except Exception as e:
        # LCD failure must NEVER kill Cappy.
        print("LCD ERROR:", repr(e), flush=True)

    finally:
        try:
            if lcd is not None:
                lcd.clear()
        except Exception:
            pass

        print("LCD CLOCK STOPPED.", flush=True)


def start_lcd_clock():
    global lcd_thread

    lcd_thread = threading.Thread(
        target=lcd_clock_worker,
        name="LCDClock",
        daemon=True
    )

    lcd_thread.start()


# Camera state.
camera = None
camera_ready = False
camera_lock = threading.Lock()

# Emotion models.
ser_model = None
fer_model = None


# ============================================================
# SHARED STATE
# ============================================================

class SpeakingEvent(asyncio.Event):
    pass


# ============================================================
# AUDIO HELPERS
# ============================================================

def save_wave(data):
    if not data:
        print("AUDIO SAVE FAILED: NO AUDIO", flush=True)
        return False

    if len(data) < AUDIO_CLIP_BYTES:
        data = data + b"\x00" * (AUDIO_CLIP_BYTES - len(data))

    try:
        with wave.open(AUDIO_FILE, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(GEMINI_RATE)
            wav.writeframes(data)

        print("AUDIO SAVED", flush=True)
        return True

    except Exception as e:
        print("WAVE SAVE ERROR:", repr(e), flush=True)
        return False


def add_history(history, state, data):
    history.append(data)
    state[0] += len(data)

    while state[0] > AUDIO_CLIP_BYTES:
        old = history.popleft()
        state[0] -= len(old)


def get_history(history):
    data = b"".join(history)

    if len(data) > AUDIO_CLIP_BYTES:
        data = data[-AUDIO_CLIP_BYTES:]

    if len(data) < AUDIO_CLIP_BYTES:
        data += b"\x00" * (AUDIO_CLIP_BYTES - len(data))

    return data


# ============================================================
# SER
# ============================================================

def make_ser_input(audio_data):
    audio = (
        np.frombuffer(audio_data, dtype=np.int16)
        .astype(np.float32)
        / 32768.0
    )

    if len(audio) > AUDIO_CLIP_SAMPLES:
        audio = audio[:AUDIO_CLIP_SAMPLES]
    elif len(audio) < AUDIO_CLIP_SAMPLES:
        audio = np.pad(
            audio,
            (0, AUDIO_CLIP_SAMPLES - len(audio)),
            mode="constant"
        )

    # One-time resample outside the realtime microphone path.
    audio = resample_audio_for_ser(audio)

    target_samples = AUDIO_CLIP_SECONDS * SER_RATE

    if len(audio) > target_samples:
        audio = audio[:target_samples]
    elif len(audio) < target_samples:
        audio = np.pad(
            audio,
            (0, target_samples - len(audio)),
            mode="constant"
        )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SER_RATE,
        n_mels=64,
        n_fft=1024,
        hop_length=512
    )

    mel = librosa.power_to_db(mel).astype(np.float32)

    if mel.shape[1] < 130:
        mel = np.pad(
            mel,
            ((0, 0), (0, 130 - mel.shape[1])),
            mode="constant"
        )
    elif mel.shape[1] > 130:
        mel = mel[:, :130]

    mel = np.expand_dims(mel, axis=-1)

    os.makedirs(MELSPEC_FOLDER, exist_ok=True)
    torch.save(mel, MELSPEC_FILE)

    return mel


def resample_audio_for_ser(audio):
    return resample_poly(
        audio,
        SER_RATE,
        GEMINI_RATE
    ).astype(np.float32)


def predict_ser(audio_data):
    if ser_model is None:
        return None, None

    try:
        model_input = np.expand_dims(
            make_ser_input(audio_data),
            axis=0
        )

        prediction = ser_model.predict(
            model_input,
            verbose=0
        )[0]

        cls = int(np.argmax(prediction))
        confidence = float(prediction[cls] * 100.0)
        emotion = EMOTION_LABELS.get(cls, "Unknown")

        print("", flush=True)
        print("--- SER ---", flush=True)
        print("Emotion:", emotion, flush=True)
        print(f"Confidence: {confidence:.2f}%", flush=True)
        print("", flush=True)

        return emotion, confidence

    except Exception as e:
        print("SER ERROR:", repr(e), flush=True)
        return None, None


# ============================================================
# FER
# ============================================================

def fer_preprocess(frame):
    if fer_model is None:
        return None

    shape = fer_model.input_shape

    # Expected image dimensions.
    height = 96
    width = 96
    channels = 3

    if isinstance(shape, tuple) and len(shape) == 4:
        if shape[1] is not None:
            height = int(shape[1])
        if shape[2] is not None:
            width = int(shape[2])
        if shape[3] is not None:
            channels = int(shape[3])

    image = cv2.resize(
        frame,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    if channels == 1:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2GRAY
        )
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=-1)

    else:
        image = image.astype(np.float32) / 255.0

    return np.expand_dims(image, axis=0)


def predict_fer_from_frame(frame):
    if fer_model is None or frame is None:
        return None, None

    try:
        model_input = fer_preprocess(frame)

        prediction = fer_model.predict(
            model_input,
            verbose=0
        )[0]

        cls = int(np.argmax(prediction))
        confidence = float(prediction[cls] * 100.0)
        emotion = EMOTION_LABELS.get(cls, "Unknown")

        print("--- FER ---", flush=True)
        print(
            f"{emotion} {confidence:.1f}%",
            flush=True
        )

        return emotion, confidence

    except Exception as e:
        print("FER ERROR:", repr(e), flush=True)
        return None, None


def capture_and_run_fer():
    global camera
    global camera_ready

    if camera is None or not camera_ready:
        print("CAMERA NOT READY", flush=True)
        return None, None

    try:
        with camera_lock:
            if camera is None or not camera_ready:
                return None, None

            frame = camera.capture_array()

        if frame is None:
            print("CAMERA RETURNED NO FRAME", flush=True)
            return None, None

        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]

        # Save a correctly colored RGB image as BGR JPEG.
        bgr = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2BGR
        )

        success = cv2.imwrite(
            PIC_FILE,
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, 90]
        )

        if success:
            print("96x96 PHOTO SAVED", flush=True)
        else:
            print("PICTURE SAVE FAILED", flush=True)

        return predict_fer_from_frame(frame)

    except Exception as e:
        print("FER/CAMERA ERROR:", repr(e), flush=True)
        return None, None


# ============================================================
# MODEL LOADING
# ============================================================

def load_models():
    global ser_model
    global fer_model

    print("Loading SER model...", flush=True)

    ser_model = keras.models.load_model(
        SER_MODEL_PATH,
        custom_objects={
            "SpecAugment": SpecAugment
        }
    )

    print("SER MODEL LOADED", flush=True)

    # Warm SER so the first real inference is faster.
    try:
        dummy = np.zeros(
            (1, 64, 130, 1),
            dtype=np.float32
        )
        ser_model.predict(dummy, verbose=0)
        print("SER WARMED UP", flush=True)
    except Exception as e:
        print("SER WARMUP ERROR:", repr(e), flush=True)

    print("Loading FER model...", flush=True)

    try:
        fer_model = keras.models.load_model(
            FER_MODEL_PATH
        )

        print("FER MODEL LOADED", flush=True)

        # Warm FER using the model's declared input shape.
        try:
            shape = fer_model.input_shape

            height = int(shape[1]) if shape[1] is not None else 96
            width = int(shape[2]) if shape[2] is not None else 96
            channels = int(shape[3]) if shape[3] is not None else 3

            dummy = np.zeros(
                (1, height, width, channels),
                dtype=np.float32
            )

            fer_model.predict(dummy, verbose=0)

            print("FER WARMED UP", flush=True)

        except Exception as e:
            print("FER WARMUP ERROR:", repr(e), flush=True)

    except Exception as e:
        fer_model = None
        print("FER MODEL FAILED TO LOAD:", repr(e), flush=True)
        print("Continuing without FER.", flush=True)


# ============================================================
# CAMERA
# ============================================================

async def camera_task():
    global camera
    global camera_ready

    print("Starting Pi Camera...", flush=True)

    local_camera = None

    try:
        local_camera = Picamera2()

        config = local_camera.create_preview_configuration(
            main={
                "size": (96, 96),
                "format": "RGB888"
            }
        )

        local_camera.configure(config)
        local_camera.start()

        await asyncio.sleep(1.0)

        try:
            local_camera.set_controls(
                {
                    "AwbEnable": True
                }
            )
        except Exception as e:
            print(
                "WHITE BALANCE CONTROL:",
                repr(e),
                flush=True
            )

        camera = local_camera
        camera_ready = True

        print("PI CAMERA ACTIVE", flush=True)
        print("CAMERA READY", flush=True)

        while running:
            await asyncio.sleep(0.5)

    except asyncio.CancelledError:
        pass

    except Exception as e:
        # Camera failure must NEVER kill Gemini.
        camera_ready = False
        camera = None

        print(
            "CAMERA ERROR:",
            repr(e),
            flush=True
        )

    finally:
        camera_ready = False

        if local_camera is not None:
            try:
                local_camera.stop()
            except Exception:
                pass

            try:
                local_camera.close()
            except Exception:
                pass

        if camera is local_camera:
            camera = None

        print("Camera task stopped.", flush=True)


# ============================================================
# GEMINI INPUT SENDER
#
# IMPORTANT:
# The microphone NEVER waits for session.send_realtime_input().
#
# The microphone puts events into this queue immediately.
# This prevents network/API latency from making the microphone
# fall behind and losing the beginning of the user's sentence.
# ============================================================

async def gemini_input_sender(
    session,
    input_queue
):
    print("Gemini input sender started.", flush=True)

    try:
        while running:
            item = await input_queue.get()

            try:
                kind = item[0]
                payload = item[1]

                if kind == "start":
                    await session.send_realtime_input(
                        activity_start=types.ActivityStart()
                    )

                    print(
                        "ACTIVITY START SENT",
                        flush=True
                    )

                elif kind == "audio":
                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=payload,
                            mime_type="audio/pcm;rate=16000"
                        )
                    )

                elif kind == "finish":
                    # All previous audio is already ahead of this
                    # queue item, so it has been sent first.
                    #
                    # payload:
                    #   (ser_task, fer_task)
                    ser_task, fer_task = payload

                    ser_result = None
                    fer_result = None

                    wait_tasks = []

                    if ser_task is not None:
                        wait_tasks.append(ser_task)

                    if fer_task is not None:
                        wait_tasks.append(fer_task)

                    if wait_tasks:
                        done, pending = await asyncio.wait(
                            wait_tasks,
                            timeout=SER_FER_WAIT_SECONDS
                        )

                        for task in done:
                            try:
                                result = task.result()

                                if task is ser_task:
                                    ser_result = result

                                elif task is fer_task:
                                    fer_result = result

                            except Exception as e:
                                print(
                                    "EMOTION TASK ERROR:",
                                    repr(e),
                                    flush=True
                                )

                        if pending:
                            print(
                                "EMOTION STILL PROCESSING; RESPONSE WILL NOT WAIT LONGER",
                                flush=True
                            )

                    # Never cancel unfinished emotion tasks.
                    # They are allowed to finish in the background.
                    emotion_lines = []

                    if ser_result is not None:
                        emotion, confidence = ser_result

                        if emotion is not None:
                            emotion_lines.append(
                                f"Speech emotion: {emotion} "
                                f"({confidence:.2f}%)"
                            )

                    if fer_result is not None:
                        emotion, confidence = fer_result

                        if emotion is not None:
                            emotion_lines.append(
                                f"Facial emotion: {emotion} "
                                f"({confidence:.2f}%)"
                            )

                    if emotion_lines:
                        emotion_message = (
                            "[INTERNAL EMOTION DATA]\n"
                            + "\n".join(emotion_lines)
                            + "\n[/INTERNAL EMOTION DATA]"
                        )

                        print("SENDING EMOTION CONTEXT:", flush=True)
                        print(emotion_message, flush=True)

                        await session.send_realtime_input(
                            text=emotion_message
                        )

                        print("EMOTION DATA SENT TO CAPPY", flush=True)

                    await session.send_realtime_input(
                        activity_end=types.ActivityEnd()
                    )

                    print(
                        "ACTIVITY END SENT",
                        flush=True
                    )

                    print(
                        "CAPPY CAN NOW RESPOND",
                        flush=True
                    )

                elif kind == "text":
                    await session.send_realtime_input(
                        text=payload
                    )

            finally:
                input_queue.task_done()

    except asyncio.CancelledError:
        raise

    except Exception as e:
        print(
            "GEMINI INPUT SENDER ERROR:",
            repr(e),
            flush=True
        )
        raise

    finally:
        print(
            "Gemini input sender stopped.",
            flush=True
        )


# ============================================================
# MICROPHONE
# ============================================================

async def microphone(
    input_queue,
    gemini_speaking,
    turn_busy
):
    print("Microphone task started.", flush=True)

    mic = None

    speech_active = False
    last_voice_time = 0.0
    noise_floor = 300.0

    preroll = deque(
        maxlen=VAD_PREROLL_CHUNKS
    )

    # Audio belonging to the current user turn.
    turn_history = deque()
    turn_history_size = [0]

    # First ~3 seconds of the turn for SER.
    beginning_audio = bytearray()
    ser_task = None

    # FER starts immediately when the user begins speaking.
    fer_task = None

    try:
        print("Opening microphone...", flush=True)

        mic = await asyncio.create_subprocess_exec(
            "arecord",
            "-D", MIC_DEVICE,
            "-f", "S16_LE",
            "-r", str(MIC_RATE),
            "-c", "1",
            "-t", "raw",
            "-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL
        )

        print(
            "MIC ACTIVE 16kHz",
            flush=True
        )

        while running:
            # ------------------------------------------------
            # CRITICAL:
            # This read is the ONLY thing that must happen
            # synchronously with the microphone.
            #
            # No Gemini network operation occurs in this loop.
            # ------------------------------------------------

            data = await mic.stdout.read(
                MIC_CHUNK * 2
            )

            if not data:
                raise RuntimeError(
                    "MIC STREAM ENDED"
                )

            pcm = np.frombuffer(
                data,
                dtype=np.int16
            )

            if len(pcm) == 0:
                continue

            # Direct 16 kHz path.
            audio_bytes = pcm.tobytes()

            rms = float(
                np.sqrt(
                    np.mean(
                        pcm.astype(np.float32) ** 2
                    )
                )
            )

            # Update noise floor only while idle.
            if (
                not speech_active
                and not turn_busy.is_set()
                and rms < noise_floor * 2.0
            ):
                noise_floor = (
                    noise_floor * 0.95
                    + rms * 0.05
                )

            threshold = max(
                VAD_MIN_THRESHOLD,
                noise_floor * VAD_NOISE_MULTIPLIER
            )

            voice_detected = (
                rms > threshold
            )

            # ------------------------------------------------
            # CAPPY IS SPEAKING
            #
            # Hardware microphone stays alive.
            # Everything is discarded.
            # ------------------------------------------------

            if gemini_speaking.is_set():
                preroll.clear()
                speech_active = False
                turn_history.clear()
                turn_history_size[0] = 0
                beginning_audio.clear()

                if ser_task is not None and ser_task.done():
                    ser_task = None

                if fer_task is not None and fer_task.done():
                    fer_task = None

                continue

            # ------------------------------------------------
            # WAITING FOR CAPPY TO FINISH
            # ------------------------------------------------

            if (
                turn_busy.is_set()
                and not speech_active
            ):
                preroll.clear()
                continue

            # ------------------------------------------------
            # WAITING FOR USER TO START
            # ------------------------------------------------

            if not speech_active:
                preroll.append(audio_bytes)

                if voice_detected:
                    speech_active = True
                    last_voice_time = time.monotonic()

                    turn_busy.set()

                    # Start a completely new user-turn buffer.
                    turn_history.clear()
                    turn_history_size[0] = 0

                    beginning_audio.clear()

                    ser_task = None
                    fer_task = None

                    # Include the preroll so the beginning of
                    # the user's first word is not clipped.
                    preroll_audio = list(preroll)

                    for old_audio in preroll_audio:
                        add_history(
                            turn_history,
                            turn_history_size,
                            old_audio
                        )

                        if (
                            len(beginning_audio)
                            < AUDIO_CLIP_BYTES
                        ):
                            remaining = (
                                AUDIO_CLIP_BYTES
                                - len(beginning_audio)
                            )

                            beginning_audio.extend(
                                old_audio[:remaining]
                            )

                    preroll.clear()

                    print("", flush=True)
                    print(
                        "USER SPEECH DETECTED",
                        flush=True
                    )
                    print(
                        f"RMS: {rms:.0f} "
                        f"Threshold: {threshold:.0f}",
                        flush=True
                    )
                    print(
                        "STARTING GEMINI USER TURN",
                        flush=True
                    )

                    # Queue ActivityStart immediately.
                    input_queue.put_nowait(
                        ("start", None)
                    )

                    # Queue the entire preroll immediately.
                    for old_audio in preroll_audio:
                        input_queue.put_nowait(
                            ("audio", old_audio)
                        )

                    # Start FER immediately in background.
                    fer_task = asyncio.create_task(
                        asyncio.to_thread(
                            capture_and_run_fer
                        )
                    )

                    # Start SER immediately if the preroll
                    # already contains enough audio.
                    if (
                        len(beginning_audio)
                        >= AUDIO_CLIP_BYTES
                    ):
                        ser_sample = bytes(
                            beginning_audio[
                                :AUDIO_CLIP_BYTES
                            ]
                        )

                        ser_task = asyncio.create_task(
                            asyncio.to_thread(
                                predict_ser,
                                ser_sample
                            )
                        )

                    # No separate send of audio_bytes here:
                    # it is already inside preroll_audio.

            # ------------------------------------------------
            # USER IS CURRENTLY SPEAKING
            # ------------------------------------------------

            else:
                add_history(
                    turn_history,
                    turn_history_size,
                    audio_bytes
                )

                if (
                    len(beginning_audio)
                    < AUDIO_CLIP_BYTES
                ):
                    remaining = (
                        AUDIO_CLIP_BYTES
                        - len(beginning_audio)
                    )

                    beginning_audio.extend(
                        audio_bytes[:remaining]
                    )

                # ------------------------------------------------
                # EARLY SER:
                # As soon as the first 3 seconds exist, start
                # SER in the background.
                # ------------------------------------------------

                if (
                    ser_task is None
                    and len(beginning_audio)
                    >= AUDIO_CLIP_BYTES
                ):
                    ser_sample = bytes(
                        beginning_audio[
                            :AUDIO_CLIP_BYTES
                        ]
                    )

                    ser_task = asyncio.create_task(
                        asyncio.to_thread(
                            predict_ser,
                            ser_sample
                        )
                    )

                    print(
                        "EARLY SER STARTED",
                        flush=True
                    )

                # ------------------------------------------------
                # CRITICAL:
                # DO NOT await Gemini here.
                #
                # Put audio into the local queue immediately.
                # ------------------------------------------------

                input_queue.put_nowait(
                    ("audio", audio_bytes)
                )

                if voice_detected:
                    last_voice_time = time.monotonic()

                elif (
                    time.monotonic()
                    - last_voice_time
                    >= VAD_SILENCE_SECONDS
                ):
                    speech_active = False

                    print(
                        "USER STOPPED SPEAKING",
                        flush=True
                    )

                    # Save the latest 3 seconds of this user turn.
                    audio_data = get_history(
                        turn_history
                    )

                    asyncio.create_task(
                        asyncio.to_thread(
                            save_wave,
                            audio_data
                        )
                    )

                    # If the user spoke less than 3 seconds,
                    # still run SER on the available beginning.
                    if ser_task is None:
                        ser_sample = bytes(
                            beginning_audio
                        )

                        if not ser_sample:
                            ser_sample = audio_data

                        if len(ser_sample) < AUDIO_CLIP_BYTES:
                            ser_sample += (
                                b"\x00"
                                * (
                                    AUDIO_CLIP_BYTES
                                    - len(ser_sample)
                                )
                            )

                        ser_sample = ser_sample[
                            :AUDIO_CLIP_BYTES
                        ]

                        ser_task = asyncio.create_task(
                            asyncio.to_thread(
                                predict_ser,
                                ser_sample
                            )
                        )

                    # IMPORTANT:
                    # ActivityEnd is also queued.
                    #
                    # It comes AFTER every audio packet already
                    # placed into input_queue. The sender therefore
                    # cannot end the turn before the final audio.
                    input_queue.put_nowait(
                        (
                            "finish",
                            (
                                ser_task,
                                fer_task
                            )
                        )
                    )

                    print(
                        "AUDIO QUEUED FOR GEMINI",
                        flush=True
                    )

                    # Do NOT clear turn_busy here.
                    #
                    # It remains set until the speaker receives
                    # Gemini's turn-complete marker.
                    turn_history.clear()
                    turn_history_size[0] = 0
                    beginning_audio.clear()

                    ser_task = None
                    fer_task = None

    except asyncio.CancelledError:
        raise

    except Exception as e:
        print(
            "MICROPHONE ERROR:",
            repr(e),
            flush=True
        )
        raise

    finally:
        if mic is not None:
            try:
                mic.terminate()
            except Exception:
                pass

            try:
                await mic.wait()
            except Exception:
                pass

        print(
            "Microphone closed.",
            flush=True
        )


# ============================================================
# SPEAKER
# ============================================================

async def speaker(
    audio_queue,
    gemini_speaking,
    turn_busy
):
    """
    Play Gemini's 24 kHz mono PCM output.

    IMPORTANT:
    Gemini Live audio can arrive in uneven bursts.  Starting aplay on the
    very first chunk makes ALSA underrun whenever the next network chunk is
    late.  We therefore collect ~0.8 seconds before starting playback.
    After playback starts, audio continues through the queue normally.
    """
    print("Speaker task started.", flush=True)

    player = None
    stderr_task = None
    pending = bytearray()
    playback_started = False

    async def start_player():
        nonlocal stderr_task

        proc = await asyncio.create_subprocess_exec(
            "aplay",
            "--quiet",
            "--device", SPEAKER_DEVICE,
            "--file-type=raw",
            "--format=S16_LE",
            "--rate", str(SPEAKER_RATE),
            "--channels=1",
            "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE
        )

        async def drain_stderr():
            if proc.stderr is None:
                return
            try:
                while True:
                    line = await proc.stderr.readline()
                    if not line:
                        break
                    text = line.decode(errors="replace").strip()
                    if text:
                        print("APLAY:", text, flush=True)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print("APLAY STDERR ERROR:", repr(e), flush=True)

        stderr_task = asyncio.create_task(drain_stderr())
        return proc

    async def close_player():
        nonlocal player, stderr_task

        if player is None:
            return

        proc = player
        player = None

        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass

        try:
            await proc.wait()
        except Exception:
            pass

        if stderr_task is not None:
            try:
                await asyncio.wait_for(stderr_task, timeout=0.5)
            except Exception:
                stderr_task.cancel()
                try:
                    await stderr_task
                except Exception:
                    pass
            stderr_task = None

    async def write_audio(data):
        """Write PCM to aplay, restarting once if its pipe dies."""
        nonlocal player

        if player is None or player.stdin is None:
            player = await start_player()
            if player.stdin is None:
                raise RuntimeError("APLAY STDIN WAS NOT CREATED")
            print("APLAY STARTED FOR CAPPY TURN", flush=True)

        try:
            player.stdin.write(data)
            await player.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as e:
            print("APLAY PIPE ERROR:", repr(e), flush=True)
            await close_player()

            player = await start_player()
            if player.stdin is None:
                raise RuntimeError("APLAY STDIN WAS NOT CREATED AFTER RESTART")

            player.stdin.write(data)
            await player.stdin.drain()

    async def start_playback_if_ready(force=False):
        """Start playback once enough audio is buffered, or on turn end."""
        nonlocal pending, playback_started

        if playback_started or not pending:
            return

        if not force and len(pending) < SPEAKER_PREBUFFER_BYTES:
            return

        buffered_seconds = len(pending) / (SPEAKER_RATE * 2)
        print(
            f"SPEAKER PREBUFFER READY: {buffered_seconds:.2f}s",
            flush=True
        )

        playback_started = True
        data = bytes(pending)
        pending.clear()
        await write_audio(data)

    try:
        print("SPEAKER READY", flush=True)
        print(
            f"SPEAKER: {SPEAKER_DEVICE} | RAW S16_LE | "
            f"{SPEAKER_RATE} Hz | MONO",
            flush=True
        )
        print(
            f"SPEAKER PREBUFFER: {SPEAKER_PREBUFFER_SECONDS:.2f}s",
            flush=True
        )

        while running:
            audio = await audio_queue.get()

            try:
                # ------------------------------------------------
                # END OF GEMINI TURN
                # ------------------------------------------------
                if audio is None:
                    # If the response was shorter than the prebuffer, play
                    # whatever we have rather than throwing it away.
                    await start_playback_if_ready(force=True)

                    # Wait for all PCM already written to aplay to finish.
                    await close_player()

                    pending.clear()
                    playback_started = False
                    gemini_speaking.clear()
                    turn_busy.clear()

                    print("", flush=True)
                    print("CAPY FINISHED SPEAKING", flush=True)
                    print("MIC AUDIO ACTIVE", flush=True)
                    print("READY FOR USER", flush=True)
                    continue

                if not audio:
                    continue

                if isinstance(audio, str):
                    print("DROPPING STRING AUDIO DATA", flush=True)
                    continue

                if len(audio) & 1:
                    print("DROPPING ODD PCM BYTE", flush=True)
                    audio = audio[:-1]

                if not audio:
                    continue

                # ------------------------------------------------
                # PREBUFFER
                # ------------------------------------------------
                if not playback_started:
                    pending.extend(audio)
                    await start_playback_if_ready(force=False)
                    continue

                # ------------------------------------------------
                # NORMAL STREAMING PLAYBACK
                # ------------------------------------------------
                await write_audio(audio)

            finally:
                audio_queue.task_done()

    except asyncio.CancelledError:
        raise

    except Exception as e:
        print("SPEAKER ERROR:", repr(e), flush=True)
        raise

    finally:
        pending.clear()
        await close_player()
        gemini_speaking.clear()
        turn_busy.clear()
        print("Speaker closed.", flush=True)


# ============================================================
# GEMINI RECEIVE
#
# IMPORTANT:
# session.receive() can finish its iterator at the end of a
# server turn. That does NOT necessarily mean the Live session
# died. Re-enter receive() and continue.
# ============================================================

async def receive(
    session,
    audio_queue,
    gemini_speaking,
    turn_busy
):
    print("Receiver started.", flush=True)

    try:
        while running:
            async for response in session.receive():

                if not running:
                    return

                content = response.server_content

                if content is None:
                    continue

                # ------------------------------------------------
                # INTERRUPTION
                # ------------------------------------------------

                if getattr(
                    content,
                    "interrupted",
                    False
                ):
                    print(
                        "CAPY INTERRUPTED",
                        flush=True
                    )

                    while True:
                        try:
                            old = audio_queue.get_nowait()
                            audio_queue.task_done()

                            if old is None:
                                continue

                        except asyncio.QueueEmpty:
                            break

                    gemini_speaking.clear()

                    continue

                # ------------------------------------------------
                # USER TRANSCRIPTION
                # ------------------------------------------------

                transcription = getattr(
                    content,
                    "input_transcription",
                    None
                )

                if transcription is not None:
                    text = getattr(
                        transcription,
                        "text",
                        None
                    )

                    if text:
                        print(
                            "USER:",
                            text,
                            flush=True
                        )

                # ------------------------------------------------
                # CAPPY OUTPUT
                # ------------------------------------------------

                model_turn = getattr(
                    content,
                    "model_turn",
                    None
                )

                if model_turn is not None:
                    for part in model_turn.parts:

                        text = getattr(
                            part,
                            "text",
                            None
                        )

                        if text:
                            print(
                                "CAPPY:",
                                text,
                                flush=True
                            )

                        inline_data = getattr(
                            part,
                            "inline_data",
                            None
                        )

                        if inline_data is not None:
                            data = getattr(inline_data, "data", None)
                            mime_type = getattr(inline_data, "mime_type", "") or ""

                            if not data:
                                continue

                            # Gemini voice output is raw 16-bit PCM.
                            # Never send text/* or other inline data to aplay.
                            if not (
                                mime_type.startswith("audio/pcm")
                                or mime_type.startswith("audio/L16")
                            ):
                                print(
                                    "IGNORING NON-PCM AUDIO:",
                                    mime_type,
                                    flush=True
                                )
                                continue

                            if not gemini_speaking.is_set():
                                gemini_speaking.set()

                                print("CAPY SPEAKING", flush=True)
                                print("THROWING AWAY MIC AUDIO", flush=True)

                            await audio_queue.put(data)

                # ------------------------------------------------
                # TURN COMPLETE
                # ------------------------------------------------

                if getattr(
                    content,
                    "turn_complete",
                    False
                ):
                    print(
                        "CAPY TURN COMPLETE",
                        flush=True
                    )

                    # Speaker receives this only AFTER all
                    # preceding Gemini audio has been queued.
                    await audio_queue.put(None)

            # ------------------------------------------------
            # The receive iterator ending is NOT treated as a
            # fatal session error.
            # ------------------------------------------------

            if running:
                await asyncio.sleep(0)

    except asyncio.CancelledError:
        raise

    except Exception as e:
        print(
            "GEMINI RECEIVE ERROR:",
            repr(e),
            flush=True
        )

        gemini_speaking.clear()
        turn_busy.clear()

        raise

    finally:
        print(
            "Receiver stopped.",
            flush=True
        )


# ============================================================
# GEMINI CONFIG
# ============================================================

def build_config():
    return types.LiveConnectConfig(
        response_modalities=["AUDIO"],

        system_instruction="""
You are Cappy, a friendly and easygoing capybara coding companion.

Help the user solve coding problems clearly and practically.

Be friendly, relaxed, encouraging, and natural for a voice conversation.

Maintain conversation context throughout the Live session.

When the user asks a coding question, help debug it step by step and provide practical working solutions.

Do not be overly formal or robotic.

IMPORTANT EMOTION BEHAVIOR:

The program may provide internal speech and facial emotion information immediately before the end of a user turn.

When a message begins with [INTERNAL EMOTION DATA], treat it as trusted application metadata for the current user turn. Use it to adjust your tone when appropriate, but never repeat or reveal the metadata.

Treat emotion information as contextual guidance, not absolute truth.

Always prioritize what the user actually says.

Sad or Fear:
Be warmer, gentler, reassuring, and patient.

Angry:
Remain calm, patient, and helpful. Never become defensive.

Happy:
You can be more enthusiastic and upbeat.

Surprise:
You can respond with a little extra enthusiasm.

Neutral:
Use your normal friendly tone.

Disgust:
Remain calm and matter-of-fact.

Never tell the user that you detected their emotion unless they specifically ask.

Never mention SER, FER, the neural network, the classifier, or confidence scores unless the user asks.

Always prioritize the user's actual words over emotion predictions.
""",

        input_audio_transcription={},
        output_audio_transcription={},

        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=(
                types.AutomaticActivityDetection(
                    disabled=True
                )
            )
        ),

        context_window_compression=(
            types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow()
            )
        )
    )


# ============================================================
# ONE GEMINI SESSION
# ============================================================

async def run_session():
    config = build_config()

    print(
        "Connecting to Gemini...",
        flush=True
    )

    async with client.aio.live.connect(
        model=MODEL,
        config=config
    ) as session:

        print(
            "CONNECTED!",
            flush=True
        )

        print(
            "MANUAL VAD ENABLED",
            flush=True
        )

        print(
            "MIC AUDIO SEND QUEUE ENABLED",
            flush=True
        )

        print(
            "RECEIVE LOOP PERSISTS ACROSS TURNS",
            flush=True
        )

        print(
            "EARLY SER ENABLED",
            flush=True
        )

        print(
            "FER BACKGROUND ENABLED",
            flush=True
        )

        print(
            "MIC AUDIO DISCARDED DURING CAPPY SPEECH",
            flush=True
        )

        print(
            "READY FOR USER",
            flush=True
        )

        # Unbounded queues are intentional.
        #
        # A bounded input queue could force the microphone to
        # wait when the network is temporarily slower than realtime.
        # That is exactly what we are trying to prevent.
        input_queue = asyncio.Queue()

        audio_queue = asyncio.Queue()

        gemini_speaking = SpeakingEvent()
        turn_busy = SpeakingEvent()

        sender_task = asyncio.create_task(
            gemini_input_sender(
                session,
                input_queue
            )
        )

        mic_task = asyncio.create_task(
            microphone(
                input_queue,
                gemini_speaking,
                turn_busy
            )
        )

        receive_task = asyncio.create_task(
            receive(
                session,
                audio_queue,
                gemini_speaking,
                turn_busy
            )
        )

        speaker_task = asyncio.create_task(
            speaker(
                audio_queue,
                gemini_speaking,
                turn_busy
            )
        )

        tasks = (
            sender_task,
            mic_task,
            receive_task,
            speaker_task
        )

        try:
            await asyncio.gather(*tasks)

        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()

            await asyncio.gather(
                *tasks,
                return_exceptions=True
            )


# ============================================================
# RECONNECT LOOP
# ============================================================

async def run():
    global running

    while running:
        try:
            await run_session()

            if running:
                print(
                    "SESSION ENDED; RECONNECTING...",
                    flush=True
                )

        except asyncio.CancelledError:
            raise

        except Exception as e:
            if not running:
                break

            print(
                "GEMINI SESSION ERROR:",
                repr(e),
                flush=True
            )

            print(
                f"RECONNECTING IN {RECONNECT_DELAY} SECONDS...",
                flush=True
            )

            await asyncio.sleep(
                RECONNECT_DELAY
            )

    print(
        "Run loop stopped.",
        flush=True
    )


# ============================================================
# MAIN
# ============================================================

async def main():
    global running

    print(
        "==========================================",
        flush=True
    )

    print(
        "             CAPPY LIVE",
        flush=True
    )

    print(
        "==========================================",
        flush=True
    )

    print(
        "DIRECT 16kHz MICROPHONE",
        flush=True
    )

    print(
        "MIC SEND QUEUE",
        flush=True
    )

    print(
        "MANUAL CLIENT-SIDE VAD",
        flush=True
    )

    print(
        "EARLY 3-SECOND SER",
        flush=True
    )

    print(
        "BACKGROUND FER",
        flush=True
    )

    print(
        "MULTI-TURN GEMINI LIVE",
        flush=True
    )

    print(
        "NO CAMERA PREVIEW",
        flush=True
    )

    print(
        "==========================================",
        flush=True
    )

    print("", flush=True)

    os.makedirs(
        AUDIO_FOLDER,
        exist_ok=True
    )

    os.makedirs(
        PIC_FOLDER,
        exist_ok=True
    )

    os.makedirs(
        MELSPEC_FOLDER,
        exist_ok=True
    )

    # Startup is allowed to take time.
    # Warm models here so they do not interrupt the conversation.
    try:
        load_models()

    except Exception as e:
        print(
            "MODEL STARTUP ERROR:",
            repr(e),
            flush=True
        )

        print(
            "Continuing with available models.",
            flush=True
        )

    # LCD clock is deliberately independent of Gemini/camera.
    # A display failure therefore cannot terminate Cappy.
    start_lcd_clock()

    # Camera is deliberately outside the Gemini session.
    # A camera failure therefore cannot terminate Gemini.
    camera_background_task = asyncio.create_task(
        camera_task()
    )

    try:
        await run()

    except asyncio.CancelledError:
        pass

    except Exception as e:
        print(
            "GEMINI ERROR:",
            repr(e),
            flush=True
        )

    finally:
        running = False

        if lcd_thread is not None and lcd_thread.is_alive():
            lcd_thread.join(timeout=2.0)

        if not camera_background_task.done():
            camera_background_task.cancel()

        await asyncio.gather(
            camera_background_task,
            return_exceptions=True
        )

        print(
            "Stopped.",
            flush=True
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        running = False
        print(
            "\nStopped.",
            flush=True
        )

    except Exception as e:
        running = False
        print(
            "\nFATAL ERROR:",
            repr(e),
            flush=True
        )




