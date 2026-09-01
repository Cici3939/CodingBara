import asyncio
import os
import sys
import cv2
import numpy as np
import wave
import time
from collections import deque
from scipy.signal import resample_poly
from picamera2 import Picamera2,Preview
from google import genai
from google.genai import types
import keras
import librosa
import torch

sys.path.insert(0,"/home/cosmos")
from CNN import SpecAugment

MODEL="gemini-3.1-flash-live-preview"

MIC_RATE=44100
GEMINI_RATE=16000
SPEAKER_RATE=24000

MIC_DEVICE="plughw:3,0"
SPEAKER_DEVICE="plughw:CARD=Headphones,DEV=0"

MIC_CHUNK=4410

AUDIO_FOLDER="/home/cosmos/CBAudio"
MELSPEC_FOLDER="/home/cosmos/MelSpec"
PIC_FOLDER="/home/cosmos/CBPic"

AUDIO_FILE=os.path.join(AUDIO_FOLDER,"user_speaking.wav")
MELSPEC_FILE=os.path.join(MELSPEC_FOLDER,"user_speaking.pt")
PIC_FILE=os.path.join(PIC_FOLDER,"user_speaking.jpg")

MODEL_PATH="/home/cosmos/Downloads/best_model.keras"

EMOTION_LABELS={
    0:"Angry",
    1:"Disgust",
    2:"Happy",
    3:"Fear",
    4:"Neutral",
    5:"Sad",
    6:"Surprise"
}

AUDIO_CLIP_SECONDS=3
AUDIO_CLIP_SAMPLES=GEMINI_RATE*AUDIO_CLIP_SECONDS
AUDIO_CLIP_BYTES=AUDIO_CLIP_SAMPLES*2

client = genai.Client(api_key="INSERT API KEY HERE")

VAD_MIN_THRESHOLD=500
VAD_NOISE_MULTIPLIER=2.5
VAD_SILENCE_SECONDS=0.8
VAD_PREROLL_CHUNKS=4

running=True
camera=None
camera_ready=False
emotion_model=None

audio_history=deque()
audio_history_size=0


class SpeakingEvent(asyncio.Event):
    pass


def add_audio_to_history(data):
    global audio_history_size

    audio_history.append(data)
    audio_history_size+=len(data)

    while audio_history_size>AUDIO_CLIP_BYTES:
        old=audio_history.popleft()
        audio_history_size-=len(old)


def get_audio_history():
    data=b"".join(audio_history)

    if len(data)>AUDIO_CLIP_BYTES:
        data=data[-AUDIO_CLIP_BYTES:]

    if len(data)<AUDIO_CLIP_BYTES:
        data=data+b"\x00"*(AUDIO_CLIP_BYTES-len(data))

    return data


def save_wave(data):
    if not data:
        print("AUDIO SAVE FAILED: NO AUDIO",flush=True)
        return False

    if len(data)<AUDIO_CLIP_BYTES:
        data=data+b"\x00"*(AUDIO_CLIP_BYTES-len(data))

    try:
        with wave.open(AUDIO_FILE,"wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(GEMINI_RATE)
            wav.writeframes(data)

        print("AUDIO CLIP SAVED:",AUDIO_FILE,flush=True)
        return True

    except Exception as e:
        print("WAVE SAVE ERROR:",repr(e),flush=True)
        return False


def prep_audio(input_file,output_file,target_duration_s=3,n_mels=64):
    os.makedirs(os.path.dirname(output_file),exist_ok=True)

    audio,sr=librosa.load(
        input_file,
        sr=22050,
        mono=True
    )

    target_samples=int(target_duration_s*sr)

    if len(audio)>target_samples:
        processed_audio=audio[:target_samples]
    elif len(audio)<target_samples:
        processed_audio=np.pad(
            audio,
            (0,target_samples-len(audio)),
            mode="constant"
        )
    else:
        processed_audio=audio

    mel_spectrogram=librosa.feature.melspectrogram(
        y=processed_audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=1024,
        hop_length=512
    )

    log_mel_spectrogram=librosa.power_to_db(
        mel_spectrogram
    )

    data=torch.tensor(
        log_mel_spectrogram,
        dtype=torch.float32
    ).numpy()

    if data.shape[1]<130:
        data=np.pad(
            data,
            ((0,0),(0,130-data.shape[1])),
            mode="constant"
        )
    elif data.shape[1]>130:
        data=data[:,:130]

    data=np.expand_dims(data,axis=-1)

    torch.save(data,output_file)

    print(
        "MEL SPECTROGRAM SAVED:",
        output_file,
        "SHAPE:",
        data.shape,
        flush=True
    )

    return data


def load_emotion_model():
    global emotion_model

    print("Loading emotion model...",flush=True)

    emotion_model=keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "SpecAugment":SpecAugment
        }
    )

    print("EMOTION MODEL LOADED",flush=True)


def predict_emotion():
    if emotion_model is None:
        print("EMOTION MODEL NOT AVAILABLE",flush=True)
        return None,None

    try:
        prep_audio(
            AUDIO_FILE,
            MELSPEC_FILE
        )

        tensor_data=torch.load(
            MELSPEC_FILE,
            weights_only=False
        )

        batch_data=np.expand_dims(
            tensor_data,
            axis=0
        )

        predictions=emotion_model.predict(
            batch_data,
            verbose=0
        )

        predicted_class=int(
            np.argmax(predictions[0])
        )

        confidence=float(
            predictions[0][predicted_class]*100
        )

        emotion=EMOTION_LABELS.get(
            predicted_class,
            "Unknown"
        )

        print("",flush=True)
        print("--- SER RESULT ---",flush=True)
        print("Emotion:",emotion,flush=True)
        print(
            f"Confidence: {confidence:.2f}%",
            flush=True
        )
        print("",flush=True)

        return emotion,confidence

    except Exception as e:
        print(
            "EMOTION PREDICTION ERROR:",
            repr(e),
            flush=True
        )
        return None,None


async def send_emotion_to_cappy(
    session,
    emotion,
    confidence
):
    if emotion is None:
        emotion="Unknown"

    if confidence is None:
        confidence=0.0

    message=(
        "[INTERNAL SER EMOTION DATA]\n"
        f"Detected emotion: {emotion}\n"
        f"Confidence: {confidence:.2f}%\n\n"
        "Use this information to subtly adapt your "
        "response to the user's current message.\n"
        "High confidence means the signal can be relied "
        "on more strongly.\n"
        "Low confidence means it should be treated only "
        "as a weak clue.\n"
        "Do not mention the SER system, emotion classifier, "
        "or confidence score to the user unless the user "
        "specifically asks about it.\n"
        "Do not assume the detected emotion is definitely "
        "correct. Prioritize what the user actually says."
    )

    try:
        await session.send_realtime_input(
            text=message
        )

        print(
            "SER SENT TO CAPPY:",
            emotion,
            f"{confidence:.2f}%",
            flush=True
        )

        return True

    except Exception as e:
        print(
            "SER SEND ERROR:",
            repr(e),
            flush=True
        )
        return False


async def process_user_turn(
    session,
    audio_data
):
    try:
        print(
            "PROCESSING USER AUDIO...",
            flush=True
        )

        if not save_wave(audio_data):
            return

        emotion,confidence=await asyncio.to_thread(
            predict_emotion
        )

        await send_emotion_to_cappy(
            session,
            emotion,
            confidence
        )

    except Exception as e:
        print(
            "USER TURN PROCESSING ERROR:",
            repr(e),
            flush=True
        )


async def take_picture():
    global camera,camera_ready

    if camera is None or not camera_ready:
        print(
            "CAMERA NOT READY",
            flush=True
        )
        return

    try:
        frame=camera.capture_array()

        if frame is None:
            print(
                "CAMERA RETURNED NO FRAME",
                flush=True
            )
            return

        if frame.ndim==3 and frame.shape[2]==4:
            frame=frame[:,:,:3]

        success=cv2.imwrite(
            PIC_FILE,
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95
            ]
        )

        if success:
            print(
                "PICTURE SAVED:",
                PIC_FILE,
                flush=True
            )
        else:
            print(
                "PICTURE SAVE FAILED",
                flush=True
            )

    except Exception as e:
        print(
            "PICTURE ERROR:",
            repr(e),
            flush=True
        )


async def microphone(
    session,
    gemini_speaking,
    turn_busy
):
    print(
        "Microphone task started.",
        flush=True
    )

    mic=None
    speech_active=False
    last_voice_time=0.0
    noise_floor=300.0

    preroll=deque(
        maxlen=VAD_PREROLL_CHUNKS
    )

    try:
        print(
            "Opening microphone...",
            flush=True
        )

        mic=await asyncio.create_subprocess_exec(
            "arecord",
            "-D",
            MIC_DEVICE,
            "-f",
            "S16_LE",
            "-r",
            str(MIC_RATE),
            "-c",
            "1",
            "-t",
            "raw",
            "-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL
        )

        print(
            "MIC ACTIVE",
            flush=True
        )

        while running:

            data=await mic.stdout.read(
                MIC_CHUNK*2
            )

            if not data:
                print(
                    "MIC STREAM ENDED",
                    flush=True
                )
                break

            pcm=np.frombuffer(
                data,
                dtype=np.int16
            )

            if len(pcm)==0:
                continue

            pcm16=resample_poly(
                pcm,
                GEMINI_RATE,
                MIC_RATE
            ).astype(np.int16)

            if len(pcm16)==0:
                continue

            audio_bytes=pcm16.tobytes()

            add_audio_to_history(
                audio_bytes
            )

            rms=float(
                np.sqrt(
                    np.mean(
                        pcm16.astype(
                            np.float32
                        )**2
                    )
                )
            )

            if not speech_active and not turn_busy.is_set():
                if rms<noise_floor*2.0:
                    noise_floor=(
                        noise_floor*0.95+
                        rms*0.05
                    )

            threshold=max(
                VAD_MIN_THRESHOLD,
                noise_floor*VAD_NOISE_MULTIPLIER
            )

            voice_detected=(
                rms>threshold
            )

            # CAPPY IS SPEAKING:
            # keep microphone hardware running,
            # but throw away everything.
            if gemini_speaking.is_set():

                preroll.clear()
                speech_active=False

                continue

            # IMPORTANT:
            # If we are already inside the user's speech turn,
            # DO NOT use turn_busy to discard the audio.
            #
            # turn_busy stays TRUE during the entire user turn
            # and while waiting for Cappy's response.
            #
            # Therefore this check must only happen when
            # speech_active is FALSE.
            if turn_busy.is_set() and not speech_active:

                preroll.clear()

                continue

            # ------------------------------------------
            # WAITING FOR USER TO START SPEAKING
            # ------------------------------------------

            if not speech_active:

                preroll.append(
                    audio_bytes
                )

                if voice_detected:

                    speech_active=True
                    last_voice_time=time.monotonic()

                    turn_busy.set()

                    print(
                        "",
                        flush=True
                    )

                    print(
                        "USER SPEECH DETECTED",
                        flush=True
                    )

                    print(
                        f"RMS: {rms:.0f}  "
                        f"Threshold: {threshold:.0f}",
                        flush=True
                    )

                    print(
                        "STARTING GEMINI USER TURN",
                        flush=True
                    )

                    try:

                        await session.send_realtime_input(
                            activity_start=types.ActivityStart()
                        )

                        print(
                            "ACTIVITY START SENT",
                            flush=True
                        )

                        # Send pre-roll audio so the beginning
                        # of the user's speech isn't clipped.
                        for old_audio in preroll:

                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=old_audio,
                                    mime_type="audio/pcm;rate=16000"
                                )
                            )

                        preroll.clear()

                        # Send current speech chunk.
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=audio_bytes,
                                mime_type="audio/pcm;rate=16000"
                            )
                        )

                        asyncio.create_task(
                            take_picture()
                        )

                    except Exception as e:

                        print(
                            "GEMINI USER TURN START ERROR:",
                            repr(e),
                            flush=True
                        )

                        speech_active=False
                        turn_busy.clear()

            # ------------------------------------------
            # USER IS CURRENTLY SPEAKING
            # ------------------------------------------

            else:

                try:

                    # KEEP SENDING AUDIO WHILE USER SPEAKS.
                    await session.send_realtime_input(
                        audio=types.Blob(
                            data=audio_bytes,
                            mime_type="audio/pcm;rate=16000"
                        )
                    )

                except Exception as e:

                    print(
                        "MIC SEND ERROR:",
                        repr(e),
                        flush=True
                    )

                    speech_active=False
                    turn_busy.clear()

                    break

                if voice_detected:

                    last_voice_time=time.monotonic()

                elif (
                    time.monotonic()-last_voice_time
                    >=VAD_SILENCE_SECONDS
                ):

                    speech_active=False

                    print(
                        "USER STOPPED SPEAKING",
                        flush=True
                    )

                    print(
                        "WAITING FOR SER...",
                        flush=True
                    )

                    audio_data=get_audio_history()

                    # SER MUST FINISH BEFORE WE END
                    # THE GEMINI USER ACTIVITY.
                    await process_user_turn(
                        session,
                        audio_data
                    )

                    print(
                        "SER COMPLETE",
                        flush=True
                    )

                    try:

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

                    except Exception as e:

                        print(
                            "ACTIVITY END ERROR:",
                            repr(e),
                            flush=True
                        )

                        turn_busy.clear()

    except asyncio.CancelledError:
        pass

    except Exception as e:

        print(
            "MICROPHONE ERROR:",
            repr(e),
            flush=True
        )

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


async def speaker(
    audio_queue,
    gemini_speaking,
    turn_busy
):
    print(
        "Starting speaker...",
        flush=True
    )

    player=None

    try:

        player=await asyncio.create_subprocess_exec(
            "aplay",
            "-D",
            SPEAKER_DEVICE,
            "-f",
            "S16_LE",
            "-r",
            str(SPEAKER_RATE),
            "-c",
            "1",
            "-q",
            "-",
            stdin=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL
        )

        print(
            "Speaker started.",
            flush=True
        )

        while running:

            audio=await audio_queue.get()

            try:

                if audio is None:

                    gemini_speaking.clear()
                    turn_busy.clear()

                    print(
                        "",
                        flush=True
                    )

                    print(
                        "CAPPY FINISHED SPEAKING",
                        flush=True
                    )

                    print(
                        "MIC AUDIO ACTIVE",
                        flush=True
                    )

                    print(
                        "READY FOR USER",
                        flush=True
                    )

                    continue

                if player.stdin is not None:

                    player.stdin.write(
                        audio
                    )

                    await player.stdin.drain()

            except(
                BrokenPipeError,
                ConnectionResetError
            ):
                break

            finally:
                audio_queue.task_done()

    except asyncio.CancelledError:
        pass

    except Exception as e:

        print(
            "SPEAKER ERROR:",
            repr(e),
            flush=True
        )

    finally:

        gemini_speaking.clear()
        turn_busy.clear()

        if player is not None:

            try:
                player.stdin.close()
            except Exception:
                pass

            try:
                await player.wait()
            except Exception:
                pass

        print(
            "Speaker closed.",
            flush=True
        )


async def camera_display():
    global camera,camera_ready

    print(
        "Starting Pi Camera...",
        flush=True
    )

    try:

        camera=Picamera2()

        config=camera.create_preview_configuration(
            main={
                "size":(640,480),
                "format":"XRGB8888"
            }
        )

        camera.configure(config)

        camera.start_preview(
            Preview.QTGL
        )

        camera.start()

        await asyncio.sleep(2)

        try:

            camera.set_controls({
                "AwbEnable":True
            })

        except Exception as e:

            print(
                "WHITE BALANCE CONTROL:",
                repr(e),
                flush=True
            )

        camera_ready=True

        print(
            "PI CAMERA ACTIVE",
            flush=True
        )

        print(
            "CAMERA READY",
            flush=True
        )

        while running:
            await asyncio.sleep(0.1)

    except asyncio.CancelledError:
        pass

    except Exception as e:

        print(
            "CAMERA ERROR:",
            repr(e),
            flush=True
        )

    finally:

        camera_ready=False

        if camera is not None:

            try:
                camera.stop_preview()
            except Exception:
                pass

            try:
                camera.stop()
            except Exception:
                pass

            try:
                camera.close()
            except Exception:
                pass

            camera=None

        print(
            "Camera closed.",
            flush=True
        )


async def receive(
    session,
    audio_queue,
    gemini_speaking,
    turn_busy
):
    print(
        "Receiver started.",
        flush=True
    )

    try:

        while running:

            async for response in session.receive():

                if not running:
                    return

                content=response.server_content

                if content is None:
                    continue

                if content.input_transcription:

                    text=content.input_transcription.text

                    if text:

                        print(
                            "HEARD:",
                            text,
                            flush=True
                        )

                if content.model_turn:

                    for part in content.model_turn.parts:

                        text=getattr(
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

                        inline_data=getattr(
                            part,
                            "inline_data",
                            None
                        )

                        if inline_data:

                            audio=inline_data.data

                            if audio:

                                if not gemini_speaking.is_set():

                                    gemini_speaking.set()

                                    print(
                                        "",
                                        flush=True
                                    )

                                    print(
                                        "CAPPY SPEAKING",
                                        flush=True
                                    )

                                    print(
                                        "THROWING AWAY MIC AUDIO",
                                        flush=True
                                    )

                                await audio_queue.put(
                                    audio
                                )

                if content.turn_complete:

                    print(
                        "CAPPY TURN COMPLETE",
                        flush=True
                    )

                    await audio_queue.put(
                        None
                    )

            await asyncio.sleep(0)

    except asyncio.CancelledError:
        pass

    except Exception as e:

        print(
            "",
            flush=True
        )

        print(
            "GEMINI RECEIVE ERROR:",
            repr(e),
            flush=True
        )

        gemini_speaking.clear()
        turn_busy.clear()


async def run():
    global running

    config=types.LiveConnectConfig(

        response_modalities=[
            "AUDIO"
        ],

        system_instruction="""You are Cappy, a friendly and easygoing capybara coding companion.

Help the user solve coding problems clearly and practically.

Be friendly, relaxed, encouraging, and natural for a voice conversation.

Maintain conversation context throughout the Live session.

When the user asks a coding question, help debug it step by step and provide practical working solutions.

Do not be overly formal or robotic.

IMPORTANT EMOTION BEHAVIOR:

The program may provide you with speech emotion recognition information before you respond.

It will look like:

[INTERNAL SER EMOTION DATA]
Detected emotion: Happy
Confidence: 87.00%

Use the detected emotion and confidence to subtly adapt your response.

High confidence:
Rely more strongly on the emotion signal.

Medium confidence:
Use the emotion as moderate contextual guidance.

Low confidence:
Treat the emotion as a weak clue and prioritize the user's actual words.

Examples:

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

Never mention SER, the neural network, the classifier, or the confidence score unless the user asks.

The emotion information is contextual guidance only.

Always prioritize what the user actually says over the emotion prediction.""",

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
            "SER BEFORE GEMINI RESPONSE ENABLED",
            flush=True
        )

        print(
            "MIC STAYS RUNNING",
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

        audio_queue=asyncio.Queue(
            maxsize=100
        )

        gemini_speaking=SpeakingEvent()
        turn_busy=SpeakingEvent()

        camera_task=asyncio.create_task(
            camera_display()
        )

        await asyncio.sleep(2)

        mic_task=asyncio.create_task(
            microphone(
                session,
                gemini_speaking,
                turn_busy
            )
        )

        receive_task=asyncio.create_task(
            receive(
                session,
                audio_queue,
                gemini_speaking,
                turn_busy
            )
        )

        speaker_task=asyncio.create_task(
            speaker(
                audio_queue,
                gemini_speaking,
                turn_busy
            )
        )

        try:

            await asyncio.gather(
                mic_task,
                receive_task,
                speaker_task,
                camera_task
            )

        except asyncio.CancelledError:
            pass

        finally:

            for task in (
                mic_task,
                receive_task,
                speaker_task,
                camera_task
            ):
                task.cancel()

            await asyncio.gather(
                mic_task,
                receive_task,
                speaker_task,
                camera_task,
                return_exceptions=True
            )


async def main():
    global running

    print(
        "==========================================",
        flush=True
    )

    print(
        "       GEMINI LIVE RASPBERRY PI",
        flush=True
    )

    print(
        "==========================================",
        flush=True
    )

    print(
        "CONTINUOUS GEMINI CONVERSATION",
        flush=True
    )

    print(
        "MANUAL CLIENT-SIDE VAD",
        flush=True
    )

    print(
        "SER EMOTION + CONFIDENCE",
        flush=True
    )

    print(
        "SER SENT TO CAPPY BEFORE RESPONSE",
        flush=True
    )

    print(
        "MIC STAYS RUNNING",
        flush=True
    )

    print(
        "CAPPY AUDIO DOES NOT INTERRUPT",
        flush=True
    )

    print(
        "PI CAMERA LIVE DISPLAY",
        flush=True
    )

    print(
        "==========================================",
        flush=True
    )

    print()

    os.makedirs(
        AUDIO_FOLDER,
        exist_ok=True
    )

    os.makedirs(
        MELSPEC_FOLDER,
        exist_ok=True
    )

    os.makedirs(
        PIC_FOLDER,
        exist_ok=True
    )

    try:

        load_emotion_model()

    except Exception as e:

        print(
            "EMOTION MODEL FAILED TO LOAD:",
            repr(e),
            flush=True
        )

        print(
            "Continuing without emotion classification.",
            flush=True
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

        running=False

        print(
            "Stopped.",
            flush=True
        )


try:

    asyncio.run(
        main()
    )

except KeyboardInterrupt:

    running=False

    print(
        "\nStopped.",
        flush=True
    )

except Exception as e:

    running=False

    print(
        "\nFATAL ERROR:",
        repr(e),
        flush=True
    )
