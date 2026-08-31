import asyncio
import os
import numpy as np
from scipy.signal import resample_poly
from google import genai
from google.genai import types

MODEL = "gemini-3.1-flash-live-preview"

MIC_RATE = 44100
GEMINI_RATE = 16000
SPEAKER_RATE = 24000

MIC_DEVICE = "plughw:3,0"
SPEAKER_DEVICE = "plughw:CARD=Headphones,DEV=0"

MIC_CHUNK = 4410

API_KEY = os.environ.get(
    "GEMINI_API_KEY",
    "INSERT_API_KEY_HERE"
)

client = genai.Client(api_key=API_KEY)

running = True


# ============================================================
# MICROPHONE
#
# arecord is COMPLETELY STOPPED while Gemini talks.
# A NEW arecord process is created after Gemini finishes.
# ============================================================

async def microphone(session, gemini_speaking):

    print("Microphone task started.", flush=True)

    while running:

        mic = None

        try:
            # Wait until Gemini is NOT speaking.
            await gemini_speaking.wait_clear()

            if not running:
                break

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
                stderr=asyncio.subprocess.PIPE
            )

            print("MIC ACTIVE", flush=True)

            while running and not gemini_speaking.is_set():

                data = await mic.stdout.read(MIC_CHUNK * 2)

                if not data:
                    break

                # Gemini started talking while arecord was reading.
                # THROW AWAY this chunk.
                if gemini_speaking.is_set():
                    break

                pcm = np.frombuffer(
                    data,
                    dtype=np.int16
                )

                if len(pcm) == 0:
                    continue

                pcm16 = resample_poly(
                    pcm,
                    GEMINI_RATE,
                    MIC_RATE
                ).astype(np.int16)

                await session.send_realtime_input(
                    audio=types.Blob(
                        data=pcm16.tobytes(),
                        mime_type="audio/pcm;rate=16000"
                    )
                )

        except asyncio.CancelledError:
            break

        except Exception as e:
            print(
                "MIC ERROR:",
                repr(e),
                flush=True
            )

        finally:

            # ALWAYS kill arecord when Gemini starts talking.
            if mic is not None:

                try:
                    mic.terminate()
                except Exception:
                    pass

                try:
                    await mic.wait()
                except Exception:
                    pass

            if running and gemini_speaking.is_set():

                print(
                    "MIC STOPPED - GEMINI TALKING",
                    flush=True
                )

                # Wait until Gemini has completely finished.
                await gemini_speaking.wait_clear()

                if running:
                    print(
                        "MIC RESTARTING",
                        flush=True
                    )

            elif running:

                # Small yield so we don't spin if arecord exits.
                await asyncio.sleep(0.05)

    print("Microphone closed.", flush=True)


# ============================================================
# EVENT HELPER
#
# asyncio.Event has wait(), but no wait_clear().
# ============================================================

class SpeakingEvent(asyncio.Event):

    async def wait_clear(self):

        while self.is_set() and running:
            await asyncio.sleep(0.01)


# ============================================================
# SPEAKER
# ============================================================

async def speaker(audio_queue, gemini_speaking):

    print("Starting speaker...", flush=True)

    try:
        player = await asyncio.create_subprocess_exec(
            "aplay",
            "-D", SPEAKER_DEVICE,
            "-f", "S16_LE",
            "-r", str(SPEAKER_RATE),
            "-c", "1",
            "-q",
            "-",
            stdin=asyncio.subprocess.PIPE
        )

    except Exception as e:
        print(
            "SPEAKER ERROR:",
            repr(e),
            flush=True
        )
        return

    print("Speaker started.", flush=True)

    try:

        while running:

            audio = await audio_queue.get()

            if audio is None:

                audio_queue.task_done()

                # Gemini has finished sending audio.
                # Since this marker comes AFTER every audio chunk,
                # clearing here means ALL Gemini audio has already
                # been written to aplay.
                gemini_speaking.clear()

                print(
                    "GEMINI AUDIO FINISHED",
                    flush=True
                )

                print(
                    "READY FOR USER",
                    flush=True
                )

                continue

            try:

                player.stdin.write(audio)

                await player.stdin.drain()

            except (
                BrokenPipeError,
                ConnectionResetError
            ):

                break

            finally:

                audio_queue.task_done()

    except asyncio.CancelledError:
        pass

    finally:

        gemini_speaking.clear()

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


# ============================================================
# GEMINI RECEIVER
#
# google-genai 2.20.0 ends receive() at turn_complete.
# Therefore we reopen receive() without closing the session.
# ============================================================

async def receive(
    session,
    audio_queue,
    gemini_speaking
):

    print(
        "Receiver started.",
        flush=True
    )

    try:

        while running:

            print(
                "LISTENING FOR GEMINI...",
                flush=True
            )

            async for response in session.receive():

                if not running:
                    return

                content = response.server_content

                if content is None:
                    continue

                # --------------------------------------------
                # USER TRANSCRIPTION
                # --------------------------------------------

                if content.input_transcription:

                    text = content.input_transcription.text

                    if text:

                        print(
                            "HEARD:",
                            text,
                            flush=True
                        )

                # --------------------------------------------
                # GEMINI TURN
                # --------------------------------------------

                if content.model_turn:

                    for part in content.model_turn.parts:

                        if getattr(part, "text", None):

                            print(
                                "GEMINI:",
                                part.text,
                                flush=True
                            )

                        if part.inline_data:

                            audio = part.inline_data.data

                            if audio:

                                # FIRST AUDIO FROM GEMINI:
                                #
                                # Immediately stop microphone.
                                if not gemini_speaking.is_set():

                                    gemini_speaking.set()

                                    print(
                                        "STOPPING MICROPHONE",
                                        flush=True
                                    )

                                await audio_queue.put(audio)

                # --------------------------------------------
                # TURN COMPLETE
                # --------------------------------------------

                if content.turn_complete:

                    print(
                        "GEMINI TURN COMPLETE",
                        flush=True
                    )

                    # This marker goes AFTER every audio chunk.
                    #
                    # The speaker will clear gemini_speaking
                    # only after reaching this marker.
                    await audio_queue.put(None)

            # SDK 2.20.0 exits receive() here.
            # Keep the SAME Live session and reopen receiver.

            await asyncio.sleep(0)

    except asyncio.CancelledError:
        pass

    except Exception as e:

        print(
            "GEMINI RECEIVE ERROR:",
            repr(e),
            flush=True
        )


# ============================================================
# RUN
# ============================================================

async def run():

    global running

    config = types.LiveConnectConfig(

        response_modalities=["AUDIO"],

        input_audio_transcription={},

        output_audio_transcription={},

        realtime_input_config=types.RealtimeInputConfig(

            automatic_activity_detection=(
                types.AutomaticActivityDetection(
                    disabled=False,
                    prefix_padding_ms=200,
                    silence_duration_ms=700,
                    start_of_speech_sensitivity=(
                        "START_SENSITIVITY_HIGH"
                    ),
                    end_of_speech_sensitivity=(
                        "END_SENSITIVITY_HIGH"
                    )
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
            "ONE CONTINUOUS SESSION",
            flush=True
        )

        print(
            "SESSION HISTORY ENABLED",
            flush=True
        )

        print(
            "MIC WILL COMPLETELY STOP DURING GEMINI SPEECH",
            flush=True
        )

        print(
            "READY FOR USER",
            flush=True
        )

        audio_queue = asyncio.Queue(
            maxsize=100
        )

        gemini_speaking = SpeakingEvent()

        mic_task = asyncio.create_task(
            microphone(
                session,
                gemini_speaking
            )
        )

        receive_task = asyncio.create_task(
            receive(
                session,
                audio_queue,
                gemini_speaking
            )
        )

        speaker_task = asyncio.create_task(
            speaker(
                audio_queue,
                gemini_speaking
            )
        )

        try:

            await asyncio.gather(
                mic_task,
                receive_task,
                speaker_task
            )

        except asyncio.CancelledError:
            pass

        finally:

            mic_task.cancel()
            receive_task.cancel()
            speaker_task.cancel()

            await asyncio.gather(
                mic_task,
                receive_task,
                speaker_task,
                return_exceptions=True
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
        "       GEMINI LIVE RASPBERRY PI",
        flush=True
    )

    print(
        "==========================================",
        flush=True
    )

    print(
        "ONE CONTINUOUS CONVERSATION",
        flush=True
    )

    print(
        "SESSION CONTEXT PRESERVED",
        flush=True
    )

    print(
        "MIC PROCESS STOPS DURING GEMINI SPEECH",
        flush=True
    )

    print()

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

        print(
            "Stopped.",
            flush=True
        )


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
