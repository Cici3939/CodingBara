import asyncio
import os
import cv2
import numpy as np
from scipy.signal import resample_poly
from picamera2 import Picamera2,Preview
from google import genai
from google.genai import types

MODEL="gemini-3.1-flash-live-preview"

MIC_RATE=44100
GEMINI_RATE=16000
SPEAKER_RATE=24000

MIC_DEVICE="plughw:3,0"
SPEAKER_DEVICE="plughw:CARD=Headphones,DEV=0"

MIC_CHUNK=4410

API_KEY=os.environ.get(
    "GEMINI_API_KEY",
    "INSERT API KEY HERE"
)

client=genai.Client(api_key=API_KEY)

running=True
camera=None
camera_ready=False


class SpeakingEvent(asyncio.Event):
    pass


async def microphone(session,gemini_speaking):
    print("Microphone task started.",flush=True)

    mic=None

    try:
        print("Opening microphone...",flush=True)

        mic=await asyncio.create_subprocess_exec(
            "arecord",
            "-D",MIC_DEVICE,
            "-f","S16_LE",
            "-r",str(MIC_RATE),
            "-c","1",
            "-t","raw",
            "-",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL
        )

        print("MIC ACTIVE",flush=True)

        while running:
            data=await mic.stdout.read(MIC_CHUNK*2)

            if not data:
                print("MIC STREAM ENDED",flush=True)
                break

            # =================================================
            # GEMINI IS TALKING
            #
            # KEEP THE MICROPHONE RUNNING.
            # DO NOT SEND THIS AUDIO TO GEMINI.
            # =================================================

            if gemini_speaking.is_set():
                continue

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

            if gemini_speaking.is_set():
                continue

            try:
                await session.send_realtime_input(
                    audio=types.Blob(
                        data=pcm16.tobytes(),
                        mime_type="audio/pcm;rate=16000"
                    )
                )

            except Exception as e:
                print(
                    "MIC SEND ERROR:",
                    repr(e),
                    flush=True
                )

    except asyncio.CancelledError:
        pass

    except Exception as e:
        print(
            "MIC ERROR:",
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


async def speaker(audio_queue,gemini_speaking):
    print("Starting speaker...",flush=True)

    try:
        player=await asyncio.create_subprocess_exec(
            "aplay",
            "-D",SPEAKER_DEVICE,
            "-f","S16_LE",
            "-r",str(SPEAKER_RATE),
            "-c","1",
            "-q",
            "-",
            stdin=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL
        )

    except Exception as e:
        print(
            "SPEAKER ERROR:",
            repr(e),
            flush=True
        )
        return

    print(
        "Speaker started.",
        flush=True
    )

    try:
        while running:
            audio=await audio_queue.get()

            if audio is None:
                audio_queue.task_done()

                # All audio received from Gemini has now been
                # written to aplay.
                gemini_speaking.clear()

                print(
                    "GEMINI AUDIO FINISHED",
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

            try:
                player.stdin.write(audio)
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

        # XRGB8888 is B,G,R,X in memory.
        # OpenCV expects B,G,R, so DO NOT swap channels.
        if frame.ndim==3 and frame.shape[2]==4:
            frame=frame[:,:,:3]

        filename=os.path.abspath(
            "user_speaking.jpg"
        )

        success=cv2.imwrite(
            filename,
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95
            ]
        )

        if success:
            print(
                "PICTURE SAVED:",
                filename,
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
            "XRGB8888 CAMERA MODE",
            flush=True
        )

        print(
            "AUTOMATIC WHITE BALANCE ENABLED",
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
    gemini_speaking
):
    print(
        "Receiver started.",
        flush=True
    )

    photo_taken=False

    try:
        # IMPORTANT:
        # Keep reopening receive() on the SAME Live session.
        while running:

            print(
                "LISTENING FOR GEMINI...",
                flush=True
            )

            async for response in session.receive():

                if not running:
                    return

                content=response.server_content

                if content is None:
                    continue

                # =================================================
                # USER TRANSCRIPTION
                # =================================================

                if content.input_transcription:

                    text=content.input_transcription.text

                    if text:

                        print(
                            "HEARD:",
                            text,
                            flush=True
                        )

                        # Take ONE picture per user turn.
                        if not photo_taken:

                            photo_taken=True

                            print(
                                "USER SPEAKING DETECTED - TAKING PICTURE",
                                flush=True
                            )

                            asyncio.create_task(
                                take_picture()
                            )

                # =================================================
                # GEMINI RESPONSE
                # =================================================

                if content.model_turn:

                    for part in content.model_turn.parts:

                        if getattr(
                            part,
                            "text",
                            None
                        ):

                            print(
                                "GEMINI:",
                                part.text,
                                flush=True
                            )

                        if part.inline_data:

                            audio=part.inline_data.data

                            if audio:

                                # FIRST GEMINI AUDIO:
                                # immediately throw away
                                # microphone audio.
                                if not gemini_speaking.is_set():

                                    gemini_speaking.set()

                                    print(
                                        "GEMINI SPEAKING - THROWING AWAY MIC AUDIO",
                                        flush=True
                                    )

                                await audio_queue.put(
                                    audio
                                )

                # =================================================
                # GEMINI TURN COMPLETE
                # =================================================

                if content.turn_complete:

                    print(
                        "GEMINI TURN COMPLETE",
                        flush=True
                    )

                    photo_taken=False

                    # This comes AFTER Gemini's audio chunks.
                    await audio_queue.put(
                        None
                    )

            # google-genai can exit receive() after turn_complete.
            # Keep the SAME session alive and listen again.
            await asyncio.sleep(0)

    except asyncio.CancelledError:
        pass

    except Exception as e:

        print(
            "GEMINI RECEIVE ERROR:",
            repr(e),
            flush=True
        )


async def run():
    global running

    config=types.LiveConnectConfig(

        response_modalities=["AUDIO"],

        system_instruction="""You are Cappy, a friendly and easygoing capybara coding companion.

Help the user solve coding problems clearly and practically.

Be friendly, relaxed, encouraging, and natural for a voice conversation.

Maintain conversation context throughout the Live session.

When the user asks a coding question, help debug it step by step and provide practical working solutions.

Do not be overly formal or robotic.

You are a coding companion first, but you should also be supportive and personable.""",

        input_audio_transcription={},
        output_audio_transcription={},

        realtime_input_config=types.RealtimeInputConfig(

            automatic_activity_detection=(
                types.AutomaticActivityDetection(
                    disabled=False,
                    prefix_padding_ms=200,
                    silence_duration_ms=500,
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
            "MIC STAYS RUNNING",
            flush=True
        )

        print(
            "MIC AUDIO DISCARDED WHILE GEMINI TALKS",
            flush=True
        )

        print(
            "PI CAMERA LIVE DISPLAY",
            flush=True
        )

        print(
            "PICTURE SAVED WHEN USER SPEAKS",
            flush=True
        )

        print(
            "PICTURE NOT SENT TO GEMINI",
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

        mic_task=asyncio.create_task(
            microphone(
                session,
                gemini_speaking
            )
        )

        receive_task=asyncio.create_task(
            receive(
                session,
                audio_queue,
                gemini_speaking
            )
        )

        speaker_task=asyncio.create_task(
            speaker(
                audio_queue,
                gemini_speaking
            )
        )

        camera_task=asyncio.create_task(
            camera_display()
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

            mic_task.cancel()
            receive_task.cancel()
            speaker_task.cancel()
            camera_task.cancel()

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
        "ONE CONTINUOUS CONVERSATION",
        flush=True
    )

    print(
        "SESSION CONTEXT PRESERVED",
        flush=True
    )

    print(
        "MIC STAYS RUNNING",
        flush=True
    )

    print(
        "MIC AUDIO DISCARDED DURING GEMINI SPEECH",
        flush=True
    )

    print(
        "PI CAMERA LIVE DISPLAY",
        flush=True
    )

    print(
        "PICTURE SAVED WHEN USER SPEAKS",
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

        running=False

        print(
            "Stopped.",
            flush=True
        )


try:

    asyncio.run(main())

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
