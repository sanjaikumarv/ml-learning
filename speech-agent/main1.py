from dotenv import load_dotenv
import asyncio

load_dotenv()

import speech_recognition as sr

from openai import OpenAI, AsyncOpenAI, OpenAIError
from openai.helpers import LocalAudioPlayer

client = OpenAI()


SYSTEM_PROMPT = """
You are an expert voice assistant.

Respond naturally and conversationally.
Your response will be converted into audio and played to the user.
"""


# TTS function
async def tts(async_client: AsyncOpenAI, speech: str):

    async with async_client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=speech,
        instructions=(
            "Always speak in a cheerful manner " "with full of delight and happiness."
        ),
        speed=1.3,
        response_format="pcm",
    ) as response:

        player = LocalAudioPlayer()

        await player.play(response)


def main():

    r = sr.Recognizer()

    r.pause_threshold = 2

    # Create ONE event loop for the entire application
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Create ONE async client for this event loop
    async_client = AsyncOpenAI()

    try:

        with sr.Microphone() as source:

            print("Calibrating microphone...")
            r.adjust_for_ambient_noise(source)

            while True:

                try:

                    print("\nSpeak something....")

                    audio = r.listen(source)

                    print("Processing audio...")

                    # Speech to text
                    stt = r.recognize_google(audio, language="en-IN")

                    print("You said:", stt)

                    # LLM request
                    response = client.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": stt},
                        ],
                    )

                    ai_response = response.choices[0].message.content

                    if not ai_response:
                        print("Empty AI response")
                        continue

                    print("AI response:", ai_response)

                    # Reuse the SAME event loop
                    loop.run_until_complete(tts(async_client, ai_response))

                except sr.UnknownValueError:
                    print("Could not understand audio")

                except sr.RequestError as e:
                    print("Speech recognition error:", e)

                except OpenAIError as e:
                    print("OpenAI API error:", e)

                except KeyboardInterrupt:
                    print("\nVoice assistant stopped.")
                    break

                except Exception as e:
                    print(f"Unexpected error: {type(e).__name__}: {e}")

    finally:

        # Close AsyncOpenAI client using the SAME event loop
        try:
            loop.run_until_complete(async_client.close())
        finally:
            loop.close()
            asyncio.set_event_loop(None)


main()
