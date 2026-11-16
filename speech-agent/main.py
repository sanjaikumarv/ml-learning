from dotenv import load_dotenv
import asyncio

load_dotenv()

import speech_recognition as sr
from openai import OpenAI, AsyncOpenAI
from openai.helpers import LocalAudioPlayer

client = OpenAI()

async_client = AsyncOpenAI()


async def tts(speech: str):
    async with async_client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=speech,
        instructions="Always speak in a cheerful manner with full of delight and happy",
        response_format="pcm",
        speed=1.3,
    ) as response:
        await LocalAudioPlayer().play(response)


def main():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        r.pause_threshold = 2

        while True:
            print("Speak something....")
            audio = r.listen(source)

            print("Processing audio")

            SYSTEM_PROMPT = """
            You're an expert voice agent. You have transcript of what user said using voice.
    
            You need to output as if you are an voice agent and whatever you speak will be converted back to audio using AI and played back to user
            """

            try:
                stt = r.recognize_google(audio, language="en-IN")

                print("You said:", stt)
                response = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "user", "content": stt},
                        {"role": "system", "content": SYSTEM_PROMPT},
                    ],
                )

                print("Ai response", response.choices[0].message.content)
                asyncio.run(tts(response.choices[0].message.content))

            except sr.UnknownValueError:
                print("Could not understand audio")

            except sr.RequestError as e:
                print("Google recognition error:", e)


main()
