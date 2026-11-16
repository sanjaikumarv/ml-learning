import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
from pydantic import BaseModel, Field
from typing import Optional
import asyncio
import speech_recognition as sr

from openai import OpenAI, AsyncOpenAI, OpenAIError
from openai.helpers import LocalAudioPlayer

load_dotenv()

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


def get_weather(city: str):
    url = f"https://wttr.in/{city.lower()}?format=%C+%t"
    response = requests.get(url)
    if response.status_code == 200:
        return f"The weather in {city} is {response.text}"
    else:
        return f"Sorry could not retrieve the weather in {city}."


def run_command(cmd: str):
    return os.system(cmd)


available_tools = {"get_weather": get_weather, "run_command": run_command}


SYSTEM_PROMPT = """
    You're an export AI assistant in resolving user queries using chain of thoughts. You work on START , PLAN and OUTPUT steps.
    You need to first PLAN what needs to be done. The PLAN can be multiple steps.
    Once you think enough PLAN has been done, finally you can give an OUTPUT.
    You can also call a tool if required from the list of available tools.
    for every tool call wait for observe step which is the output from called tool.

    Rules:
    - Strictly Follow given JSON output format
    - Only run one step at a time.
    - The sequence os steps is START (where user gives an input), PLAN (That can be multiple times) and finally OUTPUT (which is going to be displayed to the user)

    Output JSON Format:
    {"Step": "START" | "PLAN" | "OUTPUT" | "TOOL", "content": "string", "tool": "string", "input": "string"}

    Available tools:
    - get_weather(city: str): Takes city name as input string and returns the weather info about city
    - run_command(cmd: str): Takes system command as input string and execute the command in the user system

    Example 1:
    START: Hey , Can you solve 2 + 3 * 5 / 10
    PLAN: {"step": "PLAN","content":"Seems like user is interested in math problem"}
    PLAN: {"step":"PLAN": "content":"looking at  the problem we should solve using BODMAS method"}
    PLAN: {"step": "PLAN" : "content":"Yes, The BODMAS is correct thing to be done"}
    PLAN: {"step": "PLAN" : "content": "first we must multiply 3 * 5 = 15 "}
    PLAN: {"step": "PLAN" : "content": "Now the new equation is 2+ 15 / 10"}
    PLAN: {"step": "PLAN" : "content": "We must perform divide that is 15 / 10 = 1.5 "}
    PLAN: {"step": "PLAN" : "content": "Now the new equation is 2 + 1.5 "}
    PLAN: {"step": "PLAN" : "content": "Now finally lets perform the add 3.5"}
    PLAN: {"step": "PLAN" : "content": "Great, we have solved and finally left with 3.5 as ans"}
    PLAN: {"step": "OUTPUT" : "content": "3.5"}

    Example 2:
    START: What the weather of delhi
    PLAN: {"step": "PLAN","content":"Seems like user is interested in getting weather of delhi in india"}
    PLAN: {"step":"PLAN": "content":"Let see if we have any available tool from the list of available tools"}
    PLAN: {"step": "PLAN" : "content":"Great, we have get_weather tool available for this query."}
    PLAN: {"step": "PLAN" : "content": "I need to call get_weather tool for delhi as input for city"}
    PLAN: {"step": "TOOL" : "input": "delhi" ,"tool": "get_weather"}
    PLAN: {"step": "OBSERVE" : "tool": "get_weather" , "output": "The temp of delhi is cloudy with 20 C"}
    PLAN: {"step": "PLAN" : "content": "Great, I got the weather info about delhi"}
    PLAN: {"step": "OUTPUT" : "content": "The current weather in delhi is 20 C with some cloudy sky"}
"""

print("\n\n\n")


class MyOutputFormat(BaseModel):
    step: str = Field(
        ..., description="The ID of the step. Example: PLAN, OUTPUT, TOOL, etc"
    )
    content: Optional[str] = Field(
        None, description="The optional string content for the step"
    )
    tool: Optional[str] = Field(
        None,
        description="The ID of the tool to call.",
    )
    input: Optional[str] = Field(None, description="The input params of the tool")


message_history = [{"role": "system", "content": SYSTEM_PROMPT}]


r = sr.Recognizer()

try:
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        r.pause_threshold = 2

        while True:
            print("\nSpeak something...")

            try:
                audio = r.listen(source)
                print("Processing Audio...(STT)")

                user_query = r.recognize_google(audio, language="en-IN")

                if not user_query.strip():
                    print("Please speak again.")
                    continue

                print("You said:", user_query)

            except sr.UnknownValueError:
                print("Sorry, couldn't understand your speech.")
                continue

            except sr.RequestError as e:
                print(f"Speech recognition error: {e}")
                continue

            # IMPORTANT: Add the user query to message history
            message_history.append({"role": "user", "content": user_query})

            while True:
                response = client.chat.completions.parse(
                    model="gpt-4.1-mini",
                    response_format=MyOutputFormat,
                    messages=message_history,
                )

                parsed_result = response.choices[0].message.parsed

                if parsed_result is None:
                    print("Could not parse the model response.")
                    break

                raw_result = response.choices[0].message.content

                message_history.append({"role": "assistant", "content": raw_result})

                if parsed_result.step == "START":
                    print("🔥", parsed_result.content)
                    continue

                elif parsed_result.step == "PLAN":
                    print("🚀", parsed_result.content)
                    continue

                elif parsed_result.step == "TOOL":
                    tool_to_call = parsed_result.tool
                    tool_input = parsed_result.input

                    if not tool_to_call or tool_to_call not in available_tools:
                        print(f"Invalid tool: {tool_to_call}")
                        break

                    print(f"🔥 {tool_to_call} ({tool_input})")

                    tool_response = available_tools[tool_to_call](tool_input)

                    print(f"🔥 {tool_to_call} ({tool_input}) = {tool_response}")

                    message_history.append(
                        {
                            "role": "developer",
                            "content": json.dumps(
                                {
                                    "step": "OBSERVE",
                                    "tool": tool_to_call,
                                    "input": tool_input,
                                    "output": tool_response,
                                }
                            ),
                        }
                    )

                    continue

                elif parsed_result.step == "OUTPUT":
                    output = parsed_result.content or ""

                    print("📤", output)

                    # IMPORTANT: Reuse the same event loop
                    loop.run_until_complete(tts(async_client, output))

                    break

finally:
    print("\nClosing asynchronous resources...")

    loop.run_until_complete(async_client.close())

    loop.close()
    asyncio.set_event_loop(None)

    print("Application stopped.")
