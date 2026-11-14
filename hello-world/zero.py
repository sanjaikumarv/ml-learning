from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

SYSTEM_PROMPT = "You should only answer coding related questions. Do not answer anything. Your name is alexa. If user asks something other than coding , just0 ,ik,./ say sorry"
response = client.chat.completions.create(
    model="gemini-3.8-flash",
    messages=[
        {"role": "system", "content": "You are expert in maths.only answer maths related question. if not just simply say i cannot answer this question"},
        {"role": "user", "content": "Hello, I am sanjaikumar , who are you"},
  
    ]
)

print(response.choices[0].message.content)




