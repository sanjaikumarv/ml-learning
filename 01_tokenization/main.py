import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")

text = "Hey there!, iam a sanjaikumar"

tokens = enc.encode(text)

print(tokens)

decoded = enc.decode([25216, 1354, 27942, 106460, 261, 7690, 2067, 507, 30463])
print(decoded)
