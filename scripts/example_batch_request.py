import time

import requests

# Define the server URL
URL = "http://localhost:8000/generate"

# Prepare the prompts
prompts = [
    "Once upon a time in a faraway land",
    "The quick brown fox jumps over the lazy dog",
    "In the year 2525, humans will",
    "A journey of a thousand miles begins with",
]

# Create the payload for the request
payload = {
    "prompts": prompts,
}


# Send the POST request to the server
start_time = time.time()
response = requests.post(URL, json=payload, timeout=5 * 60)
end_time = time.time()

# Check if the request was successful
if response.status_code == 200:
    # Extract and print the generated texts
    generated_texts = response.json().get("generated_texts", [])
    for idx, text in enumerate(generated_texts):
        print(f"Prompt {idx + 1}: {prompts[idx]}")
        print(f"Generated Text: {text}\n")
else:
    print(f"Failed to get a response. Status code: {response.status_code}")
    print(f"Response content: {response.text}")
    print(f"Response content: {response.text}")

response_time = end_time - start_time
print(f"Response time: {response_time} seconds")
