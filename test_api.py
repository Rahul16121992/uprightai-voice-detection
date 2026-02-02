import base64
import requests

print("Starting API test...")

# Read MP3 file and convert to Base64
with open("data/ai_voice/ai-4.mp3", "rb") as f:
    audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

print("Audio converted to Base64")

payload = {
    "language": "Hindi",
    "audioFormat": "mp3",
    "audioBase64": audio_b64
}

headers = {
    "x-api-key": "test_key_123",
    "Content-Type": "application/json"
}

print("Sending request to API...")

response = requests.post(
    "http://127.0.0.1:8000/voice-detection",
    json=payload,
    headers=headers
)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
