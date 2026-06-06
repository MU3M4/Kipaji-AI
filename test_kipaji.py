import requests
import json

payload = {
    "merchant_id": "merchant_mama_mboga_402",
    "phone_number": "+254712345678",
    "channel": "whatsapp",
    "message_body": "Leo nimeuza viazi magunia mbili KES 4500 na nyanya KES 1200. Naomba buffer line."
}

print("Sending request to Kipaji API...")
response = requests.post("http://127.0.0.1:8000/api/v1/gateway", json=payload)

print(f"Status Code: {response.status_code}")
print("Response:")
print(json.dumps(response.json(), indent=2))