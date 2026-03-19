# Run this as a standalone test — NOT in the app
import requests, base64, cv2
import numpy as np

GEMINI_API_KEY = "AIzaSyCcE4T3HVNbPL18aMPCas_gAmV7LHsWCYg"

# Simulate a plate crop
img = np.ones((100, 400, 3), dtype=np.uint8) * 255
cv2.putText(img, "AP37DD7042", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 3)

_, buf = cv2.imencode('.jpg', img)
b64 = base64.b64encode(buf).decode('utf-8')

for model in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]:
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}",
        json={"contents": [{"parts": [
            {"text": "Read the vehicle number plate. Return ONLY the plate number."},
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}}
        ]}]},
        timeout=15
    )
    print(f"\n{model}: status={resp.status_code}")
    print(resp.json())