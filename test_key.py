import os
from dotenv import load_dotenv
load_dotenv(override=True)

key = os.getenv('GOOGLE_API_KEY') or os.getenv('google_api_key')
print("Key starts with:", key[:10] if key else "None")

from google import genai
client = genai.Client(api_key=key)

models = ['gemini-2.0-flash', 'gemini-2.0-flash-lite', 'gemini-2.5-flash', 'gemini-flash-latest']
for m in models:
    try:
        res = client.models.generate_content(model=m, contents="Hello")
        print(f"[{m}] SUCCESS ->", res.text.strip())
    except Exception as e:
        print(f"[{m}] ERROR ->", e)
