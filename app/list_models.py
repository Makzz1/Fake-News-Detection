
import google.generativeai as genai
import os

# Using the key from main.py
GEMINI_API_KEY = "AIzaSyApfzVcknoE5JVpGaNpTrJ_WQwO7eUC4A8"

genai.configure(api_key=GEMINI_API_KEY)

print("Listing available models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"Name: {m.name}")
except Exception as e:
    print(f"Error: {e}")
