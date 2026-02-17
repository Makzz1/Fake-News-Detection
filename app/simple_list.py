
import google.generativeai as genai
import os

GEMINI_API_KEY = "AIzaSyApfzVcknoE5JVpGaNpTrJ_WQwO7eUC4A8"
genai.configure(api_key=GEMINI_API_KEY)

with open("all_models.txt", "w", encoding="utf-8") as f:
    try:
        f.write("Starting model list...\n")
        for m in genai.list_models():
            f.write(f"Model: {m.name} | Methods: {m.supported_generation_methods}\n")
        f.write("Finished listing models.\n")
    except Exception as e:
        f.write(f"Error: {e}\n")
