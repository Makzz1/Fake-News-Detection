
import google.generativeai as genai
import pkg_resources

try:
    version = pkg_resources.get_distribution("google-generativeai").version
    print(f"Library Version: {version}")
except Exception as e:
    print(f"Could not determine version: {e}")

GEMINI_API_KEY = "AIzaSyApfzVcknoE5JVpGaNpTrJ_WQwO7eUC4A8"
genai.configure(api_key=GEMINI_API_KEY)

print("\nAvailable Models:")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")
