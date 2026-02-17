
import google.generativeai as genai
import os

GEMINI_API_KEY = "AIzaSyApfzVcknoE5JVpGaNpTrJ_WQwO7eUC4A8"
genai.configure(api_key=GEMINI_API_KEY)

print("Searching for working models...")
working_model = None

try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"Found model: {m.name}")
            # Prefer flash, then pro, then anything else
            if 'flash' in m.name and '1.5' in m.name:
                working_model = m.name
            elif 'pro' in m.name and '1.5' in m.name and working_model is None:
                working_model = m.name
            elif 'gemini-pro' in m.name and working_model is None:
                 working_model = m.name

    if working_model:
        print(f"Attempting to use model: {working_model}")
        try:
            # The SDK might need just the name without 'models/' prefix sometimes, or with it.
            # Usually list_models returns 'models/gemini-pro'. 
            # GenerativeModel constructor handles it, but let's try both if one fails.
            
            clean_name = working_model.replace("models/", "")
            print(f"Testing with clean name: {clean_name}")
            
            model = genai.GenerativeModel(clean_name)
            response = model.generate_content("Hello")
            print(f"Success! Response: {response.text}")
            with open("best_model.txt", "w") as f:
                f.write(clean_name)
            print(f"RECOMMENDED_MODEL_NAME={clean_name}")
        except Exception as e:
            print(f"Failed with {clean_name}: {e}")
            
            # Try with full name
            print(f"Testing with full name: {working_model}")
            try:
                model = genai.GenerativeModel(working_model)
                response = model.generate_content("Hello")
                print(f"Success! Response: {response.text}")
                with open("best_model.txt", "w") as f:
                     f.write(working_model)
                print(f"RECOMMENDED_MODEL_NAME={working_model}")
            except Exception as e2:
                print(f"Failed with full name: {e2}")
            
    else:
        print("No suitable models found.")

except Exception as e:
    print(f"Global Error: {e}")
