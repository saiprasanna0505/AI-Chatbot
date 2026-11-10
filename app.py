import os
from flask import Flask, render_template, request, jsonify, session
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
# Replace with a secure random key in production
app.secret_key = "medical_coder_prompt_engineer_secret_key"

# Initialize GenAI Client
client = genai.Client()

SYSTEM_INSTRUCTION = """
You are an expert Prompt Engineer specializing in healthcare informatics and medical coding. 
Your ultimate duty is to solve issues, optimize queries, and clarify data for a Medical Coder.

Core competencies and behavioral guardrails:
1. When a user provides a diagnosis or procedure, provide the most likely medical codes (ICD-10-CM, CPT, or HCPCS) along with brief coding guidelines, chapters, or modifier advice if relevant.
2. When a user provides a medical code, look it up immediately and provide the complete official diagnosis or procedure description, breakdown, and inclusions/exclusions if necessary.
3. Because you are a Prompt Engineer, if the coder asks a vague question, give them the immediate answer BUT also suggest an optimized prompt they could use next time to get cleaner, more specific coding documentation results.
4. Maintain a professional, highly analytical, and clinical tone.
"""

@app.route('/')
def home():
    # Clear history on page refresh for a clean slate
    session['chat_history'] = []
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '').strip()
    if not user_message:
        return jsonify({'error': 'Message cannot be empty'}), 400

    # Retrieve or initialize chat history from session
    history = session.get('chat_history', [])

    try:
        # Reconstruct chat contents for multi-turn history manually using SDK types
        contents = []
        for msg in history:
            role = "user" if msg['role'] == "user" else "model"
            contents.append(types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg['text'])]
            ))
        
        # Append current user message
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)]
        ))

        # Build Config with System Instructions
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2, # Lower temperature for more accurate factual code matching
        )

        # CHANGED: Updated model string to resolve the 503 service / endpoint error
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=contents,
            config=config
        )

        bot_reply = response.text

        # Append current turn to session history
        history.append({'role': 'user', 'text': user_message})
        history.append({'role': 'model', 'text': bot_reply})
        session['chat_history'] = history

        return jsonify({'reply': bot_reply})

    except Exception as e:
        return jsonify({'error': f"API Error: {str(e)}"}), 500

if __name__ == '__main__':
    # Ensure your API key is available before starting
    if not os.environ.get("GEMINI_API_KEY"):
        print("Warning: GEMINI_API_KEY environment variable not detected.")
    app.run(debug=True)