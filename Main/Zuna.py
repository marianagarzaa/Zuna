from transformers import pipeline
from flask import Flask, request, jsonify

# Initialize the model pipeline
# 'gpt2' is a lightweight model, but you can choose other models from Hugging Face if needed
zuna_model = pipeline("text-generation", model="gpt2")

# Flask app for hosting Zuna
app = Flask(__name__)

# Personality and tone for Zuna
ZUNA_PROMPT = (
    "You are Zuna, a funny, sarcastic AI agent who educates people about AI and crypto. "
    "You love memes and dislike uninformed opinions. Keep responses light-hearted and engaging."
)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Generate a response using the model
    full_prompt = f"{ZUNA_PROMPT}\n\nUser: {user_input}\nZuna:"
    response = zuna_model(full_prompt, max_length=100, num_return_sequences=1, pad_token_id=50256)

    # Return the generated response
    zuna_reply = response[0]["generated_text"].split("Zuna:")[1].strip()
    return jsonify({"response": zuna_reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
