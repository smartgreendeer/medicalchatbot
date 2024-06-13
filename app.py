from flask import Flask, render_template, jsonify, request
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.llms import CTransformers
from dotenv import load_dotenv
import os
import time

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Load data and embeddings
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

# Setup CTransformers LLM
llm = CTransformers(
    model="llama-2-7b-chat.ggmlv3.q4_0.bin", 
    model_type="llama", 
    config={'max_new_tokens': 256, 'temperature': 0.3}  # Adjusted for performance
)

# Flask routes
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        input_text = msg
        print(f"Received message: {input_text}")
        
        # Display spinner
        result = {"generated_text": "Thinking..."}
        
        # Simulate processing delay
        time.sleep(1)
        
        # Retrieve response from the model
        result = llm.generate([input_text])
        print(f"LLMResult: {result}")
        
        # Access the generated text from the result object
        if result.generations and result.generations[0]:
            generated_text = result.generations[0][0].text
        else:
            generated_text = "No response generated."
        
        print(f"Response: {generated_text}")
        
        return str(generated_text)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)