import os
import base64
import mimetypes
import json
import requests
from flask import Flask, render_template, request, jsonify, send_from_directory
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("CRITICAL ERROR: GEMINI_API_KEY environment variable is not set. Please ensure the .env file exists and contains a valid API key.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/process-document', methods=['POST'])
def process_document():
    if 'document' not in request.files:
        return jsonify({'error': 'No document file uploaded.'}), 400

    file = request.files['document']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        mime_type = "image/jpeg"

    # Encode Image to Base64
    with open(file_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")

    # Call Gemini REST API
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}

    prompt = (
        "You are an expert Document Processing OCR Engine. "
        "Extract key information from this document in valid raw JSON format ONLY without markdown backticks.\n"
        "Formatting Rules:\n"
        "- Convert all extracted names and addresses into proper Title Case format (e.g., convert 'RANA' or 'Rana' to 'Rana', 'GULBERG' to 'Gulberg').\n"
        "Keys to extract:\n"
        "- Name\n"
        "- Age\n"
        "- Address\n"
        "- Phone_Number\n"
        "- Document_Type\n"
        "- Raw_Extracted_Text\n"
        "If a specific field (like Name, Age, Address, Phone_Number) is NOT present in the document, set its value to empty string ''."
    )

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64_image
                    }
                }
            ]
        }]
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            raw_text = result['candidates'][0]['content']['parts'][0]['text']
            
            # Clean JSON formatting wrappers
            cleaned_text = raw_text.replace("```json", "").replace("```", "").strip()
            
            try:
                # Parse raw string to JSON
                parsed_json = json.loads(cleaned_text)
                
                if isinstance(parsed_json, dict):
                    for key, val in parsed_json.items():
                        if val == "N/A":
                            parsed_json[key] = ""
                        # Convert String values like Name, Address to Title Case (excluding Raw_Extracted_Text)
                        elif isinstance(val, str) and key in ["Name", "Address"]:
                            parsed_json[key] = val.title()
                            
            except Exception:
                parsed_json = {"Raw_Extracted_Text": cleaned_text}

            return jsonify({
                'status': 'success',
                'image_url': f'/uploads/{file.filename}',
                'extracted_data': parsed_json
            })
        else:
            return jsonify({'error': f"API Error {response.status_code}: {response.text}"}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/save-document', methods=['POST'])
def save_document():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No payload provided'}), 400

        records_file = os.path.join(BASE_DIR, 'saved_records.json')
        
        # Load existing collection
        existing_data = []
        if os.path.exists(records_file):
            with open(records_file, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                except Exception:
                    existing_data = []

        # Document record structure prioritizing Raw Extracted Text
        record = {
            "raw_extracted_text": data.get("Raw_Extracted_Text", ""),
            "metadata": {
                "name": data.get("Name", ""),
                "age": data.get("Age", ""),
                "address": data.get("Address", ""),
                "phone_number": data.get("Phone_Number", ""),
                "document_type": data.get("Document_Type", "")
            }
        }

        existing_data.append(record)

        # Append to JSON storage file
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)

        return jsonify({'status': 'success', 'message': 'Raw text & parsed metadata saved successfully!'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask Development Server at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)