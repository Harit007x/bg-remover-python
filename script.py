# app.py
from transformers import pipeline
from PIL import Image
from io import BytesIO
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import base64

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

print("⏳ Loading RMBG-1.4 model...")
pipe = pipeline(
    "image-segmentation",
    model="briaai/RMBG-1.4",
    trust_remote_code=True
)
print("✅ Model loaded successfully!")

@app.route("/remove-bg", methods=["POST"])
def remove_bg():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        return jsonify({"error": "File must be an image"}), 400

    try:
        image = Image.open(file.stream)
        
        # Convert RGBA to RGB if necessary
        if image.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process image
        result = pipe(image)
        
        # The pipeline returns a list of dictionaries, we want the mask
        mask = None
        for item in result:
            if item['label'] == 'mask':
                mask = item['mask']
                break
        
        if mask is None:
            return jsonify({"error": "No mask found in result"}), 500
        
        # Convert mask to transparent background
        transparent_bg = Image.new('RGBA', image.size, (0, 0, 0, 0))
        transparent_bg.paste(image, (0, 0), mask=mask)
        
        # Save to buffer
        buffer = BytesIO()
        transparent_bg.save(buffer, format="PNG", optimize=True)
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype="image/png",
            as_attachment=False
        )

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)