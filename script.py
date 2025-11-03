# app.py
from transformers import pipeline
from PIL import Image
from io import BytesIO
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)

# Configure CORS properly for ngrok
CORS(app, resources={r"/remove-bg": {"origins": "*"}})

print("⏳ Loading RMBG-1.4 model...")
pipe = pipeline(
    "image-segmentation",
    model="briaai/RMBG-1.4",
    trust_remote_code=True
)
print("✅ Model loaded successfully!")

@app.route("/remove-bg", methods=["POST"])
def remove_bg():
    app.logger.info('Received a request to /remove-bg')

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file.content_type.startswith('image/'):
        return jsonify({"error": "File must be an image"}), 400

    try:
        image = Image.open(file.stream)

        # Ensure correct mode
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Run model — RMBG outputs an RGBA image directly
        result = pipe(image)

        if isinstance(result, Image.Image):
            output_image = result
        elif isinstance(result, dict) and "image" in result:
            output_image = result["image"]
        else:
            app.logger.error(f"Unexpected pipeline output: {type(result)}")
            return jsonify({"error": "Unexpected model output"}), 500

        # Save to buffer
        buffer = BytesIO()
        output_image.save(buffer, format="PNG", optimize=True)
        buffer.seek(0)

        app.logger.info('Successfully processed image and returning result')

        return send_file(buffer, mimetype="image/png")

    except Exception as e:
        app.logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy", 
        "model_loaded": True,
        "service": "background-removal"
    })

if __name__ == "__main__":
    # Use environment variable for port with fallback
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)  # Set debug=False for production