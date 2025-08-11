import os
import json
import uuid
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import numpy as np
import io
from base64 import b64decode

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# UPDATED: Add your Cloudflare domain to the allowed origins.
# You may need to update this with your specific domain.
CORS(app, resources={
    r"/predict": {
        "origins": [
            "https://neural-pcb-project.vercel.app", 
            "http://localhost:3000", 
            "http://127.0.0.1:3000",
            "https://appropriate-accuracy-suffering-d.trycloudflare.com",
            "https://sensitive-delivers-peas-research.trycloudflare.com"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    },
    r"/health": {
        "origins": [
            "https://neural-pcb-project.vercel.app", 
            "http://localhost:3000", 
            "http://127.0.0.1:3000",
            "https://appropriate-accuracy-suffering-d.trycloudflare.com",
            "https://sensitive-delivers-peas-research.trycloudflare.com"
        ],
        "methods": ["GET", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Define the global model and class names
model = None
CLASS_NAMES = []
# Define the device for model inference
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """Loads the YOLOv5 model from disk."""
    global model, CLASS_NAMES
    model_path = os.getenv('MODEL_PATH', 'best.pt')
    logger.info(f"Attempting to load model from {model_path} on device {DEVICE}...")
    try:
        # NOTE: Using a simple reload here to ensure the latest changes are picked up.
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True, device=DEVICE)
        model.eval()
        CLASS_NAMES = model.names
        logger.info(f"Model loaded successfully. Found {len(CLASS_NAMES)} classes.")
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        model = None
        CLASS_NAMES = []

def process_single_image(image_stream, frontend_image_id, frontend_name):
    """
    Processes a single image stream for object detection.
    Args:
        image_stream: An image file object or base64 string.
        frontend_image_id: Unique ID for the image from the frontend.
        frontend_name: The original name of the image file.
    Returns:
        A dictionary with detection results or an error message.
    """
    if model is None:
        return {"image_id": frontend_image_id, "error": "Model is not loaded."}
    
    pil_image = None
    try:
        if isinstance(image_stream, str): # Handle base64 encoded images
            base64_data = image_stream.split(',')[1]
            image_stream = io.BytesIO(b64decode(base64_data))

        # Load the image using PIL
        pil_image = Image.open(image_stream).convert('RGB')
        original_width, original_height = pil_image.size
        
        # Check for valid image dimensions
        if original_width == 0 or original_height == 0:
            return {"image_id": frontend_image_id, "error": "Processing failed: Image dimensions are zero."}

        img_np = np.array(pil_image)
        
        # Perform prediction
        results = model(img_np, size=original_width)
        
        # Extract predictions, ensuring it's a list even if no detections are found
        predictions_list = []
        # The .pred attribute is a list of tensors for each image in the batch.
        # We access the first item [0] for our single image.
        if results.pred and len(results.pred[0]):
            for *box, conf, cls in results.pred[0]:
                predictions_list.append({
                    "box": [float(b) for b in box],
                    "confidence": float(conf),
                    "class": CLASS_NAMES[int(cls)]
                })

        result = {
            "image_id": frontend_image_id,
            "predictions": predictions_list,
            "image_dimensions": {
                "width": original_width,
                "height": original_height
            },
            "total_detections": len(predictions_list)
        }
        return result

    except Exception as e:
        # Return a result with zero dimensions if an error occurs during processing
        logger.error(f"Error processing image {frontend_name} (ID: {frontend_image_id}): {e}", exc_info=True)
        return {
            "image_id": frontend_image_id,
            "predictions": [],
            "image_dimensions": { "width": 0, "height": 0 },
            "total_detections": 0,
            "error": f"Processing failed: {str(e)}"
        }

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
    
    if model is None:
        logger.error("Predict endpoint called but model is not loaded.")
        return jsonify({"error": "Model not loaded. Please check server logs."}), 500

    results = []
    total_defects = 0
    processing_errors = []

    logger.info("--- Starting prediction request ---")
    
    try:
        # JSON body (base64 images)
        if request.is_json and 'images_data' in request.json:
            images_data = request.json['images_data']
            logger.info(f"Received {len(images_data)} images from JSON data.")
            for img_data in images_data:
                frontend_id = img_data.get('id', f"json_{uuid.uuid4().hex[:8]}")
                image_src = img_data.get('src')
                image_name = img_data.get('name', "JSON_Image")
                if image_src:
                    result = process_single_image(image_src, frontend_id, image_name)
                    results.append(result)
                    if 'error' not in result:
                        total_defects += result['total_detections']
                    else:
                        processing_errors.append(f"Image '{image_name}' (ID: {frontend_id}): {result['error']}")
                else:
                    processing_errors.append(f"Image '{image_name}' (ID: {frontend_id}): Missing image source data.")

        # Multiple files (FormData)
        elif 'images' in request.files:
            files = request.files.getlist('images')
            image_metadata_json = request.form.get('image_metadata')
            image_metadata = []
            if image_metadata_json:
                try:
                    image_metadata = json.loads(image_metadata_json)
                except json.JSONDecodeError:
                    logger.warning("Could not decode image_metadata JSON from frontend.")

            logger.info(f"Received {len(files)} uploaded files via FormData.")
            
            for i, file in enumerate(files):
                if file and file.filename:
                    frontend_id = f"upload_{uuid.uuid4().hex[:8]}"
                    frontend_name = file.filename
                    
                    if i < len(image_metadata):
                        meta = image_metadata[i]
                        frontend_id = meta.get('id', frontend_id)
                        frontend_name = meta.get('name', frontend_name)

                    result = process_single_image(file, frontend_id, frontend_name)
                    results.append(result)
                    if 'error' not in result:
                        total_defects += result['total_detections']
                    else:
                        processing_errors.append(f"Image '{frontend_name}' (ID: {frontend_id}): {result['error']}")
                else:
                    processing_errors.append(f"Skipping empty or invalid file at index {i}.")
        
        # Single file (old format)
        elif 'image' in request.files:
            file = request.files['image']
            if file and file.filename:
                frontend_id = f"single_{uuid.uuid4().hex[:8]}"
                logger.info(f"Received single image '{file.filename}' via old format.")
                result = process_single_image(file, frontend_id, file.filename)
                results.append(result)
                if 'error' not in result:
                    total_defects += result['total_detections']
                else:
                    processing_errors.append(f"Image '{file.filename}' (ID: {frontend_id}): {result['error']}")
        else:
            return jsonify({"error": "No valid images provided for detection. Please check your request format."}), 400
        
        if not results:
            return jsonify({"error": "No valid images were processed."}), 400
        
        defect_summary = {}
        for result in results:
            if 'error' not in result and isinstance(result.get('predictions'), list):
                for detection in result['predictions']:
                    defect_type = detection['class']
                    defect_summary[defect_type] = defect_summary.get(defect_type, 0) + 1
        
        response_data = {
            "results": results,
            "summary": {
                "total_images_processed": len(results),
                "total_defects_found": total_defects,
                "defect_breakdown": defect_summary,
                "processing_errors": processing_errors
            },
            "debug_info": {
                "model_device": str(model.device) if hasattr(model, 'device') else "unknown",
                "model_names": CLASS_NAMES,
                "total_results": len(results)
            }
        }
        
        logger.info(f"Processed {len(results)} images. Total defects found: {total_defects}")
        return jsonify(response_data), 200
        
    except Exception as e:
        logger.critical(f"CRITICAL ERROR: Unexpected error in predict endpoint: {str(e)}", exc_info=True)
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    if model is not None:
        return jsonify({"status": "online", "message": "Model is loaded and ready."}), 200
    else:
        return jsonify({"status": "offline", "message": "Model is not loaded. Check server logs for details."}), 503

if __name__ == '__main__':
    load_model()
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
