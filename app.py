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

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

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
    
    try:
        if isinstance(image_stream, str): # Handle base64 encoded images
            from base64 import b64decode
            image_stream = io.BytesIO(b64decode(image_stream.split(',')[1]))

        # Load the image using PIL and convert to RGB
        pil_image = Image.open(image_stream).convert('RGB')
        original_width, original_height = pil_image.size
        img_np = np.array(pil_image)
        
        # Perform prediction
        results = model(img_np)
        
        # Extract predictions. The 'results' object is not iterable directly.
        # We need to access the specific prediction data.
        predictions_list = []
        if results.pred and len(results.pred[0]):
            # The results.pred attribute is a list of tensors.
            # We iterate over the first tensor for the first image in the batch.
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
        logger.error(f"Error processing image {frontend_name} (ID: {frontend_image_id}): {e}", exc_info=True)
        return {"image_id": frontend_image_id, "error": f"Processing failed: {str(e)}"}

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
    
    if model is None:
        logger.error("Predict endpoint called but model is not loaded.")
        return jsonify({"error": "Model not loaded. Please ensure your model file (best.pt or last.pt) is available or check server logs for critical errors during startup."}), 500

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

if __name__ == '__main__':
    load_model()
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port, debug=True)
