from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
import io
import base64
import numpy as np
import os
import json
import uuid
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import YOLOv5 using the pip package
try:
    import yolov5
    YOLOV5_AVAILABLE = True
    logger.info("YOLOv5 package imported successfully from pip.")
except ImportError as e:
    logger.critical(f"CRITICAL ERROR: Failed to import YOLOv5 package: {e}")
    logger.critical("Please ensure yolov5 is installed. Render build might be failing.")
    YOLOV5_AVAILABLE = False
    sys.exit(1)

# Initialize Flask app
app = Flask(__name__)

# UPDATED: Add your new Cloudflare domain to the allowed origins.
CORS(app, 
     resources={
         r"/predict": {
             "origins": [
                 "https://neural-pcb-project.vercel.app", 
                 "http://localhost:3000", 
                 "http://127.0.0.1:3000",
                 "https://appropriate-accuracy-suffering-d.trycloudflare.com",
                 "https://sensitive-delivers-peas-research.trycloudflare.com" # <-- ADD THIS LINE
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
                 "https://sensitive-delivers-peas-research.trycloudflare.com"# <-- AND THIS LINE
             ],
             "methods": ["GET", "OPTIONS"],
             "allow_headers": ["Content-Type", "Authorization"]
         }
     },
     supports_credentials=True)

# Load YOLOv5 model
model = None
CLASS_NAMES = {}

if YOLOV5_AVAILABLE:
    logger.info("Attempting to load YOLOv5 model...")
    try:
        model_path = None
        if os.path.exists('best.pt'):
            model_path = 'best.pt'
            logger.info("Found custom model: best.pt")
        elif os.path.exists('last.pt'):
            model_path = 'last.pt'
            logger.info("Found custom model: last.pt")
        else:
            logger.warning("No custom model found (best.pt or last.pt). Attempting to use pretrained YOLOv5s as fallback.")
            model_path = 'yolov5s.pt'

        if model_path:
            logger.info(f"Loading model from path: {model_path}")
            try:
                model = yolov5.load(model_path)
                
                model.conf = 0.25
                model.iou = 0.45
                model.max_det = 5
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                
                logger.info(f"Model loaded successfully: {model_path} on device: {device}")
                
                if hasattr(model, 'names') and isinstance(model.names, dict):
                    CLASS_NAMES = model.names
                    logger.info(f"Model class names (from model.names): {CLASS_NAMES}")
                else:
                    CLASS_NAMES = {
                        0: 'missing_hole', 1: 'mouse_bite', 2: 'open_circuit',
                        3: 'short', 4: 'spur', 5: 'spurious_copper'
                    }
                    logger.warning(f"Model.names not found or invalid. Using hardcoded class names: {CLASS_NAMES}")

            except Exception as load_error:
                logger.critical(f"CRITICAL ERROR: Failed to load model with yolov5.load(): {load_error}", exc_info=True)
                model = None
        else:
            logger.critical("CRITICAL ERROR: No model path determined. Model will not be loaded.")
            model = None
        
    except Exception as e:
        logger.critical(f"CRITICAL ERROR: General error during model loading block initialization: {e}", exc_info=True)
        model = None
else:
    logger.critical("CRITICAL ERROR: YOLOv5 not available (import failed). Model will not be loaded.")

if not CLASS_NAMES:
    CLASS_NAMES = {
        0: 'missing_hole', 1: 'mouse_bite', 2: 'open_circuit',
        3: 'short', 4: 'spur', 5: 'spurious_copper'
    }
    logger.warning("CLASS_NAMES was not populated from model, using default hardcoded names.")


@app.route('/')
def home():
    """Simple home route for health check."""
    model_status = "loaded" if model is not None else "not loaded"
    return f"PCB Defect Detector Backend is running! Model status: {model_status}", 200

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "yolov5_available": YOLOV5_AVAILABLE
    }), 200

def expand_bounding_box(bbox, image_width, image_height, expansion_factor=0.05):
    """Expand bounding box by a given factor"""
    x1, y1, x2, y2 = bbox
    
    width = x2 - x1
    height = y2 - y1
    
    expand_x = width * expansion_factor
    expand_y = height * expansion_factor
    
    new_x1 = max(0, x1 - expand_x)
    new_y1 = max(0, y1 - expand_y)
    new_x2 = min(image_width, x2 + expand_x)
    new_y2 = min(image_height, y2 + expand_y)
    
    return new_x1, new_y1, new_x2, new_y2

def process_single_image(image_data, frontend_image_id, filename=None):
    """Process a single image for defect detection using YOLOv5"""
    if model is None:
        logger.error(f"Attempted to process image {filename or frontend_image_id} but model is not loaded.")
        return {
            "image_id": frontend_image_id,
            "error": "Model is not loaded, cannot process image.",
            "predictions": [],  # Changed from detections to predictions
            "image_dimensions": {"width": 0, "height": 0},
            "total_detections": 0
        }
    try:
        # Handle different image input types
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            header, encoded = image_data.split(',', 1)
            image_bytes = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        else:
            image = Image.open(image_data.stream).convert('RGB')

        original_width, original_height = image.size
        logger.info(f"Processing image: {filename or frontend_image_id}, Size: {original_width}x{original_height}")

        logger.info(f"STARTING model inference for {filename or frontend_image_id}...")
        with torch.no_grad():
            results = model(image)  # YOLOv5 inference
        logger.info(f"COMPLETED model inference for {filename or frontend_image_id}.")

        predictions = []  # Changed from detections to predictions

        # Handle both single image and batch outputs
        pandas_results = results.pandas().xyxy
        if not isinstance(pandas_results, list):
            pandas_results = [pandas_results]

        for df in pandas_results:
            for _, detection in df.iterrows():
                x1, y1, x2, y2 = detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']
                confidence = detection['confidence']
                class_id = int(detection['class'])
                class_name = detection['name']

                if class_id in CLASS_NAMES:
                    class_name = CLASS_NAMES[class_id]

                # Expand bbox slightly
                x1_exp, y1_exp, x2_exp, y2_exp = expand_bounding_box(
                    (x1, y1, x2, y2), original_width, original_height
                )

                # Return original pixel coordinates (not normalized)
                detection_data = {
                    "class_id": class_id,
                    "class": class_name,
                    "confidence": float(confidence),
                    "bbox": {
                        "x1": float(x1_exp),  # Keep as pixel coordinates
                        "y1": float(y1_exp),
                        "x2": float(x2_exp),
                        "y2": float(y2_exp)
                    }
                }

                predictions.append(detection_data)  # Changed from detections to predictions
                logger.debug(f"Detection: {class_name} ({confidence:.3f}) at ({x1_exp}, {y1_exp}, {x2_exp}, {y2_exp})")

        # Keep only top-N detections
        max_detections_per_image = 5
        predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)[:max_detections_per_image]

        result = {
            "image_id": frontend_image_id,
            "predictions": predictions,  # Changed from detections to predictions
            "image_dimensions": {
                "width": original_width,
                "height": original_height
            },
            "total_detections": len(predictions)  # Changed from detections to predictions
        }

        logger.info(f"Found {len(predictions)} detections for {filename or frontend_image_id}")
        return result

    except Exception as e:
        logger.critical(f"CRITICAL ERROR: Failed during image processing for {filename or frontend_image_id}: {str(e)}", exc_info=True)
        return {
            "image_id": frontend_image_id,
            "error": f"Processing failed: {str(e)}",
            "predictions": [],  # Changed from detections to predictions
            "image_dimensions": {"width": 0, "height": 0},
            "total_detections": 0
        }

        
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
            if 'error' not in result and isinstance(result.get('predictions'), list):  # Changed from 'detections' to 'predictions'
                for detection in result['predictions']:  # Changed from 'detections' to 'predictions'
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
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=False, host='0.0.0.0', port=port)
