#!/usr/bin/env python3
"""
Flask API backend for MTG Card Detection
Provides endpoints for the React Native frontend to detect MTG cards
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import os
import base64
import tempfile
from PIL import Image
import io
import json

# Import our MTG detection modules
from border_detection_enhanced import detect_borders_enhanced, CardBorders
from framers_enhanced import MTGCardDetectorEnhanced, create_enhanced_detection_visualization
from enhanced_ocr_scanner import EnhancedMTGScanner
from mtg_mana_detector_improved import ImprovedManaDetector

app = Flask(__name__)
CORS(app)  # Enable CORS for React Native

# Initialize all detectors
detector = MTGCardDetectorEnhanced()
ocr_scanner = EnhancedMTGScanner()
mana_detector = ImprovedManaDetector(use_ocr=True, debug=True)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "MTG Card Detection API is running"})

@app.route('/detect-card', methods=['POST'])
def detect_card():
    """Detect MTG card from uploaded image"""
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No image file selected"}), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            file.save(tmp_file.name)
            temp_image_path = tmp_file.name
        
        try:
            # Load image for border detection
            image = cv2.imread(temp_image_path)
            if image is None:
                return jsonify({"error": "Could not load image"}), 400
            
            # Step 1: Detect MTG black border first
            from border_detection_enhanced import detect_mtg_black_border
            borders = detect_borders_enhanced(image)
            card_contour = detect_mtg_black_border(image)
            
            # Step 2: Run enhanced detection pipeline
            results = detector.detect_cards_adaptive(temp_image_path)
            
            if not results:
                return jsonify({"error": "No MTG card detected in the image"}), 400
            
            # Get the first (and should be only) result
            detected_card, content, regions, scaled_canonical, canonical_coords = results[0]
            
            # Create visualization using original image for better context
            vis_image = image.copy()
            
            # Draw actual card contour if found
            if card_contour is not None:
                cv2.polylines(vis_image, [card_contour.astype(np.int32)], True, (0, 255, 0), 4)
                cv2.putText(vis_image, "Detected Card", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw card borders (bounding box)
            cv2.rectangle(vis_image, (borders.left, borders.top), (borders.right, borders.bottom), (0, 255, 255), 2)
            cv2.putText(vis_image, "Card Bounds", (borders.left, borders.top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Step 3: Extract text and mana from detected regions
            extracted_text = {}
            extracted_mana = {}
            
            # Process each detected region
            for region_name, region in regions.items():
                if region.exists:
                    # Get region image in original coordinate space
                    x1 = region.x1 + borders.left
                    y1 = region.y1 + borders.top
                    x2 = region.x2 + borders.left
                    y2 = region.y2 + borders.top
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, min(x1, image.shape[1]))
                    y1 = max(0, min(y1, image.shape[0]))
                    x2 = max(0, min(x2, image.shape[1]))
                    y2 = max(0, min(y2, image.shape[0]))
                    
                    region_image = image[y1:y2, x1:x2]
                    
                    if region_image.size > 0:
                        print(f"    Processing {region_name} region: {region_image.shape}")
                        
                        # Run OCR on ALL frame segments for complete text extraction
                        try:
                            # Choose appropriate OCR method based on region type
                            if region_name == 'title':
                                # Tesseract optimized for card titles
                                text, conf = ocr_scanner.ocr_processor.extract_text_tesseract(region_image, 'title')
                            elif region_name == 'mana_cost':
                                # First try mana detection, then OCR as fallback
                                mana_result = mana_detector.detect_mana_cost(region_image)
                                extracted_mana['mana_cost'] = {
                                    'mana_string': mana_result.mana_string,
                                    'confidence': mana_result.confidence,
                                    'symbols': [{'symbol': s.symbol, 'confidence': s.confidence} 
                                              for s in mana_result.symbols],
                                    'detection_method': mana_result.detection_method,
                                    'region': (x1, y1, x2, y2)
                                }
                                # Also run OCR for text fallback
                                text, conf = ocr_scanner.ocr_processor.extract_text_tesseract(region_image, 'mana')
                            elif region_name == 'type_line':
                                # EasyOCR for type lines (better for game terminology)
                                text, conf = ocr_scanner.ocr_processor.extract_text_easyocr(region_image)
                            elif region_name == 'set_symbol':
                                # Tesseract for set symbols (small text/numbers)
                                text, conf = ocr_scanner.ocr_processor.extract_text_tesseract(region_image, 'type')
                            else:
                                # Default to EasyOCR for other regions
                                text, conf = ocr_scanner.ocr_processor.extract_text_easyocr(region_image)
                            
                            # Clean and store text result for ALL regions
                            cleaned_text = ocr_scanner.ocr_processor.clean_text(text)
                            extracted_text[region_name] = {
                                'text': cleaned_text,
                                'confidence': conf,
                                'region': (x1, y1, x2, y2),
                                'raw_region_size': f"{region_image.shape[1]}x{region_image.shape[0]}"
                            }
                            print(f"      OCR result: '{cleaned_text}' (confidence: {conf:.3f})")
                            
                        except Exception as e:
                            print(f"    OCR/Detection error for {region_name}: {e}")
                            extracted_text[region_name] = {
                                'text': '',
                                'confidence': 0.0,
                                'region': (x1, y1, x2, y2),
                                'error': str(e)
                            }
                            if region_name == 'mana_cost':
                                extracted_mana['mana_cost'] = {
                                    'mana_string': '',
                                    'confidence': 0.0,
                                    'symbols': [],
                                    'detection_method': 'error',
                                    'region': (x1, y1, x2, y2)
                                }
            
            # Define colors for different regions
            colors = {
                'title': (0, 255, 255),      # Cyan
                'type_line': (255, 0, 0),    # Red
                'text': (0, 0, 255),         # Blue
                'mana_cost': (255, 255, 0),  # Yellow
                'set_symbol': (255, 0, 255)  # Magenta
            }
            
            # Draw bounding boxes and labels
            detected_regions = []
            for region_name, region in regions.items():
                if region.exists:
                    color = colors.get(region_name, (128, 128, 128))
                    # Adjust coordinates to original image space (add border offset)
                    x1 = region.x1 + borders.left
                    y1 = region.y1 + borders.top
                    x2 = region.x2 + borders.left
                    y2 = region.y2 + borders.top
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label with extracted text preview
                    if region_name in extracted_text:
                        text_preview = extracted_text[region_name]['text'][:20]
                        label = f"{region_name}: {text_preview}... ({region.confidence:.2f})"
                    elif region_name in extracted_mana:
                        mana_preview = extracted_mana[region_name]['mana_string']
                        label = f"{region_name}: {mana_preview} ({region.confidence:.2f})"
                    else:
                        label = f"{region_name} ({region.confidence:.2f})"
                    
                    cv2.putText(vis_image, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Store region data for frontend (use adjusted coordinates)
                    detected_regions.append({
                        'name': region_name,
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'confidence': region.confidence,
                        'color': [int(c) for c in color]
                    })
            
            # Add card info overlay
            info_text = f"Style: {detected_card.card_style}, Confidence: {detected_card.confidence:.2f}"
            cv2.putText(vis_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add canonical coordinates info
            if canonical_coords:
                y_content_pct = canonical_coords.get('y_content_pct', 0)
                info_text2 = f"Content %: {y_content_pct:.3f}"
                cv2.putText(vis_image, info_text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Convert the visualization to base64 for frontend
            _, buffer = cv2.imencode('.jpg', vis_image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Clean up temporary file
            os.unlink(temp_image_path)
            
            return jsonify({
                "success": True,
                "detected_regions": detected_regions,
                "card_style": detected_card.card_style,
                "confidence": detected_card.confidence,
                "extracted_text": extracted_text,
                "extracted_mana": extracted_mana,
                "visualization": img_base64,
                "message": "MTG card detected successfully"
            })
            
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
            raise e
            
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

@app.route('/detect-borders', methods=['POST'])
def detect_borders():
    """Detect just the card borders from uploaded image"""
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No image file selected"}), 400
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            file.save(tmp_file.name)
            temp_image_path = tmp_file.name
        
        try:
            # Load image
            image = cv2.imread(temp_image_path)
            if image is None:
                return jsonify({"error": "Could not load image"}), 400
            
            # Detect borders
            borders = detect_borders_enhanced(image)
            
            # Create visualization
            vis_image = image.copy()
            cv2.rectangle(vis_image, (borders.left, borders.top), (borders.right, borders.bottom), (0, 255, 0), 3)
            
            # Add border info
            cv2.putText(vis_image, f"Borders: L={borders.left}, R={borders.right}, T={borders.top}, B={borders.bottom}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', vis_image)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Clean up temporary file
            os.unlink(temp_image_path)
            
            return jsonify({
                "success": True,
                "borders": {
                    "left": borders.left,
                    "right": borders.right,
                    "top": borders.top,
                    "bottom": borders.bottom
                },
                "visualization": img_base64,
                "message": "Card borders detected successfully"
            })
            
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
            raise e
            
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

if __name__ == '__main__':
    print("Starting MTG Card Detection API...")
    print("API will be available at http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)
