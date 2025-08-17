#!/usr/bin/env python3
"""
Enhanced MTG OCR Scanner - Complete Version
Combines EasyOCR + Tesseract with comprehensive dictionary validation
Includes visual mana symbol detection and advanced preprocessing
"""

import cv2
import numpy as np
import pytesseract
import easyocr
from typing import Dict, List, Tuple, Optional, Set, Any
import re
import time
import os
import json
import sqlite3
from dataclasses import dataclass
from difflib import get_close_matches, SequenceMatcher
from collections import defaultdict
import threading
from functools import lru_cache
import importlib.util

# Import frame detector
from framers_enhanced import MTGCardDetectorEnhanced

@dataclass
class EnhancedOCRResult:
    """Enhanced OCR result with confidence levels"""
    title: str
    mana_cost: str
    type_line: str
    title_confidence: float
    mana_confidence: float
    type_confidence: float
    overall_confidence: float
    processing_time: float
    matches: List[Tuple[str, float]] = None

class ManaSymbolDetector:
    """Visual mana symbol detector using template matching and color analysis"""
    
    def __init__(self):
        # HSV color ranges for mana symbols
        self.mana_colors = {
            'W': ([0, 0, 200], [180, 30, 255]),  # White
            'U': ([100, 50, 50], [130, 255, 255]),  # Blue
            'B': ([0, 0, 0], [180, 255, 30]),  # Black
            'R': ([0, 50, 50], [10, 255, 255]),  # Red
            'G': ([40, 50, 50], [80, 255, 255]),  # Green
        }
        
        # Number templates (simple circles)
        self.number_templates = {}
        self._create_number_templates()
    
    def _create_number_templates(self):
        """Create simple number templates"""
        for i in range(10):
            # Create a simple circle template for each number
            template = np.zeros((30, 30), dtype=np.uint8)
            cv2.circle(template, (15, 15), 12, 255, -1)
            self.number_templates[str(i)] = template
    
    def detect_mana_symbols(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect mana symbols using color analysis and template matching"""
        if image.size == 0:
            return []
        
        results = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect colored mana symbols
        for color, (lower, upper) in self.mana_colors.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Filter small noise
                    x, y, w, h = cv2.boundingRect(contour)
                    results.append({
                        'symbol': color,
                        'bbox': (x, y, w, h),
                        'confidence': min(area / 1000, 1.0)
                    })
        
        # Detect generic mana (numbers)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for num, template in self.number_templates.items():
            try:
                # Scale template to match image size
                h, w = gray.shape
                if h > 0 and w > 0:
                    scaled = cv2.resize(template, (w, h))
                    if scaled.shape[0] <= gray.shape[0] and scaled.shape[1] <= gray.shape[1]:
                        result = cv2.matchTemplate(gray, scaled, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(result)
                        
                        if max_val > 0.5:  # Threshold for detection
                            results.append({
                                'symbol': num,
                                'bbox': (max_loc[0], max_loc[1], w, h),
                                'confidence': max_val
                            })
            except cv2.error:
                continue  # Skip if template is too large
        
        return results
    
    def _detect_numbers_template(self, gray: np.ndarray) -> List[Dict[str, Any]]:
        """Detect numbers using template matching"""
        results = []
        
        for num, template in self.number_templates.items():
            try:
                # Scale template appropriately
                h, w = gray.shape
                if h > 0 and w > 0:
                    scaled = cv2.resize(template, (min(w, 30), min(h, 30)))
                    
                    # Ensure template is not larger than image
                    if scaled.shape[0] <= gray.shape[0] and scaled.shape[1] <= gray.shape[1]:
                        result = cv2.matchTemplate(gray, scaled, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(result)
                        
                        if max_val > 0.5:
                            results.append({
                                'symbol': num,
                                'bbox': (max_loc[0], max_loc[1], scaled.shape[1], scaled.shape[0]),
                                'confidence': max_val
                            })
            except cv2.error:
                continue
        
        return results

class EnhancedOCRProcessor:
    """Enhanced OCR processor with hybrid Tesseract + EasyOCR approach"""
    
    def __init__(self):
        # Initialize EasyOCR without GPU for compatibility
        try:
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False)
        except Exception as e:
            print(f"Warning: Could not initialize EasyOCR: {e}")
            self.easyocr_reader = None
        
        # Pre-compiled regex patterns
        self.artifact_patterns = [
            (re.compile(r'\s*\^\s*'), ' '),
            (re.compile(r'\s*[;,]\s*'), ' '),
            (re.compile(r'\s+'), ' '),
            (re.compile(r'[|!]'), 'l'),
            (re.compile(r'[0O]'), '0'),  # Common OCR confusion
        ]
        
        # Load dictionary and corrections
        self.mtg_dictionary = self._load_mtg_dictionary()
        self.ocr_corrections = self._load_ocr_corrections()
        
        # Tesseract configs
        self.title_config = '--oem 3 --psm 7 -c tessedit_char_blacklist=@#$%^&*(){}<>/\\|`~'
        self.type_config = '--oem 3 --psm 7'
        self.mana_config = '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789WUBRGCX{}'
    
    def _load_mtg_dictionary(self) -> Set[str]:
        """Load comprehensive MTG dictionary"""
        try:
            with open('mtg_dictionary.json', 'r') as f:
                data = json.load(f)
                return set(data.get('words', []))
        except FileNotFoundError:
            print("Warning: mtg_dictionary.json not found")
            return set()
    
    def _load_ocr_corrections(self) -> Dict[str, str]:
        """Load OCR corrections"""
        try:
            with open('mtg_ocr_corrections.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("Warning: mtg_ocr_corrections.json not found")
            return {}
    
    def preprocess_for_easyocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocessing optimized for EasyOCR"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Upscale for better OCR
        h, w = gray.shape
        scaled = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(scaled)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        return enhanced
    
    def preprocess_for_tesseract(self, image: np.ndarray) -> np.ndarray:
        """Preprocessing optimized for Tesseract"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Upscale
        h, w = gray.shape
        scaled = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
        
        # Threshold
        _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def extract_text_easyocr(self, image: np.ndarray) -> Tuple[str, float]:
        """Extract text using EasyOCR"""
        if image.size == 0 or self.easyocr_reader is None:
            return "", 0.0
        
        processed = self.preprocess_for_easyocr(image)
        
        try:
            results = self.easyocr_reader.readtext(processed)
            
            if results:
                # Combine all detected text
                text_parts = []
                confidences = []
                
                for (bbox, text, confidence) in results:
                    if confidence > 0.3:  # Filter low confidence
                        text_parts.append(text)
                        confidences.append(confidence)
                
                combined_text = ' '.join(text_parts)
                avg_confidence = np.mean(confidences) if confidences else 0.0
                
                return combined_text, avg_confidence
            else:
                return "", 0.0
                
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return "", 0.0
    
    def extract_text_tesseract(self, image: np.ndarray, region_type: str) -> Tuple[str, float]:
        """Extract text using Tesseract"""
        if image.size == 0:
            return "", 0.0
        
        processed = self.preprocess_for_tesseract(image)
        
        # Select config based on region
        if region_type == 'title':
            config = self.title_config
        elif region_type == 'type':
            config = self.type_config
        else:
            config = self.mana_config
        
        try:
            data = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
            
            text_parts = []
            confidences = []
            
            for i, conf in enumerate(data['conf']):
                if conf > 30:  # Filter low confidence
                    text_parts.append(data['text'][i])
                    confidences.append(conf)
            
            combined_text = ' '.join(text_parts)
            avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0
            
            return combined_text, avg_confidence
            
        except Exception as e:
            print(f"Tesseract error: {e}")
            return "", 0.0
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Apply artifact patterns
        cleaned = text
        for pattern, replacement in self.artifact_patterns:
            cleaned = pattern.sub(replacement, cleaned)
        
        # Apply OCR corrections
        for wrong, correct in self.ocr_corrections.items():
            cleaned = cleaned.replace(wrong, correct)
        
        # Final cleanup
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def validate_with_dictionary(self, text: str, confidence: float) -> Tuple[str, float]:
        """Validate text using MTG dictionary"""
        if not text or not self.mtg_dictionary:
            return text, confidence
        
        # Check if text is in dictionary
        if text.lower() in self.mtg_dictionary:
            return text, confidence
        
        # Try to find close matches
        matches = get_close_matches(text.lower(), self.mtg_dictionary, n=3, cutoff=0.6)
        
        if matches:
            best_match = matches[0]
            similarity = SequenceMatcher(None, text.lower(), best_match).ratio()
            adjusted_confidence = confidence * similarity
            return best_match, adjusted_confidence
        
        return text, confidence * 0.5  # Penalize if no match found

class EnhancedMTGScanner:
    """Enhanced MTG scanner with hybrid OCR and visual detection"""
    
    def __init__(self, db_path: str = "code_to_integrate/mtg_complete.db"):
        self.db_path = db_path
        self.frame_detector = MTGCardDetectorEnhanced()
        self.ocr_processor = EnhancedOCRProcessor()
        self.mana_detector = ManaSymbolDetector()
        
        # Load card names for matching
        self.card_names = self._load_card_names()
        
        print("✅ Enhanced MTG Scanner initialized")
    
    def _load_card_names(self) -> List[str]:
        """Load card names from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute("SELECT name FROM cards")
            names = [row[0] for row in cursor.fetchall()]
            conn.close()
            return names
        except Exception as e:
            print(f"Warning: Could not load card names: {e}")
            return []
    
    def process_regions_parallel(self, regions: Dict[str, np.ndarray]) -> Dict[str, Tuple[str, float]]:
        """Process regions in parallel for speed"""
        results = {}
        
        def process_region(name: str, image: np.ndarray):
            if name == 'title':
                # Use Tesseract for titles (faster)
                text, conf = self.ocr_processor.extract_text_tesseract(image, 'title')
            elif name == 'type_line':
                # Use EasyOCR for type lines (more accurate)
                text, conf = self.ocr_processor.extract_text_easyocr(image)
            elif name == 'mana_cost':
                # Try visual detection first, then OCR
                text, conf = self._process_mana_region(image)
            else:
                # Default to EasyOCR
                text, conf = self.ocr_processor.extract_text_easyocr(image)
            
            # Clean and validate
            cleaned_text = self.ocr_processor.clean_text(text)
            validated_text, adjusted_conf = self.ocr_processor.validate_with_dictionary(cleaned_text, conf)
            
            results[name] = (validated_text, adjusted_conf)
        
        # Process regions in parallel
        threads = []
        for name, image in regions.items():
            thread = threading.Thread(target=process_region, args=(name, image))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        return results
    
    def _process_mana_region(self, image: np.ndarray) -> Tuple[str, float]:
        """Process mana cost region with visual detection"""
        # Try visual detection first
        symbols = self.mana_detector.detect_mana_symbols(image)
        
        if symbols:
            # Convert visual symbols to text
            symbol_text = []
            total_confidence = 0
            
            for symbol in symbols:
                symbol_text.append(symbol['symbol'])
                total_confidence += symbol['confidence']
            
            text = ''.join(symbol_text)
            avg_confidence = total_confidence / len(symbols)
            return text, avg_confidence
        
        # Fallback to OCR
        return self.ocr_processor.extract_text_easyocr(image)
    
    def scan_card(self, image_path: str) -> List[EnhancedOCRResult]:
        """Scan a single card image"""
        start_time = time.time()
        
        # Detect card frame
        detected_cards = self.frame_detector.detect_cards_adaptive(image_path)
        
        results = []
        for card, normalized, regions in detected_cards:
            # Process regions
            ocr_results = self.process_regions_parallel(regions)
            
            # Extract results
            title, title_conf = ocr_results.get('title', ('', 0.0))
            mana, mana_conf = ocr_results.get('mana_cost', ('', 0.0))
            type_line, type_conf = ocr_results.get('type_line', ('', 0.0))
            
            # Calculate overall confidence
            overall_conf = (title_conf + mana_conf + type_conf) / 3
            
            # Find database matches
            matches = self._find_matches(title, mana, type_line)
            
            result = EnhancedOCRResult(
                title=title,
                mana_cost=mana,
                type_line=type_line,
                title_confidence=title_conf,
                mana_confidence=mana_conf,
                type_confidence=type_conf,
                overall_confidence=overall_conf,
                processing_time=time.time() - start_time,
                matches=matches
            )
            
            results.append(result)
        
        return results
    
    def _find_matches(self, title: str, mana: str, type_line: str) -> List[Tuple[str, float]]:
        """Find database matches for the scanned card"""
        if not title or not self.card_names:
            return []
        
        matches = []
        
        # Direct name match
        if title in self.card_names:
            matches.append((title, 1.0))
        
        # Fuzzy matching
        close_matches = get_close_matches(title, self.card_names, n=5, cutoff=0.6)
        for match in close_matches:
            similarity = SequenceMatcher(None, title.lower(), match.lower()).ratio()
            matches.append((match, similarity))
        
        # Sort by confidence
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:5]
    
    def process_batch(self, image_folder: str) -> Dict[str, Any]:
        """Process a batch of images"""
        results = []
        total_time = 0
        
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(image_folder, filename)
                
                try:
                    card_results = self.scan_card(image_path)
                    for result in card_results:
                        results.append({
                            'image': filename,
                            'title': result.title,
                            'mana_cost': result.mana_cost,
                            'type_line': result.type_line,
                            'confidence': result.overall_confidence,
                            'processing_time': result.processing_time,
                            'matches': result.matches
                        })
                    
                    total_time += result.processing_time
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        return {
            'total_cards': len(results),
            'total_time': total_time,
            'average_time': total_time / len(results) if results else 0,
            'results': results
        }

def test_enhanced_scanner():
    """Test the enhanced scanner"""
    scanner = EnhancedMTGScanner()
    
    # Test with sample images
    test_images = []
    test_photos_dir = "test photos"
    
    if os.path.exists(test_photos_dir):
        for file in os.listdir(test_photos_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(test_photos_dir, file))
                if len(test_images) >= 3:
                    break
    
    if not test_images:
        print("❌ No test images found")
        return
    
    print(f"🧪 Testing Enhanced OCR Scanner with {len(test_images)} images")
    print("=" * 60)
    
    for i, image_path in enumerate(test_images, 1):
        print(f"\n🔍 Processing {os.path.basename(image_path)} ({i}/{len(test_images)})")
        
        results = scanner.scan_card(image_path)
        
        for j, result in enumerate(results, 1):
            print(f"  Card {j}:")
            print(f"    Title: '{result.title}' (conf: {result.title_confidence:.3f})")
            print(f"    Mana: '{result.mana_cost}' (conf: {result.mana_confidence:.3f})")
            print(f"    Type: '{result.type_line}' (conf: {result.type_confidence:.3f})")
            print(f"    Overall: {result.overall_confidence:.3f}")
            print(f"    Time: {result.processing_time:.3f}s")
            
            if result.matches:
                print(f"    Top match: {result.matches[0][0]} ({result.matches[0][1]:.3f})")
    
    print("\n✅ Enhanced OCR Scanner test complete!")

if __name__ == "__main__":
    test_enhanced_scanner() 