#!/usr/bin/env python3
"""
Enhanced MTG Card Detector - Fast detector with enhanced border detection
Integrates the advanced border detection with the speed-first, accuracy-kept recipe
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import math

# Import enhanced border detection
from border_detection_enhanced import detect_borders_enhanced, CardBorders

@dataclass
class DetectedRegion:
    name: str
    y1: int
    y2: int
    x1: int
    x2: int
    confidence: float
    method: str
    exists: bool = True

@dataclass
class TextRegion:
    x1: int
    y1: int
    x2: int
    y2: int
    density: float

@dataclass
class LayoutAnchors:
    type_line_y: int
    text_top_y: int
    borders: CardBorders
    set_symbol_y: Optional[int]
    copyright_y: Optional[int]
    confidence: float
    detection_method: str

@dataclass
class DetectedCard:
    card_style: str
    confidence: float
    anchors: LayoutAnchors
    regions: Dict[str, DetectedRegion]

class MTGCardDetectorEnhanced:
    """Enhanced MTG Card Detector with advanced border detection"""
    
    def __init__(self):
        # Template priors (precomputed from good samples)
        self.STROKE_BETA = 0.008  # stroke ≈ β · H_content (empirical)
        self.TARGET_STROKE = 3.0  # target stroke width in pixels
        self.MIN_STROKE = 2.2     # minimum acceptable stroke width
        self.MAX_STROKE = 5.5     # maximum acceptable stroke width
        
        # Threshold parameters
        self.MIN_INK_PERCENT = 5.0
        self.MAX_INK_PERCENT = 25.0
        
        # Morphology parameters
        self.MORPH_H_RATIO = 0.8  # horizontal kernel ≈ 0.8 × stroke
        self.MORPH_V_RATIO = 0.5  # vertical kernel ≈ 0.5 × stroke
        
        # Text block parameters
        self.MIN_LINES_FOR_BLOCK = 3
        self.LINE_SPACING_MIN = 0.3
        self.LINE_SPACING_MAX = 0.8
        
        # Type line parameters
        self.MAX_TYPE_LINE_CANDIDATES = 8
        self.PEAK_MIN_DISTANCE_RATIO = 0.03  # 3% of height
        
    def detect_cards_adaptive(self, image_path: str) -> List[Tuple[DetectedCard, np.ndarray, Dict[str, DetectedRegion], np.ndarray]]:
        """Main detection pipeline - content-reference frame approach"""
        print(f"Processing: {os.path.basename(image_path)}")
        print("-" * 50)
        
        # Load and preprocess
        image = cv2.imread(image_path)
        if image is None:
            return []
        
        # Step 0: Detect borders and establish content reference frame
        print("  Detecting borders and establishing content reference frame...")
        borders = self._detect_borders_enhanced(image)
        
        # Calculate scale factor for processing
        h, w = image.shape[:2]
        proc_scale = 600.0 / w if w > 600 else 1.0
        print(f"    Processing scale: {proc_scale:.3f}")
        
        # Crop to content area
        content = image[borders.top:borders.bottom, borders.left:borders.right]
        if content.size == 0:
            print("    ❌ No content after border detection")
            return []
        
        print(f"    Content size: {content.shape}")
        
        # Step 1: Normalize geometry (content-reference frame)
        print("  Normalizing geometry to content space...")
        
        # Measure border thickness robustly
        w_top, w_bottom, w_left, w_right = self._measure_border_thickness_robust(content)
        print(f"    Border thickness: top={w_top:.1f}, bottom={w_bottom:.1f}, left={w_left:.1f}, right={w_right:.1f}")
        
        # Create outer quad (approximate)
        outer_quad = np.array([
            [borders.left, borders.top],
            [borders.right, borders.top],
            [borders.right, borders.bottom],
            [borders.left, borders.bottom]
        ], dtype=np.float32)
        
        # Inset to content quad
        content_quad = self._inset_to_content_quad(outer_quad, w_top, w_bottom, w_left, w_right)
        
        # Compute canonical homography
        H_content, H_content_inv = self._compute_canonical_homography(content_quad)
        
        # Sanity check: frame aspect ratio
        Wc, Hc = 630, 880
        canonical_aspect = Hc / Wc
        expected_aspect = 88 / 63
        if abs(canonical_aspect - expected_aspect) > 0.01:
            print(f"    ⚠️  Frame aspect ratio check failed: {canonical_aspect:.3f} vs {expected_aspect:.3f}")
        
        # Sanity check: side asymmetry
        side_asymmetry = abs(w_top - w_bottom) / Hc
        if side_asymmetry > 0.02:
            print(f"    ⚠️  Side asymmetry detected: {side_asymmetry:.3f} > 0.02")
        
        # Step 2: Process in canonical space
        print("  Processing in canonical space...")
        
        # Transform content to canonical space
        canonical_content = cv2.warpPerspective(content, H_content, (Wc, Hc))
        
        # Determine optimal scale for processing
        scaled_canonical, scale_factor = self._determine_optimal_scale(canonical_content)
        
        # Staged binarization
        print("  Processing illumination and binarization...")
        binary, method = self._fast_binarization(scaled_canonical)
        
        # Stroke-aware morphology
        print("  Applying stroke-aware morphology...")
        cleaned = self._stroke_aware_morphology(binary, scale_factor)
        
        # Fast text block decision
        print("  Detecting text blocks...")
        text_regions, text_confidence = self._fast_text_block_detection(cleaned, scaled_canonical, scale_factor)
        
        if not text_regions:
            print("    ⚠️  No text regions detected, using fallback")
            return []
        
        # Step 3: Type-line detection in canonical space
        print("  Detecting type line in canonical space...")
        type_line_y_canonical, type_confidence, type_method = self._pruned_type_line_detection_canonical(scaled_canonical, text_regions, scale_factor)
        
        # Convert back to original scale
        type_line_y_canonical = int(type_line_y_canonical / scale_factor)
        
        # Convert canonical coordinates back to original image space
        type_line_point_canonical = np.array([[Wc/2, type_line_y_canonical]], dtype=np.float32)
        type_line_point_original = cv2.perspectiveTransform(type_line_point_canonical.reshape(-1, 1, 2), H_content_inv)
        type_line_y_original = int(type_line_point_original[0, 0, 1])
        
        # Calculate content percentage for logging
        y_content_pct = type_line_y_canonical / Hc
        print(f"    Type line detected at {y_content_pct:.3f} of content height (y={type_line_y_canonical}/{Hc})")
        
        # Store canonical coordinates for later use
        canonical_coords = {
            'type_line_y': type_line_y_canonical,
            'y_content_pct': y_content_pct,
            'Hc': Hc,
            'Wc': Wc
        }
        
        # Create layout anchors
        anchors = LayoutAnchors(
            type_line_y=type_line_y_original,
            text_top_y=min([r.y1 for r in text_regions]) if text_regions else type_line_y_original + 50,
            borders=borders,
            set_symbol_y=None,
            copyright_y=None,
            confidence=type_confidence,
            detection_method=type_method
        )
        
        # Infer card style
        card_style = self._infer_card_style(canonical_content, text_regions)
        
        # Infer regions from anchors
        regions = self._infer_regions_from_anchors(content, anchors, card_style)
        
        # Create detected card
        detected_card = DetectedCard(
            card_style=card_style,
            confidence=type_confidence,
            anchors=anchors,
            regions=regions
        )
        
        # Store canonical coordinates in the result
        return [(detected_card, content, regions, scaled_canonical, canonical_coords)]
    
    def _deskew_and_normalize(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Deskew and normalize the image"""
        # For now, just return the image as-is
        # In a full implementation, you'd add deskewing logic here
        return image
    
    def _detect_borders_enhanced(self, normalized: np.ndarray) -> CardBorders:
        """Use enhanced border detection"""
        return detect_borders_enhanced(normalized)
    
    def _determine_optimal_scale(self, content: np.ndarray) -> Tuple[np.ndarray, float]:
        """Step 1: Pick the right scale once using stroke prediction"""
        h, w = content.shape[:2]
        
        # Handle edge case where content has zero height
        if h == 0 or w == 0:
            print(f"    ⚠️  Invalid content size: {content.shape}, using fallback")
            return content, 1.0
        
        # Predict stroke width using template prior
        predicted_stroke = self.STROKE_BETA * h
        
        print(f"    Predicted stroke width: {predicted_stroke:.2f}px")
        
        # Determine if resize is needed
        if predicted_stroke < self.MIN_STROKE:
            scale_factor = 3.0 / predicted_stroke
            scale_factor = min(scale_factor, 5.0)  # Cap at 5x upscaling
            print(f"    Upscaling by {scale_factor:.2f}x (stroke too small)")
        elif predicted_stroke > self.MAX_STROKE:
            scale_factor = 3.8 / predicted_stroke
            scale_factor = max(scale_factor, 0.2)  # Cap at 5x downscaling
            print(f"    Downscaling by {scale_factor:.2f}x (stroke too large)")
        else:
            scale_factor = 1.0
            print(f"    No resize needed (stroke in range)")
        
        # Apply resize if needed
        if scale_factor != 1.0:
            new_h = int(h * scale_factor)
            new_w = int(w * scale_factor)
            
            # Additional safety check for reasonable dimensions
            if new_h > 2000 or new_w > 2000:
                print(f"    ⚠️  Resize would create absurd dimensions ({new_w}x{new_h}), using fallback")
                return content, 1.0
                
            resized = cv2.resize(content, (new_w, new_h))
            print(f"    Resized to: {resized.shape}")
            return resized, scale_factor
        else:
            return content, 1.0
    
    def _fast_binarization(self, content: np.ndarray) -> Tuple[np.ndarray, str]:
        """Step 2: Staged binarization (Otsu → Sauvola fallback)"""
        # Handle edge case where content has zero dimensions
        if content.shape[0] == 0 or content.shape[1] == 0:
            print(f"    ⚠️  Invalid content after scaling: {content.shape}")
            # Return a dummy binary image
            dummy = np.zeros((100, 100), dtype=np.uint8)
            return dummy, "fallback"
        
        # Convert to grayscale
        gray = cv2.cvtColor(content, cv2.COLOR_BGR2GRAY)
        
        # Try Otsu first
        otsu_thresh, otsu_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Check ink percentage
        ink_percent = (otsu_binary == 0).sum() / otsu_binary.size * 100
        print(f"    Otsu ink percentage: {ink_percent:.1f}%")
        
        if self.MIN_INK_PERCENT <= ink_percent <= self.MAX_INK_PERCENT:
            print("    ✅ Otsu threshold successful")
            return otsu_binary, "otsu"
        
        # Otsu failed, try Sauvola in lower ROI
        print("    ⚠️ Otsu failed, trying Sauvola in lower ROI")
        
        # Use lower 60% of image for Sauvola
        h, w = gray.shape
        lower_roi = gray[int(h * 0.4):, :]
        
        if lower_roi.size > 0:
            # Calculate window size based on ROI size
            window_size = min(25, max(15, min(lower_roi.shape) // 4))
            if window_size % 2 == 0:
                window_size += 1
            
            sauvola_binary = self._sauvola_threshold(lower_roi, window_size)
            
            # Create full-size binary image
            full_binary = np.ones_like(gray) * 255
            full_binary[int(h * 0.4):, :] = sauvola_binary
            
            print("    ✅ Sauvola threshold successful")
            return full_binary, "sauvola"
        
        # Both failed, return Otsu as fallback
        print("    ⚠️ Both methods failed, using Otsu fallback")
        return otsu_binary, "otsu_fallback"
    
    def _sauvola_threshold(self, image: np.ndarray, window_size: int, k: float = 0.2) -> np.ndarray:
        """Sauvola adaptive thresholding"""
        # Convert to float
        img_float = image.astype(np.float64)
        
        # Calculate mean and standard deviation using integral images
        integral = cv2.integral(img_float)
        integral_sq = cv2.integral(img_float ** 2)
        
        h, w = image.shape
        binary = np.zeros_like(image, dtype=np.uint8)
        
        half_window = window_size // 2
        
        for y in range(h):
            for x in range(w):
                # Define window boundaries
                y1 = max(0, y - half_window)
                y2 = min(h, y + half_window + 1)
                x1 = max(0, x - half_window)
                x2 = min(w, x + half_window + 1)
                
                # Calculate mean and std using integral images
                count = (y2 - y1) * (x2 - x1)
                sum_val = integral[y2, x2] - integral[y2, x1] - integral[y1, x2] + integral[y1, x1]
                sum_sq = integral_sq[y2, x2] - integral_sq[y2, x1] - integral_sq[y1, x2] + integral_sq[y1, x1]
                
                mean = sum_val / count
                variance = (sum_sq / count) - (mean ** 2)
                std = np.sqrt(max(variance, 0))
                
                # Sauvola threshold
                threshold = mean * (1 + k * (std / 128 - 1))
                
                binary[y, x] = 0 if image[y, x] < threshold else 255
        
        return binary
    
    def _stroke_aware_morphology(self, binary: np.ndarray, scale_factor: float) -> np.ndarray:
        """Step 3: Stroke-aware morphology"""
        # Estimate stroke width at current scale
        stroke_width = self.TARGET_STROKE * scale_factor
        
        # Create horizontal and vertical kernels
        h_kernel_size = max(1, int(stroke_width * self.MORPH_H_RATIO))
        v_kernel_size = max(1, int(stroke_width * self.MORPH_V_RATIO))
        
        # Ensure odd sizes
        if h_kernel_size % 2 == 0:
            h_kernel_size += 1
        if v_kernel_size % 2 == 0:
            v_kernel_size += 1
        
        # Create kernels
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_size, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_size))
        
        # Apply morphology
        # Close to connect broken strokes
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, h_kernel)
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, v_kernel)
        
        # Open to remove noise
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        
        return opened
    
    def _fast_text_block_detection(self, binary: np.ndarray, content: np.ndarray, scale_factor: float) -> Tuple[List[TextRegion], float]:
        """Step 4: Fast text block decision with early exit"""
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return self._container_fallback(binary, content, scale_factor)
        
        # Filter contours by area
        h, w = content.shape[:2]
        min_area = (h * w) * 0.001  # 0.1% of image area
        max_area = (h * w) * 0.3    # 30% of image area
        
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                valid_contours.append(contour)
        
        if not valid_contours:
            return self._container_fallback(binary, content, scale_factor)
        
        # Group contours into lines
        text_lines = self._group_into_lines(valid_contours)
        
        if len(text_lines) >= self.MIN_LINES_FOR_BLOCK:
            # Convert to text regions
            text_regions = []
            for line in text_lines:
                if line:
                    # Get bounding box of the line
                    x_coords = [c[0] for c in line]
                    y_coords = [c[1] for c in line]
                    
                    x1, x2 = min(x_coords), max(x_coords)
                    y1, y2 = min(y_coords), max(y_coords)
                    
                    # Calculate density
                    area = (x2 - x1) * (y2 - y1)
                    density = len(line) / area if area > 0 else 0
                    
                    text_regions.append(TextRegion(x1, y1, x2, y2, density))
            
            confidence = min(1.0, len(text_regions) / 10.0)  # Cap at 1.0
            return text_regions, confidence
        
        # Not enough lines, use fallback
        return self._container_fallback(binary, content, scale_factor)
    
    def _group_into_lines(self, contours: List[Tuple]) -> List[List[Tuple]]:
        """Group contours into horizontal lines"""
        if not contours:
            return []
        
        # Get bounding boxes
        bboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            center_y = y + h // 2
            bboxes.append((center_y, x, y, w, h))
        
        # Sort by y-coordinate
        bboxes.sort(key=lambda x: x[0])
        
        # Group into lines
        lines = []
        current_line = []
        current_y = bboxes[0][0]
        
        for bbox in bboxes:
            center_y = bbox[0]
            
            # Check if this contour belongs to the current line
            if abs(center_y - current_y) <= 10:  # 10px tolerance
                current_line.append(bbox)
            else:
                if current_line:
                    lines.append(current_line)
                current_line = [bbox]
                current_y = center_y
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def _container_fallback(self, binary: np.ndarray, content: np.ndarray, scale_factor: float) -> Tuple[List[TextRegion], float]:
        """Fallback for text block detection"""
        h, w = content.shape[:2]
        
        # Create a simple text region covering the lower part of the card
        text_region = TextRegion(
            x1=int(w * 0.1),
            y1=int(h * 0.6),
            x2=int(w * 0.9),
            y2=int(h * 0.9),
            density=0.5
        )
        
        return [text_region], 0.3
    
    def _pruned_type_line_detection(self, content: np.ndarray, text_regions: List[TextRegion], scale_factor: float) -> Tuple[int, float, str]:
        """Step 5: Type-line with pruned peaks"""
        h, w = content.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(content, cv2.COLOR_BGR2GRAY)
        
        # Calculate horizontal projection
        projection = np.sum(gray < 128, axis=1)  # Sum of black pixels per row
        
        # Find peaks with prominence
        min_distance = int(h * self.PEAK_MIN_DISTANCE_RATIO)
        peaks = self._find_peaks_with_prominence(projection, min_distance)
        
        if not peaks:
            return self._container_first_fallback(content, h, scale_factor)
        
        # Score candidates
        candidates = []
        for y, prominence in peaks[:self.MAX_TYPE_LINE_CANDIDATES]:
            score = self._score_type_line_candidate(y, prominence, text_regions, h)
            candidates.append((y, score, prominence))
        
        # Sort by score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if candidates:
            best_y, best_score, best_prominence = candidates[0]
            
            # Check if we have a good candidate
            if best_score > 0.5:
                return best_y, best_score, "peak_detection"
        
        # No good candidates, use fallback
        return self._container_first_fallback(content, h, scale_factor)
    
    def _find_peaks_with_prominence(self, signal: np.ndarray, min_distance: int) -> List[Tuple[int, float]]:
        """Find peaks with prominence in a signal"""
        peaks = []
        
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                # Calculate prominence
                left_min = min(signal[max(0, i-min_distance):i])
                right_min = min(signal[i+1:min(len(signal), i+min_distance+1)])
                prominence = signal[i] - max(left_min, right_min)
                
                if prominence > 0:
                    peaks.append((i, prominence))
        
        # Sort by prominence
        peaks.sort(key=lambda x: x[1], reverse=True)
        return peaks
    
    def _score_type_line_candidate(self, y: int, prominence: float, text_regions: List[TextRegion], h: int) -> float:
        """Score a type line candidate"""
        # Base score from prominence
        prominence_score = min(1.0, prominence / 100.0)
        
        # Position score (type line should be in middle-lower half, not too low)
        # Changed from h * 0.7 (70%) to h * 0.55 (55%) to reduce downward bias
        position_score = 1.0 - abs(y - h * 0.55) / (h * 0.4)
        position_score = max(0, position_score)
        
        # Text region proximity score
        proximity_score = 0
        if text_regions:
            # Find closest text region
            min_distance = float('inf')
            for region in text_regions:
                distance = abs(y - region.y1)
                min_distance = min(min_distance, distance)
            
            proximity_score = max(0, 1.0 - min_distance / 100.0)
        
        # Combine scores
        final_score = 0.4 * prominence_score + 0.4 * position_score + 0.2 * proximity_score
        return final_score
    
    def _container_first_fallback(self, content: np.ndarray, h: int, scale_factor: float) -> Tuple[int, float, str]:
        """Fallback for type line detection"""
        # Use a more neutral position instead of 70% (was too low)
        type_y = int(h * 0.55)  # Changed from 0.7 to 0.55
        return type_y, 0.3, "fallback"
    
    def _infer_card_style(self, content: np.ndarray, text_regions: List[TextRegion]) -> str:
        """Infer card style from content and text regions"""
        # Simple heuristic: if we have many text regions, it's likely a complex card
        if len(text_regions) > 5:
            return "complex"
        elif len(text_regions) > 2:
            return "normal"
        else:
            return "simple"
    
    def _infer_regions_from_anchors(self, content: np.ndarray, anchors: LayoutAnchors, card_style: str) -> Dict[str, DetectedRegion]:
        """Infer regions from layout anchors"""
        h, w = content.shape[:2]
        regions = {}
        
        # Title box - position it higher on the card (closer to actual top)
        type_line_height = 45  # Same as type box definition
        title_box_height = type_line_height + 5  # Same as type box: 45 + 5 = 50 pixels
        title_y1 = 5  # Start closer to top (was 10, now 5)
        title_y2 = title_y1 + title_box_height  # End 50 pixels later
        
        regions['title'] = DetectedRegion(
            name='title',
            y1=title_y1,
            y2=title_y2,
            x1=10,
            x2=w - 10,
            confidence=0.9,
            method='anchor_based'
        )
        
        # Mana cost box - top right, EXACT same height as title box
        regions['mana_cost'] = DetectedRegion(
            name='mana_cost',
            y1=title_y1,  # Same start as title box
            y2=title_y2,  # Same end as title box (exact same height)
            x1=w - 80,
            x2=w - 10,
            confidence=0.9,
            method='anchor_based'
        )
        
        # Type line - make it slightly taller
        type_line_height = 45  # Increased from 35 to 45 to make it taller
        regions['type_line'] = DetectedRegion(
            name='type_line',
            y1=anchors.type_line_y - 5,
            y2=anchors.type_line_y + type_line_height,
            x1=10,
            x2=w - 10,
            confidence=anchors.confidence,
            method=anchors.detection_method
        )
        
        # Text box - starts after type line
        regions['text'] = DetectedRegion(
            name='text',
            y1=anchors.type_line_y + type_line_height + 5,
            y2=h - 20,
            x1=10,
            x2=w - 10,
            confidence=0.8,
            method='anchor_based'
        )
        
        # Set symbol - small square, dimensions slightly bigger than type box height
        set_symbol_size = type_line_height + 10  # 45 + 10 = 55 pixels (slightly bigger than type box)
        set_symbol_y1 = anchors.type_line_y - 5  # Align with type line
        set_symbol_y2 = set_symbol_y1 + set_symbol_size
        set_symbol_x1 = w - set_symbol_size - 10  # Right side, accounting for size
        set_symbol_x2 = w - 10
        
        regions['set_symbol'] = DetectedRegion(
            name='set_symbol',
            y1=set_symbol_y1,
            y2=set_symbol_y2,
            x1=set_symbol_x1,
            x2=set_symbol_x2,
            confidence=0.7,
            method='anchor_based'
        )
        
        return regions

    def _measure_border_thickness_robust(self, content: np.ndarray) -> Tuple[float, float, float, float]:
        """Measure border thickness per side robustly (Step 1.1)"""
        h, w = content.shape[:2]
        gray = cv2.cvtColor(content, cv2.COLOR_BGR2GRAY)
        
        # Sample N≈40 points per side
        N = 40
        thicknesses = {'top': [], 'bottom': [], 'left': [], 'right': []}
        
        # Sample top border
        for i in range(N):
            x = int(w * (i + 1) / (N + 1))
            y = 10
            thickness = self._measure_border_at_point_robust(gray, x, y, 'top')
            if thickness > 0:
                thicknesses['top'].append(thickness)
        
        # Sample bottom border
        for i in range(N):
            x = int(w * (i + 1) / (N + 1))
            y = h - 10
            thickness = self._measure_border_at_point_robust(gray, x, y, 'bottom')
            if thickness > 0:
                thicknesses['bottom'].append(thickness)
        
        # Sample left border
        for i in range(N):
            y = int(h * (i + 1) / (N + 1))
            x = 10
            thickness = self._measure_border_at_point_robust(gray, x, y, 'left')
            if thickness > 0:
                thicknesses['left'].append(thickness)
        
        # Sample right border
        for i in range(N):
            y = int(h * (i + 1) / (N + 1))
            x = w - 10
            thickness = self._measure_border_at_point_robust(gray, x, y, 'right')
            if thickness > 0:
                thicknesses['right'].append(thickness)
        
        # Calculate medians and handle outliers
        w_top = np.median(thicknesses['top']) if thicknesses['top'] else 0
        w_bottom = np.median(thicknesses['bottom']) if thicknesses['bottom'] else 0
        w_left = np.median(thicknesses['left']) if thicknesses['left'] else 0
        w_right = np.median(thicknesses['right']) if thicknesses['right'] else 0
        
        # Fallback for sides with insufficient valid samples
        short_side = min(w, h)
        fallback_min = 0.02 * short_side
        fallback_max = 0.05 * short_side
        
        other_sides = []
        for side in ['top', 'bottom', 'left', 'right']:
            if len(thicknesses[side]) >= N * 0.5:  # At least 50% valid samples
                other_sides.append(thicknesses[side])
        
        if other_sides:
            fallback_median = np.median([np.median(side) for side in other_sides])
            fallback_median = np.clip(fallback_median, fallback_min, fallback_max)
        else:
            fallback_median = fallback_min
        
        # Apply fallback for sides with insufficient samples
        if len(thicknesses['top']) < N * 0.5:
            w_top = fallback_median
        if len(thicknesses['bottom']) < N * 0.5:
            w_bottom = fallback_median
        if len(thicknesses['left']) < N * 0.5:
            w_left = fallback_median
        if len(thicknesses['right']) < N * 0.5:
            w_right = fallback_median
        
        return w_top, w_bottom, w_left, w_right
    
    def _measure_border_at_point_robust(self, gray: np.ndarray, x: int, y: int, direction: str) -> float:
        """Measure border thickness at a specific point with robust edge detection"""
        h, w = gray.shape
        
        # Define search directions
        if direction == 'top':
            search_range = range(y, min(y + 50, h))
        elif direction == 'bottom':
            search_range = range(max(0, y - 50), y)
        elif direction == 'left':
            search_range = range(x, min(x + 50, w))
        elif direction == 'right':
            search_range = range(max(0, x - 50), x)
        else:
            return 0
        
        # Look for strong gradient (edge) with adaptive threshold
        for i in search_range:
            if direction in ['top', 'bottom']:
                if i > 0 and i < h - 1:
                    grad = abs(int(gray[i+1, x]) - int(gray[i-1, x]))
                    # Adaptive threshold based on local variance
                    local_region = gray[max(0, i-5):min(h, i+6), max(0, x-5):min(w, x+6)]
                    local_std = np.std(local_region)
                    threshold = max(30, local_std * 2)
                    if grad > threshold:
                        return abs(i - y)
            else:  # left, right
                if i > 0 and i < w - 1:
                    grad = abs(int(gray[y, i+1]) - int(gray[y, i-1]))
                    # Adaptive threshold based on local variance
                    local_region = gray[max(0, y-5):min(h, y+6), max(0, i-5):min(w, i+6)]
                    local_std = np.std(local_region)
                    threshold = max(30, local_std * 2)
                    if grad > threshold:
                        return abs(i - x)
        
        return 0
    
    def _inset_to_content_quad(self, outer_quad: np.ndarray, w_top: float, w_bottom: float, w_left: float, w_right: float) -> np.ndarray:
        """Inset outer quad to content quad (Step 1.2)"""
        # Create a copy to avoid modifying the original
        content_quad = outer_quad.copy()
        
        # Move each side inward by its border thickness
        # Top side (lowest Y coordinates)
        top_indices = np.where(outer_quad[:, 1] == np.min(outer_quad[:, 1]))[0]
        for idx in top_indices:
            content_quad[idx, 1] += w_top  # Move top down
        
        # Bottom side (highest Y coordinates)
        bottom_indices = np.where(outer_quad[:, 1] == np.max(outer_quad[:, 1]))[0]
        for idx in bottom_indices:
            content_quad[idx, 1] -= w_bottom  # Move bottom up
        
        # Left side (lowest X coordinates)
        left_indices = np.where(outer_quad[:, 0] == np.min(outer_quad[:, 0]))[0]
        for idx in left_indices:
            content_quad[idx, 0] += w_left  # Move left right
        
        # Right side (highest X coordinates)
        right_indices = np.where(outer_quad[:, 0] == np.max(outer_quad[:, 0]))[0]
        for idx in right_indices:
            content_quad[idx, 0] -= w_right  # Move right left
        
        return content_quad
    
    def _compute_canonical_homography(self, content_quad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute homography to canonical space (Step 1.3)"""
        # Canonical dimensions: Wc=630, Hc=880 (63×88 ratio)
        Wc, Hc = 630, 880
        
        # Define canonical quad (rectangle)
        canonical_quad = np.array([
            [0, 0],      # Top-left
            [Wc, 0],     # Top-right
            [Wc, Hc],    # Bottom-right
            [0, Hc]      # Bottom-left
        ], dtype=np.float32)
        
        # Compute homography from content quad to canonical quad
        H_content = cv2.getPerspectiveTransform(content_quad.astype(np.float32), canonical_quad)
        
        # Compute inverse homography for converting back
        H_content_inv = cv2.getPerspectiveTransform(canonical_quad, content_quad.astype(np.float32))
        
        return H_content, H_content_inv

    def _pruned_type_line_detection_canonical(self, content: np.ndarray, text_regions: List[TextRegion], scale_factor: float) -> Tuple[int, float, str]:
        """Robust type-line detection in canonical space using edge pairs"""
        h, w = content.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(content, cv2.COLOR_BGR2GRAY)
        
        # Search window: y ∈ [0.52, 0.58] × Hc
        band_min = int(h * 0.52)
        band_max = int(h * 0.58)
        
        print(f"    Searching type line in canonical band: y ∈ [{band_min}, {band_max}] (52-58% of content height)")
        
        # Step 1: Compute Scharr vertical gradient
        Gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        Gy = np.abs(Gy)
        
        # Step 2: Find peaks with polarity in the search band
        edge_profile = np.mean(Gy[band_min:band_max, :], axis=1)  # Average gradient across width
        
        # Find peaks with prominence
        min_distance = max(3, int(len(edge_profile) * 0.02))  # At least 3px, or 2% of band
        peaks = self._find_peaks_with_prominence(edge_profile, min_distance)
        
        if len(peaks) < 2:
            print("    ⚠️  Insufficient peaks found, using fallback")
            return self._container_first_fallback_canonical(content, h, scale_factor)
        
        # Step 3: Form pairs (p, q) with specific gap and low horizontal edge energy
        pairs = []
        for i, (p_rel, p_prominence) in enumerate(peaks):
            p_abs = p_rel + band_min
            
            for j, (q_rel, q_prominence) in enumerate(peaks[i+1:], i+1):
                q_abs = q_rel + band_min
                
                # Check gap constraint: 3-15 pixels between edges
                gap = abs(q_abs - p_abs)
                if gap < 3 or gap > 15:
                    continue
                
                # Check horizontal edge energy between the pair
                y1, y2 = min(p_abs, q_abs), max(p_abs, q_abs)
                horizontal_energy = self._compute_horizontal_edge_energy(gray, y1, y2, w)
                
                # Low horizontal energy indicates a clean type bar
                if horizontal_energy < 20:  # Threshold for clean type bar
                    # Score this pair
                    pair_score = self._score_edge_pair(p_abs, q_abs, p_prominence, q_prominence, gap, horizontal_energy, h)
                    pairs.append((p_abs, q_abs, pair_score, p_prominence, q_prominence))
        
        if not pairs:
            print("    ⚠️  No valid edge pairs found, using fallback")
            return self._container_first_fallback_canonical(content, h, scale_factor)
        
        # Step 4: Choose the best pair and return the upper edge
        pairs.sort(key=lambda x: x[2], reverse=True)  # Sort by score
        best_p, best_q, best_score, p_prominence, q_prominence = pairs[0]
        
        # Return the upper edge (smaller y coordinate)
        upper_edge = min(best_p, best_q)
        
        # Step 5: Local snap (±3 px) to strongest negative gradient
        snapped_y = self._local_snap_to_gradient(gray, upper_edge, w)
        
        # Step 6: Sanity rails
        snapped_y = self._apply_sanity_rails(snapped_y, text_regions, h)
        
        print(f"    ✅ Found type line at y={snapped_y} (upper edge of best pair) with score={best_score:.3f}")
        return snapped_y, best_score, "canonical_edge_pair_detection"
    
    def _compute_horizontal_edge_energy(self, gray: np.ndarray, y1: int, y2: int, w: int) -> float:
        """Compute horizontal edge energy between two y-coordinates"""
        # Extract the region between the two edges
        region = gray[y1:y2, :]
        
        # Compute horizontal gradient
        Gx = cv2.Scharr(region, cv2.CV_64F, 1, 0)
        Gx = np.abs(Gx)
        
        # Return mean horizontal edge energy
        return float(np.mean(Gx))
    
    def _score_edge_pair(self, p: int, q: int, p_prominence: float, q_prominence: float, gap: int, horizontal_energy: float, h: int) -> float:
        """Score an edge pair (p, q) for type bar detection"""
        # Gap score: prefer gaps around 6-10 pixels
        gap_score = 1.0 - abs(gap - 8) / 8.0  # Optimal at 8px, penalize deviation
        gap_score = max(0, gap_score)
        
        # Prominence score: average of both edges
        prominence_score = (p_prominence + q_prominence) / 2.0 / 100.0  # Normalize
        prominence_score = min(1.0, prominence_score)
        
        # Position score: prefer pairs centered around 55% of content height
        center_y = (p + q) / 2
        position_score = 1.0 - abs(center_y - h * 0.55) / (h * 0.06)  # Target 55%, tolerance ±3%
        position_score = max(0, position_score)
        
        # Horizontal energy score: lower is better (cleaner type bar)
        energy_score = max(0, 1.0 - horizontal_energy / 50.0)  # Normalize to 0-1
        
        # Combine scores
        final_score = 0.3 * gap_score + 0.3 * prominence_score + 0.25 * position_score + 0.15 * energy_score
        return final_score
    
    def _local_snap_to_gradient(self, gray: np.ndarray, y: int, w: int) -> int:
        """Local snap (±3 px) to strongest negative gradient"""
        h = gray.shape[0]
        search_range = 3
        
        # Define search window
        y_min = max(0, y - search_range)
        y_max = min(h - 1, y + search_range + 1)
        
        best_y = y
        best_gradient = 0
        
        # Compute vertical gradient
        Gy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        
        # Search for strongest negative gradient (dark edge)
        for test_y in range(y_min, y_max):
            # Average gradient across the width
            avg_gradient = np.mean(Gy[test_y, :])
            
            # We want negative gradient (dark edge)
            if avg_gradient < best_gradient:
                best_gradient = avg_gradient
                best_y = test_y
        
        return best_y
    
    def _apply_sanity_rails(self, y: int, text_regions: List[TextRegion], h: int) -> int:
        """Apply sanity rails to type line position"""
        # If text-box top is too close to type line, snap up 2 px
        if text_regions:
            closest_text_top = min(region.y1 for region in text_regions)
            if closest_text_top - y < 10:  # Too close
                y = max(0, y - 2)
                print(f"    🔧 Sanity rail: moved type line up 2px to avoid text box")
        
        return y
    
    def _container_first_fallback_canonical(self, content: np.ndarray, h: int, scale_factor: float) -> Tuple[int, float, str]:
        """Fallback for canonical type line detection"""
        # Use the canonical target position (54% of content height)
        type_y = int(h * 0.54)
        print(f"    Using canonical fallback at y={type_y} (54% of content height)")
        return type_y, 0.3, "canonical_fallback"

def create_enhanced_detection_visualization(image_path: str, results: List[Tuple[DetectedCard, np.ndarray, Dict[str, DetectedRegion], np.ndarray, Dict]], output_dir: str = "output"):
    """Create visualization for enhanced detector results"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (detected_card, content, regions, scaled_canonical, canonical_coords) in enumerate(results):
        # Create visualization
        vis_image = content.copy()
        
        # Draw regions
        colors = {
            'title': (0, 255, 0),      # Green
            'type_line': (255, 0, 0),  # Red
            'text': (0, 0, 255),       # Blue
            'mana_cost': (255, 255, 0), # Yellow
            'set_symbol': (255, 0, 255) # Magenta
        }
        
        for region_name, region in regions.items():
            if region.exists:
                color = colors.get(region_name, (128, 128, 128))
                cv2.rectangle(vis_image, (region.x1, region.y1), (region.x2, region.y2), color, 2)
                
                # Add label
                label = f"{region_name} ({region.confidence:.2f})"
                cv2.putText(vis_image, label, (region.x1, region.y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add card info
        info_text = f"Style: {detected_card.card_style}, Confidence: {detected_card.confidence:.2f}"
        cv2.putText(vis_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add canonical coordinates info
        if canonical_coords:
            y_content_pct = canonical_coords.get('y_content_pct', 0)
            info_text2 = f"Content %: {y_content_pct:.3f}"
            cv2.putText(vis_image, info_text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save visualization
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_enhanced_detection.jpg")
        cv2.imwrite(output_path, vis_image)
        
        print(f"✅ Saved enhanced detection visualization: {output_path}")

def test_enhanced_detector():
    """Test the enhanced detector"""
    detector = MTGCardDetectorEnhanced()
    
    # Test with a sample image
    test_image = "test photos/Screenshot 2025-07-27 at 8.08.30 PM.png"
    if os.path.exists(test_image):
        results = detector.detect_cards_adaptive(test_image)
        if results:
            create_enhanced_detection_visualization(test_image, results)
        else:
            print("❌ No cards detected")
    else:
        print(f"❌ Test image not found: {test_image}")

if __name__ == "__main__":
    test_enhanced_detector() 