#!/usr/bin/env python3
"""
Improved MTG Mana Detector
Implements lightweight fixes for small-scale detection based on analysis findings
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import easyocr

@dataclass
class ManaSymbol:
    """Represents a detected mana symbol"""
    symbol: str
    confidence: float
    center: Tuple[int, int]
    radius: int
    color: str = ""

@dataclass
class ManaResult:
    """Result of mana detection"""
    mana_string: str
    confidence: float
    processing_time_ms: float
    symbols: List[ManaSymbol]
    detection_method: str

class ImprovedManaDetector:
    """Improved mana detector with scale-adaptive detection"""
    
    def __init__(self, use_ocr: bool = True, debug: bool = False):
        self.use_ocr = use_ocr
        self.debug = debug
        
        # Initialize OCR if needed
        if use_ocr:
            self.ocr_reader = easyocr.Reader(['en'], gpu=False)
        
        # Scale thresholds from analysis
        self.min_working_scale = 3.0
        self.small_scale_threshold = 120  # pixels height
        
        # Color classification thresholds
        self.color_thresholds = {
            'R': {'hue_range': [(0, 10), (170, 180)], 'min_sat': 40, 'min_val': 50},
            'U': {'hue_range': [(100, 140)], 'min_sat': 40, 'min_val': 50},
            'G': {'hue_range': [(50, 90)], 'min_sat': 40, 'min_val': 50},
            'B': {'hue_range': [(110, 130)], 'min_sat': 30, 'min_val': 30},
            'W': {'hue_range': [(0, 180)], 'min_sat': 10, 'max_val': 255, 'min_val': 200}
        }
        
        # LAB prototypes for ΔE₀₀ tie-breaker
        self.lab_prototypes = {
            'W': {'L': 90, 'a': 0, 'b': 0},  # Pure white
            'G': {'L': 60, 'a': -20, 'b': 30}  # Typical green
        }
        
        # Border reference cache
        self.border_cache = {}
    
    def detect_mana_cost(self, image: np.ndarray, mana_region: Optional[Tuple[int, int, int, int]] = None) -> ManaResult:
        """Detect mana cost with scale-adaptive approach"""
        start_time = time.time()
        
        # Extract ROI
        if mana_region:
            x1, y1, x2, y2 = mana_region
            roi = image[y1:y2, x1:x2]
        else:
            h, w = image.shape[:2]
            # Use wider ROI based on analysis findings
            roi = image[int(h * 0.05):int(h * 0.25), int(w * 0.5):w]
        
        roi_h, roi_w = roi.shape[:2]
        
        # Determine detection method based on scale
        if roi_h < self.small_scale_threshold:
            # Small scale: use integral image ring detector
            symbols = self._detect_small_scale(roi, image, mana_region)
            detection_method = "integral_ring"
        else:
            # Normal scale: use traditional Hough circles
            symbols = self._detect_normal_scale(roi)
            detection_method = "hough_circles"
        
        # If no symbols found, try scale up
        if not symbols and roi_h < self.small_scale_threshold:
            scaled_roi = cv2.resize(roi, (int(roi_w * self.min_working_scale), int(roi_h * self.min_working_scale)), 
                                  interpolation=cv2.INTER_CUBIC)
            symbols = self._detect_normal_scale(scaled_roi)
            detection_method = "scaled_hough"
        
        # Classify symbols
        classified_symbols = []
        for symbol in symbols:
            classified_symbol = self._classify_symbol(roi, symbol, image, mana_region)
            classified_symbols.append(classified_symbol)
        
        # Build mana string
        mana_string = self._build_mana_string(classified_symbols)
        confidence = np.mean([s.confidence for s in classified_symbols]) if classified_symbols else 0.0
        
        processing_time = (time.time() - start_time) * 1000
        
        return ManaResult(
            mana_string=mana_string,
            confidence=confidence,
            processing_time_ms=processing_time,
            symbols=classified_symbols,
            detection_method=detection_method
        )
    
    def _detect_small_scale(self, roi: np.ndarray, image: np.ndarray, mana_region: Optional[Tuple[int, int, int, int]]) -> List[ManaSymbol]:
        """Integral image ring detector for small scales with improved band detection"""
        h, w = roi.shape[:2]
        symbols = []
        
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        # Compute integral image for fast ring detection
        integral = cv2.integral(gray)
        
        # Test different radii (3-8 pixels for small scales)
        radii = range(3, min(9, min(h//4, w//4)))
        
        # Two-threshold detection: high threshold to seed peaks, lower threshold for neighbors
        high_threshold = 15
        low_threshold = 10
        
        # First pass: find high-confidence peaks
        high_peaks = []
        for radius in radii:
            for y in range(radius, h-radius, 2):
                for x in range(radius, w-radius, 2):
                    # Band coverage: widen search band to 16-23% of card height
                    y_ratio = y / h
                    if not (0.16 <= y_ratio <= 0.23):
                        continue
                    
                    # Compute ring score using integral image
                    score = self._compute_ring_score(integral, x, y, radius)
                    
                    if score > high_threshold:
                        # Check if this is a local maximum
                        if self._is_local_maximum(gray, x, y, radius):
                            high_peaks.append((x, y, radius, score))
        
        # Second pass: allow neighbors above lower threshold to join if they form distinct local maxima
        all_peaks = high_peaks.copy()
        
        for radius in radii:
            for y in range(radius, h-radius, 2):
                for x in range(radius, w-radius, 2):
                    # Band coverage with ±2% vertical jitter around fitted band
                    y_ratio = y / h
                    if not (0.16 <= y_ratio <= 0.23):
                        continue
                    
                    # Skip if already a high peak
                    if any(abs(x - px) < 5 and abs(y - py) < 5 for px, py, _, _ in high_peaks):
                        continue
                    
                    # Compute ring score
                    score = self._compute_ring_score(integral, x, y, radius)
                    
                    if score > low_threshold:
                        # Check if this forms a distinct local maximum ≥0.9R away from existing pips
                        too_close = False
                        for px, py, pr, _ in all_peaks:
                            distance = np.sqrt((x - px)**2 + (y - py)**2)
                            min_radius = min(radius, pr)
                            if distance < 0.9 * min_radius:
                                too_close = True
                                break
                        
                        if not too_close and self._is_local_maximum(gray, x, y, radius):
                            all_peaks.append((x, y, radius, score))
        
        # Convert peaks to symbols
        for x, y, radius, score in all_peaks:
            symbols.append(ManaSymbol(
                symbol="",  # Will be classified later
                confidence=min(score / 100.0, 1.0),
                center=(x, y),
                radius=radius
            ))
        
        # Remove duplicates with improved NMS
        symbols = self._remove_duplicates_improved(symbols)
        
        return symbols[:6]  # Limit to top 6 (max count)
    
    def _compute_ring_score(self, integral: np.ndarray, x: int, y: int, radius: int) -> float:
        """Compute ring score using integral image"""
        # Inner disk radius
        inner_r = int(radius * 0.7)
        outer_r = int(radius * 1.3)
        
        # Get sums using integral image
        def get_sum(x1, y1, x2, y2):
            return integral[y2, x2] - integral[y2, x1] - integral[y1, x2] + integral[y1, x1]
        
        # Inner disk
        inner_sum = get_sum(x-inner_r, y-inner_r, x+inner_r, y+inner_r)
        inner_area = (2*inner_r + 1) ** 2
        inner_mean = inner_sum / inner_area
        
        # Outer ring
        outer_sum = get_sum(x-outer_r, y-outer_r, x+outer_r, y+outer_r) - inner_sum
        outer_area = (2*outer_r + 1) ** 2 - inner_area
        outer_mean = outer_sum / outer_area
        
        # Score: outer should be brighter than inner
        return outer_mean - inner_mean
    
    def _is_local_maximum(self, gray: np.ndarray, x: int, y: int, radius: int) -> bool:
        """Check if point is a local maximum in its neighborhood"""
        h, w = gray.shape
        center_val = gray[y, x]
        
        # Check 3x3 neighborhood
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if gray[ny, nx] > center_val:
                        return False
        
        return True
    
    def _detect_normal_scale(self, roi: np.ndarray) -> List[ManaSymbol]:
        """Traditional Hough circle detection for normal scales"""
        h, w = roi.shape[:2]
        symbols = []
        
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        # Scale-adaptive parameters
        min_radius = max(6, h // 20)
        max_radius = h // 8
        
        # Detect circles
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=min_radius*2,
            param1=50, param2=25, minRadius=min_radius, maxRadius=max_radius
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                symbols.append(ManaSymbol(
                    symbol="",
                    confidence=0.8,  # Default confidence for Hough
                    center=(x, y),
                    radius=r
                ))
        
        return symbols
    
    def _classify_symbol(self, roi: np.ndarray, symbol: ManaSymbol, image: np.ndarray = None, mana_region: Optional[Tuple[int, int, int, int]] = None) -> ManaSymbol:
        """Classify symbol using improved color analysis with proper digit routing"""
        x, y = symbol.center
        radius = symbol.radius
        
        # Extract region around symbol
        h, w = roi.shape[:2]
        x1, y1 = max(0, x-radius), max(0, y-radius)
        x2, y2 = min(w, x+radius), min(h, y+radius)
        
        symbol_region = roi[y1:y2, x1:x2]
        
        if symbol_region.size == 0:
            return symbol
        
        # Sample annulus only (ignore center icon)
        color = self._classify_color_annulus(symbol_region, radius, image, mana_region)
        
        # Handle digit candidates and routing
        if color == "digit_candidate":
            # Route to digit recognition
            digit_result = self._check_digit_presence(symbol_region, radius)
            if digit_result and digit_result != "unknown":
                color = digit_result
            else:
                color = "unknown"  # Keep as unknown if digit recognition fails
        
        # Digit routing guard: if (C* ≤ 12 and 45 ≤ L* ≤ 80) or ΔL < 8 → digit path before color
        if color in ['W', 'B', 'C'] and self._should_check_digit_first(symbol_region, radius):
            digit_result = self._check_digit_presence(symbol_region, radius)
            if digit_result and digit_result != "unknown":
                color = digit_result
        
        # If color classification fails, try OCR for digits
        if color == "unknown" and self.use_ocr:
            digit = self._ocr_digit(symbol_region)
            if digit:
                color = digit
        
        return ManaSymbol(
            symbol=color,
            confidence=symbol.confidence,
            center=symbol.center,
            radius=symbol.radius,
            color=color
        )
    
    def _should_check_digit_first(self, region: np.ndarray, radius: int) -> bool:
        """Check if we should route to digit recognition first based on center entropy"""
        # Crop center region (≤0.55R)
        h, w = region.shape[:2]
        center_x, center_y = w // 2, h // 2
        crop_radius = int(radius * 0.55)
        
        x1 = max(0, center_x - crop_radius)
        y1 = max(0, center_y - crop_radius)
        x2 = min(w, center_x + crop_radius)
        y2 = min(h, center_y + crop_radius)
        
        center_crop = region[y1:y2, x1:x2]
        
        if center_crop.size == 0:
            return False
        
        # Convert to grayscale
        if len(center_crop.shape) == 3:
            gray = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = center_crop
        
        # Compute entropy
        entropy = self._compute_entropy(gray)
        
        # Count projection peaks
        projection_peaks = self._count_projection_peaks(gray)
        
        # Trigger digit check if entropy ≥ 2.0 or peaks ≥ 3
        return entropy >= 2.0 or projection_peaks >= 3
    
    def _convert_to_lab_proper(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR image to LAB with proper scaling"""
        # OpenCV's LAB conversion uses different scaling:
        # L: 0-255 (should be 0-100)
        # a: 0-255 (should be -128 to +127) 
        # b: 0-255 (should be -128 to +127)
        
        # Convert BGR to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Scale to proper LAB ranges
        lab_scaled = lab.astype(np.float32)
        lab_scaled[:, :, 0] = lab_scaled[:, :, 0] * 100.0 / 255.0  # L: 0-255 -> 0-100
        lab_scaled[:, :, 1] = lab_scaled[:, :, 1] - 128.0  # a: 0-255 -> -128 to +127
        lab_scaled[:, :, 2] = lab_scaled[:, :, 2] - 128.0  # b: 0-255 -> -128 to +127
        
        return lab_scaled
    
    def _classify_color_annulus(self, region: np.ndarray, radius: int, image: np.ndarray = None, mana_region: Optional[Tuple[int, int, int, int]] = None) -> str:
        """Classify color using improved annulus sampling with arc-matched ΔL"""
        h, w = region.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Sample ring annulus only (0.65-0.85R)
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        inner_r = int(radius * 0.65)
        outer_r = int(radius * 0.85)
        ring_mask = (distances >= inner_r) & (distances <= outer_r)
        
        if not np.any(ring_mask):
            return "unknown"
        
        # Sample outside reference annulus (0.95-1.15R)
        outer_inner = int(radius * 0.95)
        outer_outer = int(radius * 1.15)
        outside_mask = (distances >= outer_inner) & (distances <= outer_outer)
        
        # Extract pixels
        ring_pixels = region[ring_mask]
        outside_pixels = region[outside_mask] if np.any(outside_mask) else None
        
        if len(ring_pixels) == 0:
            return "unknown"
        
        # Convert to LAB with proper scaling
        lab_ring = self._convert_to_lab_proper(ring_pixels.reshape(-1, 1, 3))
        lab_ring = lab_ring.reshape(-1, 3)
        
        # Get arc-matched ΔL using improved background sampling
        delta_l, delta_c = self._compute_arc_matched_deltas(region, center_x, center_y, radius)
        
        # Use arc voting for robust classification
        color, confidence = self._classify_with_arc_voting_improved(ring_pixels, lab_ring, delta_l, delta_c, image, mana_region)
        
        # If still ambiguous, check for digit presence
        if color in ["W", "digit_candidate"]:
            digit_result = self._check_digit_presence(region, radius)
            if digit_result and digit_result != "unknown":
                color = digit_result
            elif color == "digit_candidate":
                color = "unknown"
        
        # Fallback to HSV if LAB fails
        if color == "unknown":
            hsv_pixels = cv2.cvtColor(ring_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
            hsv_pixels = hsv_pixels.reshape(-1, 3)
            
            h_median = np.median(hsv_pixels[:, 0])
            s_median = np.median(hsv_pixels[:, 1])
            v_median = np.median(hsv_pixels[:, 2])
            
            color = self._classify_hsv_color_improved(h_median, s_median, v_median)
        
        return color
    
    def _compute_arc_matched_deltas(self, region: np.ndarray, center_x: int, center_y: int, radius: int) -> Tuple[float, float]:
        """Compute arc-matched ΔL and ΔC for more accurate background reference"""
        h, w = region.shape[:2]
        
        # Define arc segments (12 arcs)
        num_arcs = 12
        arc_deltas = []
        
        for arc_idx in range(num_arcs):
            start_angle = arc_idx * 360 / num_arcs
            end_angle = (arc_idx + 1) * 360 / num_arcs
            
            # Sample ring arc (0.65-0.85R)
            ring_arc_pixels = self._sample_arc_pixels(region, center_x, center_y, radius, 
                                                    start_angle, end_angle, 0.65, 0.85)
            
            # Sample corresponding outside arc (0.95-1.15R)
            outside_arc_pixels = self._sample_arc_pixels(region, center_x, center_y, radius,
                                                       start_angle, end_angle, 0.95, 1.15)
            
            if len(ring_arc_pixels) > 0 and len(outside_arc_pixels) > 0:
                # Convert to LAB
                lab_ring_arc = self._convert_to_lab_proper(ring_arc_pixels.reshape(-1, 1, 3))
                lab_outside_arc = self._convert_to_lab_proper(outside_arc_pixels.reshape(-1, 1, 3))
                
                # Compute arc-specific deltas
                l_ring = np.median(lab_ring_arc[:, :, 0])
                l_outside = np.median(lab_outside_arc[:, :, 0])
                c_ring = np.sqrt(np.median(lab_ring_arc[:, :, 1])**2 + np.median(lab_ring_arc[:, :, 2])**2)
                c_outside = np.sqrt(np.median(lab_outside_arc[:, :, 1])**2 + np.median(lab_outside_arc[:, :, 2])**2)
                
                arc_delta_l = l_ring - l_outside
                arc_delta_c = c_ring - c_outside
                
                arc_deltas.append((arc_delta_l, arc_delta_c))
        
        # Use median of arc deltas for robustness
        if arc_deltas:
            delta_l = np.median([d[0] for d in arc_deltas])
            delta_c = np.median([d[1] for d in arc_deltas])
        else:
            delta_l = 0
            delta_c = 0
        
        return delta_l, delta_c
    
    def _sample_arc_pixels(self, region: np.ndarray, center_x: int, center_y: int, radius: int,
                          start_angle: float, end_angle: float, inner_ratio: float, outer_ratio: float) -> np.ndarray:
        """Sample pixels in a specific arc region"""
        h, w = region.shape[:2]
        pixels = []
        
        inner_r = int(radius * inner_ratio)
        outer_r = int(radius * outer_ratio)
        
        # Sample points in the arc
        for r in range(inner_r, outer_r + 1):
            for angle in np.linspace(start_angle, end_angle, 10):
                rad = np.radians(angle)
                x = int(center_x + r * np.cos(rad))
                y = int(center_y + r * np.sin(rad))
                
                if 0 <= x < w and 0 <= y < h:
                    pixels.append(region[y, x])
        
        return np.array(pixels) if pixels else np.array([])
    
    def _classify_with_arc_voting_improved(self, ring_pixels: np.ndarray, lab_ring: np.ndarray, delta_l: float, delta_c: float, image: np.ndarray = None, mana_region: Optional[Tuple[int, int, int, int]] = None) -> Tuple[str, float]:
        """Classify color using improved arc voting with HSV safety net and enhanced arc voting bias"""
        
        # Get polar coordinates for each pixel
        h, w = ring_pixels.shape[:2] if len(ring_pixels.shape) == 3 else (1, ring_pixels.shape[0])
        center_x, center_y = w // 2, h // 2
        
        # For simplicity, use the median LAB values for the entire ring
        l_median = np.median(lab_ring[:, 0])
        a_median = np.median(lab_ring[:, 1])
        b_median = np.median(lab_ring[:, 2])
        chroma = np.sqrt(a_median**2 + b_median**2)
        
        # Calculate hue angle in degrees
        hue_angle = np.degrees(np.arctan2(b_median, a_median))
        
        # Log LAB values for diagnostic analysis
        self._log_lab_diagnostics(l_median, a_median, b_median, chroma, hue_angle, delta_l, delta_c)
        
        # Classify using improved LAB thresholds with HSV safety net
        color, confidence = self._classify_lab_improved_v3(l_median, a_median, b_median, chroma, hue_angle, delta_l, delta_c)
        
        # ENHANCED ARC VOTING BIAS: If classified as G and low chroma, check per-arc whiteness scores
        if color == 'G' and chroma < 12 and image is not None:  # Very conservative chroma threshold
            # Get border reference for whiteness score
            l_border_ref = self._get_border_reference(image, mana_region)
            
            # Compute whiteness score: S_w = 0.5·(L*/L_border_ref) + 0.5·(1 − C*/14)
            whiteness_score = 0.5 * (l_median / l_border_ref) + 0.5 * (1 - chroma / 14)
            
            # Only bias toward white if whiteness score is very high
            if whiteness_score >= 0.9:
                if self.debug:
                    print(f"    → W (ARC VOTING BIAS: S_w={whiteness_score:.3f} ≥ 0.9, C*={chroma:.1f})")
                return "W", 0.8
        
        return color, confidence
    
    def _compute_arc_whiteness_scores(self, ring_pixels: np.ndarray, l_border_ref: float) -> List[float]:
        """Compute whiteness scores for each arc segment"""
        # For simplicity, use the overall ring statistics
        # In a full implementation, this would sample each arc separately
        
        # Convert to LAB if needed
        if len(ring_pixels.shape) == 3:
            lab_pixels = self._convert_to_lab_proper(ring_pixels.reshape(-1, 1, 3))
            lab_pixels = lab_pixels.reshape(-1, 3)
        else:
            lab_pixels = ring_pixels
        
        # Calculate overall statistics
        l_median = np.median(lab_pixels[:, 0])
        chroma = np.sqrt(np.median(lab_pixels[:, 1])**2 + np.median(lab_pixels[:, 2])**2)
        
        # Compute whiteness score: S_w = 0.5·(L*/L_border_ref) + 0.5·(1 − C*/14)
        whiteness_score = 0.5 * (l_median / l_border_ref) + 0.5 * (1 - chroma / 14)
        
        # Return the same score for all 12 arcs (simplified)
        # In full implementation, would compute per-arc scores
        return [whiteness_score] * 12
    
    def _classify_lab_improved_v3(self, l: float, a: float, b: float, chroma: float, hue_angle: float, delta_l: float, delta_c: float) -> Tuple[str, float]:
        """Improved LAB classification with FINAL surgical fixes to eliminate G→B"""
        
        # Debug output
        if self.debug:
            print(f"  LAB Debug: L={l:.1f}, a={a:.1f}, b={b:.1f}, C={chroma:.1f}, θ={hue_angle:.1f}, ΔL={delta_l:.1f}")
        
        # FINAL HARD OVERRIDE: If in green hue range AND not white, NEVER classify as black
        if 60 <= hue_angle <= 160 and not (l >= 82 and chroma <= 12):  # Exclude white
            # Force green classification with maximum bonus
            margin = min(chroma - 8, abs(hue_angle - 60), abs(hue_angle - 160))
            green_score = chroma/40 + margin/50 + 1.5  # Maximum override bonus
            if self.debug:
                print(f"    → G (FINAL HARD OVERRIDE: θ={hue_angle:.1f} in [60, 160], not white)")
            return "G", green_score
        
        # FINAL surgical color classification with proper gate order and ΔL requirements
        color_scores = {}
        
        # White (W) – dynamic reference with border normalization - SURGICAL FIX
        if l >= 82 and chroma <= 12 and delta_l >= 8:
            # Dynamic white reference (simplified - would need card border sampling)
            score_white = (l/100) + (1 - chroma/12) + min(delta_l/20, 1.0)
            color_scores['W'] = score_white
        
        # Black (B) – FINAL strict disqualifier - SURGICAL FIX
        # Hard black disqualifier: if ΔL > -6, black cannot fire
        if l <= 26 and delta_l <= -12 and chroma <= 8 and delta_l <= -6:
            margin = (26 - l) + abs(delta_l + 12) + (8 - chroma)
            color_scores['B'] = 0.8 + margin/50
        
        # Green (G) – ULTRA STRONG override when not darker than outside - SURGICAL FIX
        if 60 <= hue_angle <= 160 and (chroma >= 8 or delta_l >= -2):  # More lenient
            margin = min(chroma - 8, abs(hue_angle - 60), abs(hue_angle - 160))
            # Override B when ΔL ≥ -2 even if L* is modest
            if delta_l >= -2:
                color_scores['G'] = chroma/40 + margin/50 + 1.0  # Ultra strong override bonus
            else:
                color_scores['G'] = chroma/40 + margin/50 + 0.5  # Still strong bonus
        
        # Blue (U)
        if (-150 <= hue_angle <= -70 or hue_angle >= 200) and chroma >= 14:
            margin = min(chroma - 14, abs(hue_angle + 150), abs(hue_angle + 70))
            color_scores['U'] = chroma/40 + margin/50
        
        # Red (R) - R vs W guard - SURGICAL FIX
        if -15 <= hue_angle <= 65 and chroma >= 14 and 35 <= l <= 85:
            margin = min(chroma - 14, abs(hue_angle + 15), abs(hue_angle - 65), l - 35, 85 - l)
            color_scores['R'] = chroma/40 + margin/50
        
        # Digit candidate: if (C* ≤ 10 and 45 ≤ L* ≤ 80) or ΔL < 6, treat as digit-candidate
        if (chroma <= 10 and 45 <= l <= 80) or delta_l < 6:
            color_scores['digit_candidate'] = 0.5
        
        # Colorless (C) - IMPROVED restrictive gate - SURGICAL FIX
        # Only allow {C} when digit path fails and L* ≤ 40 and C* ≤ 8 and ΔL ≤ 0
        if l <= 40 and chroma <= 8 and delta_l <= 0:
            margin = (40 - l) + (8 - chroma) + abs(delta_l)
            color_scores['C'] = 0.6 + margin/50
        
        # Find best color
        if color_scores:
            best_color = max(color_scores, key=color_scores.get)
            best_score = color_scores[best_color]
            
            # WHITENESS OVERRIDE FAILSAFE: If classified as G, check if it should be W
            if best_color == 'G':
                # Get improved border reference
                l_border_ref = self._get_border_reference(image, mana_region)
                
                # Check if this is obvious green (high chroma + green hue)
                is_obvious_green = (chroma >= 18) and (90 <= hue_angle <= 140)
                
                # Whiteness override 2.0: if L* ≥ 0.85·L_border_ref and C* ≤ 12 and ΔL ≥ 8, flip G → W
                if not is_obvious_green and l >= 0.85 * l_border_ref and chroma <= 12 and delta_l >= 8:
                    if self.debug:
                        print(f"    → W (WHITENESS OVERRIDE 2.0: L={l:.1f} ≥ 0.85×{l_border_ref:.1f}, C={chroma:.1f} ≤ 12, ΔL={delta_l:.1f} ≥ 8)")
                    return "W", 0.9  # High confidence white
            
            # ΔE₀₀ TIE-BREAKER: Apply after color classification
            if best_color in ['G', 'W']:
                # Calculate ΔE₀₀ to prototypes
                de_w = self._delta_e00(l, a, b, self.lab_prototypes['W']['L'], self.lab_prototypes['W']['a'], self.lab_prototypes['W']['b'])
                de_g = self._delta_e00(l, a, b, self.lab_prototypes['G']['L'], self.lab_prototypes['G']['a'], self.lab_prototypes['G']['b'])
                
                # Check if this is obvious green
                is_obvious_green = (chroma >= 18) and (90 <= hue_angle <= 140)
                
                # Apply ΔE₀₀ tie-breaker
                if best_color == 'G' and not is_obvious_green and (de_w + 4 <= de_g):
                    if self.debug:
                        print(f"    → W (ΔE₀₀ TIE-BREAKER: ΔE₀₀(W)={de_w:.1f} + 4 ≤ ΔE₀₀(G)={de_g:.1f})")
                    return "W", 0.85
                elif best_color == 'W' and (de_g + 4 <= de_w):
                    if self.debug:
                        print(f"    → G (ΔE₀₀ TIE-BREAKER: ΔE₀₀(G)={de_g:.1f} + 4 ≤ ΔE₀₀(W)={de_w:.1f})")
                    return "G", 0.85
        
            # Confidence guardrails: when top class confidence ≥ 0.65, don't downgrade to unknown
            if best_score >= 0.25:  # Lowered from 0.3
                if self.debug:
                    print(f"    → {best_color} (score: {best_score:.3f})")
                return best_color, best_score
            else:
                if self.debug:
                    print(f"    → unknown (best score {best_score:.3f} < 0.25)")
                return "unknown", best_score
        else:
            if self.debug:
                print(f"    → unknown (no color matches)")
            return "unknown", 0.0
    
    def _estimate_border_luminance(self, l_ring: float, delta_l: float) -> float:
        """Estimate border luminance for whiteness override (simplified)"""
        # Simplified border estimation: use the ring L* plus a typical border offset
        # In practice, this would sample the actual card border
        typical_border_offset = 15  # Typical difference between border and ring
        l_border = l_ring + typical_border_offset
        
        # Clamp to reasonable range
        l_border = max(70, min(95, l_border))
        
        return l_border
    
    def _classify_hsv_color_improved(self, h: float, s: float, v: float) -> str:
        """Improved HSV classification with black veto safety net"""
        
        # HSV hue safety-net: if S > 0.18 and H ∈ [70°, 160°], disallow black
        if s > 0.18 and 70 <= h <= 160:
            # This is likely green, not black
            return "G"
        
        # Red: hue near 0 or 180
        if ((0 <= h <= 10) or (170 <= h <= 180)) and s >= 40 and v >= 50:
            return "R"
        
        # Blue: hue around 120
        if 100 <= h <= 140 and s >= 40 and v >= 50:
            return "U"
        
        # Green: hue around 60
        if 50 <= h <= 90 and s >= 40 and v >= 50:
            return "G"
        
        # White: high value, low saturation
        if v >= 200 and s <= 30:
            return "W"
        
        # Black: low value (but check HSV safety net first)
        if v <= 50 and s <= 50:
            return "B"
        
        # Colorless: low saturation
        if s <= 20 and 50 < v < 200:
            return "C"
        
        return "unknown"
    
    def _log_lab_diagnostics(self, l: float, a: float, b: float, chroma: float, hue_angle: float, delta_l: float, delta_c: float):
        """Log LAB values for diagnostic analysis"""
        if not hasattr(self, '_lab_log'):
            self._lab_log = []
        
        # Store LAB diagnostic data
        self._lab_log.append({
            'L': l,
            'a': a,
            'b': b,
            'chroma': chroma,
            'hue_angle': hue_angle,
            'delta_l': delta_l,
            'delta_c': delta_c,
            'timestamp': time.time()
        })
        
        # Keep only last 1000 entries to prevent memory issues
        if len(self._lab_log) > 1000:
            self._lab_log = self._lab_log[-1000:]
    
    def get_lab_diagnostics(self) -> List[Dict]:
        """Get logged LAB diagnostic data"""
        return getattr(self, '_lab_log', [])
    
    def clear_lab_diagnostics(self):
        """Clear logged LAB diagnostic data"""
        if hasattr(self, '_lab_log'):
            self._lab_log.clear()
    
    def _classify_lab_improved_v2(self, l: float, a: float, b: float, chroma: float, hue_angle: float, delta_l: float, delta_c: float) -> Tuple[str, float]:
        """Improved LAB classification with HARD OVERRIDE rule for green vs black"""
        
        # Debug output
        if self.debug:
            print(f"  LAB Debug: L={l:.1f}, a={a:.1f}, b={b:.1f}, C={chroma:.1f}, θ={hue_angle:.1f}, ΔL={delta_l:.1f}")
        
        # HARD OVERRIDE: If ΔL ≥ 0 and in green hue range, NEVER classify as black
        if delta_l >= 0 and 70 <= hue_angle <= 150:
            # Force green classification
            margin = min(chroma - 10, abs(hue_angle - 70), abs(hue_angle - 150))
            green_score = chroma/40 + margin/50 + 1.0  # Maximum override bonus
            if self.debug:
                print(f"    → G (HARD OVERRIDE: ΔL={delta_l:.1f} ≥ 0)")
            return "G", green_score
        
        # FINAL surgical color classification with proper gate order and ΔL requirements
        color_scores = {}
        
        # White (W) – relax slightly from stratified test results - FINAL FIX
        if l >= 82 and chroma <= 10 and abs(a) <= 8 and abs(b) <= 8 and delta_l >= 10:
            score_white = (l/100) + (1 - chroma/10) + min(delta_l/20, 1.0)
            color_scores['W'] = score_white
        
        # Black (B) – MUCH stricter to stop G→B - FINAL FIX
        # Disqualify B if ΔL > -6 (harder rule)
        if l <= 28 and delta_l <= -12 and chroma <= 10 and delta_l <= -6:
            margin = (28 - l) + abs(delta_l + 12) + (10 - chroma)
            color_scores['B'] = 0.8 + margin/50
        
        # Green (G) – ULTRA STRONG override when not darker than outside - FINAL FIX
        if 70 <= hue_angle <= 150 and (chroma >= 10 or delta_l >= 0):
            margin = min(chroma - 10, abs(hue_angle - 70), abs(hue_angle - 150))
            # Override B when ΔL ≥ 0 even if L* is modest
            if delta_l >= 0:
                color_scores['G'] = chroma/40 + margin/50 + 0.9  # Ultra strong override bonus
            else:
                color_scores['G'] = chroma/40 + margin/50 + 0.3  # Still strong bonus
        
        # Blue (U)
        if (-150 <= hue_angle <= -70 or hue_angle >= 200) and chroma >= 14:
            margin = min(chroma - 14, abs(hue_angle + 150), abs(hue_angle + 70))
            color_scores['U'] = chroma/40 + margin/50
        
        # Red (R) - FINAL FIX for R vs W
        if -15 <= hue_angle <= 65 and chroma >= 14 and 35 <= l <= 85:
            margin = min(chroma - 14, abs(hue_angle + 15), abs(hue_angle - 65), l - 35, 85 - l)
            color_scores['R'] = chroma/40 + margin/50
        
        # Digit candidate: if (C* ≤ 10 and 45 ≤ L* ≤ 80) or ΔL < 6, treat as digit-candidate
        if (chroma <= 10 and 45 <= l <= 80) or delta_l < 6:
            color_scores['digit_candidate'] = 0.5
        
        # Colorless (C) - IMPROVED restrictive gate - SURGICAL FIX
        # Only allow {C} when digit path fails and L* ≤ 40 and C* ≤ 8 and ΔL ≤ 0
        if l <= 40 and chroma <= 8 and delta_l <= 0:
            margin = (40 - l) + (8 - chroma) + abs(delta_l)
            color_scores['C'] = 0.6 + margin/50
        
        # Find best color
        if color_scores:
            best_color = max(color_scores, key=color_scores.get)
            best_score = color_scores[best_color]
            
            # Confidence guardrails: when top class confidence ≥ 0.65, don't downgrade to unknown
            if best_score >= 0.25:  # Lowered from 0.3
                if self.debug:
                    print(f"    → {best_color} (score: {best_score:.3f})")
                return best_color, best_score
            else:
                if self.debug:
                    print(f"    → unknown (best score {best_score:.3f} < 0.25)")
                return "unknown", best_score
        else:
            if self.debug:
                print(f"    → unknown (no color matches)")
            return "unknown", 0.0
    
    def _classify_hsv_direct(self, region: np.ndarray, radius: int) -> str:
        """Direct HSV classification for comparison testing"""
        h, w = region.shape[:2]
        center_x, center_y = w // 2, h // 2
        
        # Sample ring annulus only (0.65-0.85R)
        y_coords, x_coords = np.ogrid[:h, :w]
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        inner_r = int(radius * 0.65)
        outer_r = int(radius * 0.85)
        ring_mask = (distances >= inner_r) & (distances <= outer_r)
        
        if not np.any(ring_mask):
            return "unknown"
        
        # Extract ring pixels
        ring_pixels = region[ring_mask]
        
        if len(ring_pixels) == 0:
            return "unknown"
        
        # Convert to HSV
        hsv_pixels = cv2.cvtColor(ring_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
        hsv_pixels = hsv_pixels.reshape(-1, 3)
        
        # Get median HSV values
        h_median = np.median(hsv_pixels[:, 0])
        s_median = np.median(hsv_pixels[:, 1])
        v_median = np.median(hsv_pixels[:, 2])
        
        return self._classify_hsv_color(h_median, s_median, v_median)
    
    def _classify_lab_color_improved(self, l: float, a: float, b: float, chroma: float, delta_l: float) -> str:
        """Improved LAB classification with proper routing to digit recognition and reason codes"""
        
        # Debug output for first few symbols
        if self.debug:
            print(f"  LAB Debug: L={l:.1f}, a={a:.1f}, b={b:.1f}, C={chroma:.1f}, ΔL={delta_l:.1f}")
        
        # Color gate order: R/U/G/B → White vs Digit → else {C}
        
        # Red: high a* (red component) - more lenient
        if a >= 15 and abs(b) <= 35:  # Lowered threshold
            if self.debug:
                print(f"    → R (a={a:.1f}, |b|={abs(b):.1f})")
            return "R"
        
        # Blue: negative b* (blue component) - more lenient
        if b <= -10 and -10 <= a <= 20:  # Lowered threshold
            if self.debug:
                print(f"    → U (b={b:.1f}, a={a:.1f})")
            return "U"
        
        # Green: negative a* and positive b* - more lenient
        if a <= -5 and b >= 5:  # Lowered threshold
            if self.debug:
                print(f"    → G (a={a:.1f}, b={b:.1f})")
            return "G"
        
        # Black: low lightness
        if l <= 35 and chroma <= 20:
            if self.debug:
                print(f"    → B (L={l:.1f}, C={chroma:.1f})")
            return "B"
        
        # White vs Digit gate (new improved rule)
        # White: high lightness, low chroma, and stands out from background
        if l >= 75 and chroma <= 10 and delta_l >= 8:  # Slightly more lenient
            if self.debug:
                print(f"    → W (L={l:.1f}, C={chroma:.1f}, ΔL={delta_l:.1f})")
            return "W"
        
        # Digit-candidate: low chroma, moderate lightness, or doesn't stand out
        if (chroma <= 12 and 40 <= l <= 80) or (delta_l < 12):  # More lenient
            if self.debug:
                print(f"    → digit_candidate (C={chroma:.1f}, L={l:.1f}, ΔL={delta_l:.1f})")
            return "digit_candidate"  # Route to digit recognition
        
        # Only classify as colorless if very dark and low chroma
        if l <= 35 and chroma <= 8:  # More lenient
            if self.debug:
                print(f"    → C (L={l:.1f}, C={chroma:.1f})")
            return "C"
        
        # Otherwise keep as unknown (let grammar/prior help)
        if self.debug:
            print(f"    → unknown (no conditions met)")
            # Print reason codes for unknown
            failed_tests = []
            if not (a >= 15 and abs(b) <= 35):
                failed_tests.append("red_gate")
            if not (b <= -10 and -10 <= a <= 20):
                failed_tests.append("blue_gate")
            if not (a <= -5 and b >= 5):
                failed_tests.append("green_gate")
            if not (l <= 35 and chroma <= 20):
                failed_tests.append("black_gate")
            if not (l >= 75 and chroma <= 10 and delta_l >= 8):
                failed_tests.append("white_gate")
            if not ((chroma <= 12 and 40 <= l <= 80) or (delta_l < 12)):
                failed_tests.append("digit_gate")
            if not (l <= 35 and chroma <= 8):
                failed_tests.append("colorless_gate")
            
            print(f"    UNK: C_in={chroma:.1f}, L_in={l:.1f}, ΔL={delta_l:.1f} | failed: {failed_tests}")
        
        return "unknown"
    
    def _resolve_white_vs_generic(self, region: np.ndarray, radius: int, current_class: str) -> str:
        """Resolve white vs generic using digit presence check"""
        
        # If it's already a digit candidate, route to digit recognition
        if current_class == "digit_candidate":
            return self._check_digit_presence(region, radius)
        
        # If it's white, check if it might actually be a digit
        if current_class == "W":
            digit_result = self._check_digit_presence(region, radius)
            if digit_result != "W":
                return digit_result
        
        return current_class
    
    def _check_digit_presence(self, region: np.ndarray, radius: int) -> str:
        """Check for digit presence and route to digit recognition if detected"""
        
        # Crop center region (≤0.55R)
        h, w = region.shape[:2]
        center_x, center_y = w // 2, h // 2
        crop_radius = int(radius * 0.55)
        
        x1 = max(0, center_x - crop_radius)
        y1 = max(0, center_y - crop_radius)
        x2 = min(w, center_x + crop_radius)
        y2 = min(h, center_y + crop_radius)
        
        center_crop = region[y1:y2, x1:x2]
        
        if center_crop.size == 0:
            return "unknown"
        
        # Convert to grayscale
        if len(center_crop.shape) == 3:
            gray = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = center_crop
        
        # Upscale to fixed height (28px)
        target_height = 28
        scale = target_height / gray.shape[0]
        target_width = int(gray.shape[1] * scale)
        resized = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply unsharp mask
        blurred = cv2.GaussianBlur(resized, (0, 0), 1.0)
        unsharp = cv2.addWeighted(resized, 1.5, blurred, -0.5, 0)
        
        # Sauvola thresholding
        binary = self._sauvola_threshold(unsharp)
        
        # Check if we need to invert
        if self._should_invert(binary):
            binary = cv2.bitwise_not(binary)
        
        # 3x3 median filter
        denoised = cv2.medianBlur(binary, 3)
        
        # Compute digit presence features
        entropy = self._compute_entropy(denoised)
        projection_peaks = self._count_projection_peaks(denoised)
        
        # Digit trigger: entropy ≥ 2.0 or peaks ≥ 3 (improved thresholds)
        if entropy >= 2.0 or projection_peaks >= 3:
            # Route to digit recognition
            digit_result = self._recognize_digit_improved(denoised)
            if digit_result:
                return digit_result
        
        # If no digit detected, check if it should be colorless
        # Only classify as {C} if very dark: L_in ≤ 40 and C_in ≤ 6
        # For now, return unknown and let the caller handle it
        return "unknown"
    
    def _recognize_digit_improved(self, binary_image: np.ndarray) -> Optional[str]:
        """Improved two-pass digit recognition: OCR → template matching"""
        
        # First pass: OCR with allowlist and higher confidence
        try:
            results = self.ocr_reader.readtext(
                binary_image, 
                allowlist='0123456789X',
                paragraph=False
            )
            
            if results:
                # Return highest confidence result
                best_result = max(results, key=lambda x: x[2])
                if best_result[2] >= 0.55:  # Min confidence threshold
                    digit = best_result[1]
                    
                    # Handle multi-digit (e.g., "10", "12")
                    if len(digit) > 1 and digit.isdigit():
                        return digit  # Return as-is for multi-digit
                    
                    # Apply grammar corrections
                    digit = self._apply_digit_grammar(digit, best_result[2])
                    return digit
        except Exception:
            pass
        
        # Second pass: template matching with improved templates
        return self._template_match_digit_improved(binary_image)
    
    def _sauvola_threshold(self, image: np.ndarray, window_size: int = 15, k: float = 0.2) -> np.ndarray:
        """Sauvola adaptive thresholding for better digit recognition"""
        # Convert to float
        img_float = image.astype(np.float32)
        
        # Compute local mean and standard deviation
        mean = cv2.blur(img_float, (window_size, window_size))
        mean_sq = cv2.blur(img_float * img_float, (window_size, window_size))
        variance = mean_sq - mean * mean
        std_dev = np.sqrt(np.maximum(variance, 0))
        
        # Sauvola threshold
        threshold = mean * (1 + k * (std_dev / 128 - 1))
        
        # Apply threshold
        binary = (img_float > threshold).astype(np.uint8) * 255
        
        return binary
    
    def _compute_entropy(self, binary_image: np.ndarray) -> float:
        """Compute entropy of binary image (measure of complexity)"""
        # Count black and white pixels
        black_pixels = np.sum(binary_image == 0)
        white_pixels = np.sum(binary_image == 255)
        total_pixels = black_pixels + white_pixels
        
        if total_pixels == 0:
            return 0.0
        
        # Compute probabilities
        p_black = black_pixels / total_pixels
        p_white = white_pixels / total_pixels
        
        # Compute entropy
        entropy = 0.0
        if p_black > 0:
            entropy -= p_black * np.log2(p_black)
        if p_white > 0:
            entropy -= p_white * np.log2(p_white)
        
        return entropy
    
    def _count_projection_peaks(self, binary_image: np.ndarray) -> int:
        """Count peaks in horizontal and vertical projections"""
        # Horizontal projection
        h_proj = np.sum(binary_image == 255, axis=1)
        h_peaks = self._count_peaks(h_proj)
        
        # Vertical projection
        v_proj = np.sum(binary_image == 255, axis=0)
        v_peaks = self._count_peaks(v_proj)
        
        return h_peaks + v_peaks
    
    def _count_peaks(self, projection: np.ndarray) -> int:
        """Count peaks in a 1D array"""
        peaks = 0
        for i in range(1, len(projection) - 1):
            if projection[i] > projection[i-1] and projection[i] > projection[i+1]:
                peaks += 1
        return peaks
    
    def _classify_hsv_color(self, h: float, s: float, v: float) -> str:
        """Fallback HSV classification with improved thresholds"""
        
        # Red: hue near 0 or 180
        if ((0 <= h <= 10) or (170 <= h <= 180)) and s >= 40 and v >= 50:
            return "R"
        
        # Blue: hue around 120
        if 100 <= h <= 140 and s >= 40 and v >= 50:
            return "U"
        
        # Green: hue around 60
        if 50 <= h <= 90 and s >= 40 and v >= 50:
            return "G"
        
        # White: high value, low saturation
        if v >= 200 and s <= 30:
            return "W"
        
        # Black: low value
        if v <= 50 and s <= 50:
            return "B"
        
        # Colorless: low saturation
        if s <= 20 and 50 < v < 200:
            return "C"
        
        return "unknown"
    
    def _ocr_digit(self, region: np.ndarray) -> Optional[str]:
        """Improved OCR for digits with preprocessing and template fallback"""
        try:
            # Resize for better OCR
            h, w = region.shape[:2]
            if h < 20 or w < 20:
                region = cv2.resize(region, (max(20, w*2), max(20, h*2)), interpolation=cv2.INTER_CUBIC)
            
            # Digit-specific preprocessing
            processed_region = self._preprocess_for_digits(region)
            
            # OCR with digit allowlist
            results = self.ocr_reader.readtext(
                processed_region, 
                allowlist='0123456789X',
                paragraph=False
            )
            
            if results:
                # Return highest confidence result
                best_result = max(results, key=lambda x: x[2])
                if best_result[2] > 0.5:  # Confidence threshold
                    digit = best_result[1]
                    
                    # Apply grammar sanity checks
                    digit = self._apply_digit_grammar(digit, best_result[2])
                    
                    return digit
            
            # Fallback to template matching if OCR fails
            return self._template_match_digit(processed_region)
            
        except Exception:
            return None
    
    def _preprocess_for_digits(self, region: np.ndarray) -> np.ndarray:
        """Improved preprocessing specifically for digit recognition"""
        h, w = region.shape[:2]
        
        # Convert to grayscale
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region
        
        # Tight center crop (55% of radius)
        center_x, center_y = w // 2, h // 2
        crop_radius = int(min(w, h) * 0.55)
        
        x1 = max(0, center_x - crop_radius)
        y1 = max(0, center_y - crop_radius)
        x2 = min(w, center_x + crop_radius)
        y2 = min(h, center_y + crop_radius)
        
        cropped = gray[y1:y2, x1:x2]
        
        # Resize to fixed height (32px)
        target_height = 32
        scale = target_height / cropped.shape[0]
        target_width = int(cropped.shape[1] * scale)
        resized = cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply unsharp mask to enhance edges
        blurred = cv2.GaussianBlur(resized, (0, 0), 1.0)
        unsharp = cv2.addWeighted(resized, 1.5, blurred, -0.5, 0)
        
        # Sauvola thresholding (better than Otsu for small, low-contrast glyphs)
        binary = self._sauvola_threshold(unsharp)
        
        # Check if we need to invert (background lighter than strokes)
        if self._should_invert(binary):
            binary = cv2.bitwise_not(binary)
        
        # 3x3 median filter to remove noise
        denoised = cv2.medianBlur(binary, 3)
        
        return denoised
    
    def _should_invert(self, binary_image: np.ndarray) -> bool:
        """Check if binary image should be inverted"""
        # Count black and white pixels
        black_pixels = np.sum(binary_image == 0)
        white_pixels = np.sum(binary_image == 255)
        
        # If more white pixels, likely need to invert
        return white_pixels > black_pixels
    
    def _template_match_digit(self, region: np.ndarray) -> Optional[str]:
        """Improved template matching fallback for digit recognition"""
        # Enhanced templates for common digits
        templates = {
            '0': self._create_digit_template('0'),
            '1': self._create_digit_template('1'),
            '2': self._create_digit_template('2'),
            '3': self._create_digit_template('3'),
            '4': self._create_digit_template('4'),
            '5': self._create_digit_template('5'),
            'X': self._create_digit_template('X')
        }
        
        best_match = None
        best_score = 0.0
        scores = {}
        
        # Get input image dimensions
        h, w = region.shape[:2]
        
        for digit, template in templates.items():
            # Check if template is smaller than input image
            t_h, t_w = template.shape[:2]
            if t_h > h or t_w > w:
                continue  # Skip this template if it's too large
            
            # Normalized cross-correlation
            try:
                result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
                score = np.max(result)
                scores[digit] = score
                
                if score > best_score and score > 0.6:  # Threshold
                    best_score = score
                    best_match = digit
            except Exception:
                continue  # Skip this template if matching fails
        
        # If we have a tie between 0 and 2/5, use feature-based tie-breaking
        if best_match in ['0', '2', '5'] and best_score > 0.7:
            # Check for holes (0 has a hole, 2/5 don't)
            has_hole = self._check_for_hole(region)
            if has_hole and best_match in ['2', '5']:
                best_match = '0'
            elif not has_hole and best_match == '0':
                # Choose between 2 and 5 based on shape
                if scores.get('2', 0) > scores.get('5', 0):
                    best_match = '2'
                else:
                    best_match = '5'
        
        return best_match
    
    def _template_match_digit_improved(self, region: np.ndarray) -> Optional[str]:
        """Improved template matching fallback for digit recognition"""
        # Enhanced templates for common digits
        templates = {
            '0': self._create_digit_template('0'),
            '1': self._create_digit_template('1'),
            '2': self._create_digit_template('2'),
            '3': self._create_digit_template('3'),
            '4': self._create_digit_template('4'),
            '5': self._create_digit_template('5'),
            'X': self._create_digit_template('X')
        }
        
        best_match = None
        best_score = 0.0
        scores = {}
        
        # Get input image dimensions
        h, w = region.shape[:2]
        
        for digit, template in templates.items():
            # Check if template is smaller than input image
            t_h, t_w = template.shape[:2]
            if t_h > h or t_w > w:
                continue  # Skip this template if it's too large
            
            # Normalized cross-correlation
            try:
                result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
                score = np.max(result)
                scores[digit] = score
                
                if score > best_score and score > 0.6:  # Threshold
                    best_score = score
                    best_match = digit
            except Exception:
                continue  # Skip this template if matching fails
        
        # If we have a tie between 0 and 2/5, use feature-based tie-breaking
        if best_match in ['0', '2', '5'] and best_score > 0.7:
            # Check for holes (0 has a hole, 2/5 don't)
            has_hole = self._check_for_hole(region)
            if has_hole and best_match in ['2', '5']:
                best_match = '0'
            elif not has_hole and best_match == '0':
                # Choose between 2 and 5 based on shape
                if scores.get('2', 0) > scores.get('5', 0):
                    best_match = '2'
                else:
                    best_match = '5'
        
        return best_match
    
    def _check_for_hole(self, binary_image: np.ndarray) -> bool:
        """Check if binary image has a hole (for 0 vs 2/5 distinction)"""
        # Find contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) < 2:
            return False
        
        # Sort by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Check if second largest contour is inside the largest
        if len(contours) >= 2:
            largest = contours[0]
            second = contours[1]
            
            # Check if second contour center is inside largest contour
            largest_center = np.mean(largest, axis=0)[0]
            second_center = np.mean(second, axis=0)[0]
            
            # Simple check: if second contour center is inside largest contour
            if cv2.pointPolygonTest(largest, tuple(second_center), False) > 0:
                return True
        
        return False
    
    def _create_digit_template(self, digit: str) -> np.ndarray:
        """Create enhanced template for digit matching with smaller size"""
        # Create 12x20 template (smaller than before to avoid size issues)
        template = np.zeros((20, 12), dtype=np.uint8)
        
        if digit == '0':
            # Oval shape with hole
            cv2.ellipse(template, (6, 10), (4, 8), 0, 0, 360, 255, 2)
            cv2.ellipse(template, (6, 10), (2, 5), 0, 0, 360, 0, -1)
        elif digit == '1':
            # Vertical line with slight angle
            cv2.line(template, (6, 2), (6, 18), 255, 2)
            cv2.line(template, (4, 4), (6, 2), 255, 2)
        elif digit == '2':
            # Curved shape
            cv2.ellipse(template, (6, 6), (4, 3), 0, 0, 180, 255, 2)
            cv2.line(template, (2, 9), (10, 9), 255, 2)
            cv2.line(template, (10, 9), (10, 15), 255, 2)
            cv2.line(template, (2, 15), (10, 15), 255, 2)
        elif digit == '3':
            # S-like shape with curves
            cv2.ellipse(template, (6, 6), (4, 3), 0, 0, 180, 255, 2)
            cv2.ellipse(template, (6, 14), (4, 3), 0, 180, 360, 255, 2)
            cv2.line(template, (10, 6), (10, 14), 255, 2)
        elif digit == '4':
            # Cross shape
            cv2.line(template, (3, 2), (3, 18), 255, 2)
            cv2.line(template, (3, 10), (9, 10), 255, 2)
            cv2.line(template, (6, 10), (6, 18), 255, 2)
        elif digit == '5':
            # S-like shape
            cv2.line(template, (9, 2), (3, 2), 255, 2)
            cv2.line(template, (3, 2), (3, 8), 255, 2)
            cv2.line(template, (3, 8), (9, 8), 255, 2)
            cv2.line(template, (9, 8), (9, 14), 255, 2)
            cv2.line(template, (3, 14), (9, 14), 255, 2)
        elif digit == 'X':
            # X shape
            cv2.line(template, (3, 2), (9, 18), 255, 2)
            cv2.line(template, (9, 2), (3, 18), 255, 2)
        
        return template
    
    def _apply_digit_grammar(self, digit: str, confidence: float) -> str:
        """Apply improved grammar rules to correct common digit misclassifications"""
        
        # Prefer common digits in mana costs
        if digit == '0' and confidence < 0.8:
            # Low confidence '0' is often '1' or '2'
            return '1'  # More common in mana costs
        
        # Prefer '2' over '5' for low confidence cases
        if digit == '5' and confidence < 0.7:
            return '2'  # More common in mana costs
        
        # Prefer '4' over '1' for certain patterns
        if digit == '1' and confidence < 0.6:
            return '4'  # Could be misclassified
        
        return digit
    
    def _feature_check_digit(self, region: np.ndarray, digit: str) -> str:
        """Feature-based checks to break digit ties"""
        # Convert to binary if needed
        if len(region.shape) == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region
        
        # Find contours
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return digit
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate features
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Solidity (area / convex hull area)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Hole ratio (Euler number)
        # '0' has a hole, '1' doesn't
        if digit == '0' and solidity > 0.8:  # No hole detected
            return '1'
        elif digit == '1' and solidity < 0.6:  # Hole detected
            return '0'
        
        return digit
    
    def _remove_duplicates_improved(self, symbols: List[ManaSymbol]) -> List[ManaSymbol]:
        """Remove duplicate symbols using improved NMS with adaptive window"""
        if not symbols:
            return symbols
        
        # Sort by confidence
        symbols.sort(key=lambda s: s.confidence, reverse=True)
        
        # 1D NMS: Project centers onto RANSAC line and merge nearby centers
        if len(symbols) > 1:
            # Fit line to centers
            centers = np.array([symbol.center for symbol in symbols])
            line_vector = self._fit_line_to_centers(centers)
            
            # Project centers onto line
            projections = []
            for i, symbol in enumerate(symbols):
                center = np.array(symbol.center)
                proj = np.dot(center, line_vector)
                projections.append((proj, i))
            
            # Sort by projection
            projections.sort()
            
            # Adaptive NMS window: use 0.6·min(r_i, r_j) when both peaks are strong (top 30% ring scores)
            filtered_indices = []
            confidences = [s.confidence for s in symbols]
            top_30_threshold = np.percentile(confidences, 70)  # Top 30%
            
            for proj, idx in projections:
                # Check if too close to existing symbols
                too_close = False
                for existing_proj, existing_idx in filtered_indices:
                    distance = abs(proj - existing_proj)
                    min_radius = min(symbols[idx].radius, symbols[existing_idx].radius)
                    
                    # Adaptive window: tighter for strong peaks
                    current_conf = symbols[idx].confidence
                    existing_conf = symbols[existing_idx].confidence
                    
                    if current_conf >= top_30_threshold and existing_conf >= top_30_threshold:
                        window = 0.6 * min_radius  # Tighter for strong peaks
                    else:
                        window = 0.75 * min_radius  # Standard window
                    
                    if distance < window:
                        too_close = True
                        break
                
                if not too_close:
                    filtered_indices.append((proj, idx))
            
            # Get filtered symbols
            symbols = [symbols[idx] for _, idx in filtered_indices]
        
        # Merge radii if |r_i − r_j|/r̄ < 0.25 (double detections from inner/outer ring)
        if len(symbols) > 1:
            radii = [s.radius for s in symbols]
            mean_radius = np.mean(radii)
            
            merged_symbols = []
            used_indices = set()
            
            for i, symbol in enumerate(symbols):
                if i in used_indices:
                    continue
                
                # Find nearby symbols with similar radii
                similar_symbols = [symbol]
                used_indices.add(i)
                
                for j, other_symbol in enumerate(symbols[i+1:], i+1):
                    if j in used_indices:
                        continue
                    
                    radius_diff = abs(symbol.radius - other_symbol.radius) / mean_radius
                    if radius_diff < 0.25:
                        similar_symbols.append(other_symbol)
                        used_indices.add(j)
                
                # Keep the highest confidence symbol from similar ones
                best_symbol = max(similar_symbols, key=lambda s: s.confidence)
                merged_symbols.append(best_symbol)
            
            symbols = merged_symbols
        
        # Remove radius outliers (Z-score > 1.8)
        if len(symbols) > 2:
            radii = [s.radius for s in symbols]
            mean_radius = np.mean(radii)
            std_radius = np.std(radii)
            
            if std_radius > 0:
                z_scores = [(abs(r - mean_radius) / std_radius) for r in radii]
                symbols = [s for s, z in zip(symbols, z_scores) if z <= 1.8]
        
        # Gap sanity: enforce reasonable gaps between symbols
        if len(symbols) > 1:
            # Sort by x-coordinate
            symbols.sort(key=lambda s: s.center[0])
            
            filtered = [symbols[0]]
            for i in range(1, len(symbols)):
                prev_symbol = filtered[-1]
                curr_symbol = symbols[i]
                
                # Calculate distance between centers
                distance = np.sqrt((curr_symbol.center[0] - prev_symbol.center[0])**2 + 
                                 (curr_symbol.center[1] - prev_symbol.center[1])**2)
                
                # Expected gap: 0.35-2.5 times radius
                min_gap = 0.35 * (prev_symbol.radius + curr_symbol.radius)
                max_gap = 2.5 * (prev_symbol.radius + curr_symbol.radius)
                
                if min_gap <= distance <= max_gap:
                    filtered.append(curr_symbol)
                else:
                    # Gap is unreasonable, keep the one with higher confidence
                    if curr_symbol.confidence > prev_symbol.confidence:
                        filtered[-1] = curr_symbol
            
            symbols = filtered
        
        # Score floor tweak: raise the "keep" floor slightly from median - 1.0·IQR to median - 1.2·IQR
        if len(symbols) > 1:
            confidences = [s.confidence for s in symbols]
            median_conf = np.median(confidences)
            q75 = np.percentile(confidences, 75)
            q25 = np.percentile(confidences, 25)
            iqr = q75 - q25
            score_floor = median_conf - 1.2 * iqr  # Raised from 1.0 to 1.2
            
            symbols = [s for s in symbols if s.confidence >= score_floor]
        
        # Clamp to max 6 pips by dropping the lowest scores
        if len(symbols) > 6:
            symbols.sort(key=lambda s: s.confidence, reverse=True)
            symbols = symbols[:6]
        
        return symbols
    
    def _fit_line_to_centers(self, centers: np.ndarray) -> np.ndarray:
        """Fit a line to symbol centers using RANSAC"""
        if len(centers) < 2:
            return np.array([1, 0])  # Default horizontal line
        
        # Simple line fitting: use the direction of the first two points
        if len(centers) >= 2:
            direction = centers[1] - centers[0]
            direction = direction / np.linalg.norm(direction)
            return direction
        
        # Fallback: horizontal line
        return np.array([1, 0])
    
    def _build_mana_string(self, symbols: List[ManaSymbol]) -> str:
        """Build canonical mana cost string with improved ordering and grammar validation"""
        if not symbols:
            return ""
        
        # Robust ordering using RANSAC line fitting
        sorted_symbols = self._order_symbols_robust(symbols)
        
        # Apply grammar validation
        validated_symbols = self._validate_mana_grammar(sorted_symbols)
        
        # Build string
        mana_parts = []
        for symbol in validated_symbols:
            if symbol.symbol:
                mana_parts.append(f"{{{symbol.symbol}}}")
        
        return ''.join(mana_parts)
    
    def _order_symbols_robust(self, symbols: List[ManaSymbol]) -> List[ManaSymbol]:
        """Order symbols using RANSAC line fitting for robust ordering"""
        if len(symbols) <= 1:
            return symbols
        
        # Extract centers
        centers = np.array([symbol.center for symbol in symbols])
        
        # Fit line using RANSAC
        best_line = None
        best_inliers = 0
        
        for _ in range(10):  # RANSAC iterations
            # Randomly sample 2 points
            idx = np.random.choice(len(centers), 2, replace=False)
            p1, p2 = centers[idx[0]], centers[idx[1]]
            
            # Line equation: ax + by + c = 0
            a = p2[1] - p1[1]  # dy
            b = p1[0] - p2[0]  # -dx
            c = p2[0] * p1[1] - p1[0] * p2[1]  # x2*y1 - x1*y2
            
            # Count inliers (points close to line)
            distances = np.abs(a * centers[:, 0] + b * centers[:, 1] + c) / np.sqrt(a*a + b*b)
            inliers = np.sum(distances < 5)  # 5 pixel threshold
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_line = (a, b, c)
        
        if best_line is None:
            # Fallback to x-sort
            return sorted(symbols, key=lambda s: s.center[0])
        
        # Project centers onto line and sort by projection
        a, b, c = best_line
        line_vector = np.array([-b, a])  # Perpendicular to line
        line_vector = line_vector / np.linalg.norm(line_vector)
        
        # Project each center onto the line
        projections = []
        for i, symbol in enumerate(symbols):
            center = np.array(symbol.center)
            # Project onto line
            proj = np.dot(center, line_vector)
            projections.append((proj, i))
        
        # Sort by projection
        projections.sort()
        ordered_symbols = [symbols[idx] for _, idx in projections]
        
        return ordered_symbols
    
    def _validate_mana_grammar(self, symbols: List[ManaSymbol]) -> List[ManaSymbol]:
        """Validate and correct mana cost grammar with confidence-aware filtering"""
        if not symbols:
            return symbols
        
        # Merge consecutive digits into {N}
        merged_symbols = self._merge_digits(symbols)
        
        # Sort colors in WUBRG order
        sorted_symbols = self._sort_colors_wubrg(merged_symbols)
        
        # Soft grammar validation (drop at most 1 lowest-confidence pip)
        validated_symbols = self._soft_grammar_validation(sorted_symbols)
        
        return validated_symbols
    
    def _soft_grammar_validation(self, symbols: List[ManaSymbol]) -> List[ManaSymbol]:
        """Soft grammar validation that drops at most 1 lowest-confidence pip"""
        if len(symbols) <= 1:
            return symbols
        
        # First pass: validate gaps without dropping
        validated = self._validate_symbol_gaps_soft(symbols)
        
        # If validation failed, try dropping the lowest confidence pip
        if len(validated) < len(symbols):
            # Find lowest confidence symbol
            lowest_conf_idx = min(range(len(symbols)), key=lambda i: symbols[i].confidence)
            
            # Create new list without the lowest confidence symbol
            filtered_symbols = [s for i, s in enumerate(symbols) if i != lowest_conf_idx]
            
            # Re-validate
            re_validated = self._validate_symbol_gaps_soft(filtered_symbols)
            
            # Return the better result (more symbols is better)
            if len(re_validated) >= len(validated):
                return re_validated
            else:
                return validated
        
        return validated
    
    def _validate_symbol_gaps_soft(self, symbols: List[ManaSymbol]) -> List[ManaSymbol]:
        """Soft gap validation with confidence-aware filtering"""
        if len(symbols) <= 1:
            return symbols
        
        validated = [symbols[0]]
        
        for i in range(1, len(symbols)):
            prev_symbol = validated[-1]
            curr_symbol = symbols[i]
            
            # Calculate distance between centers
            distance = np.sqrt((curr_symbol.center[0] - prev_symbol.center[0])**2 + 
                             (curr_symbol.center[1] - prev_symbol.center[1])**2)
            
            # More lenient gap thresholds: 0.35-2.5 times radius
            min_gap = 0.35 * (prev_symbol.radius + curr_symbol.radius)
            max_gap = 2.5 * (prev_symbol.radius + curr_symbol.radius)
            
            if min_gap <= distance <= max_gap:
                validated.append(curr_symbol)
            else:
                # Gap is unreasonable, but be more lenient
                # Only drop if confidence difference is significant
                conf_diff = abs(curr_symbol.confidence - prev_symbol.confidence)
                if conf_diff > 0.3:  # Significant confidence difference
                    if curr_symbol.confidence > prev_symbol.confidence:
                        validated[-1] = curr_symbol
                    # Otherwise keep the previous symbol
        
        return validated
    
    def _merge_digits(self, symbols: List[ManaSymbol]) -> List[ManaSymbol]:
        """Merge consecutive digits into {N} format with confidence-aware handling"""
        if not symbols:
            return symbols
        
        merged = []
        current_digits = []
        digit_positions = []
        
        for i, symbol in enumerate(symbols):
            # Check if symbol is a digit (including multi-digit)
            if symbol.symbol.isdigit() or (len(symbol.symbol) > 1 and symbol.symbol.isdigit()):
                current_digits.append(int(symbol.symbol))
                digit_positions.append(i)
            else:
                # Merge accumulated digits
                if current_digits:
                    total = sum(current_digits)
                    # Use position of first digit for center
                    first_pos = digit_positions[0]
                    merged.append(ManaSymbol(
                        symbol=str(total),
                        confidence=np.mean([symbols[pos].confidence for pos in digit_positions]),
                        center=symbols[first_pos].center,
                        radius=symbols[first_pos].radius,
                        color="generic"
                    ))
                    current_digits = []
                    digit_positions = []
                
                merged.append(symbol)
        
        # Handle trailing digits
        if current_digits:
            total = sum(current_digits)
            first_pos = digit_positions[0]
            merged.append(ManaSymbol(
                symbol=str(total),
                confidence=np.mean([symbols[pos].confidence for pos in digit_positions]),
                center=symbols[first_pos].center,
                radius=symbols[first_pos].radius,
                color="generic"
            ))
        
        return merged
    
    def _sort_colors_wubrg(self, symbols: List[ManaSymbol]) -> List[ManaSymbol]:
        """Sort color symbols in WUBRG order, prefer digits over colorless"""
        wubrg_order = {'W': 0, 'U': 1, 'B': 2, 'R': 3, 'G': 4}
        
        # Separate colors, digits, and others
        colors = []
        digits = []
        others = []
        
        for symbol in symbols:
            if symbol.symbol in wubrg_order:
                colors.append(symbol)
            elif symbol.symbol.isdigit():
                digits.append(symbol)
            else:
                others.append(symbol)
        
        # Sort colors by WUBRG order
        colors.sort(key=lambda s: wubrg_order.get(s.symbol, 999))
        
        # Reconstruct with colors in order, digits preferred over others
        result = []
        color_idx = 0
        digit_idx = 0
        other_idx = 0
        
        for symbol in symbols:
            if symbol.symbol in wubrg_order:
                result.append(colors[color_idx])
                color_idx += 1
            elif symbol.symbol.isdigit():
                result.append(digits[digit_idx])
                digit_idx += 1
            else:
                result.append(others[other_idx])
                other_idx += 1
        
        return result
    
    def _validate_symbol_gaps(self, symbols: List[ManaSymbol]) -> List[ManaSymbol]:
        """Validate reasonable gaps between symbols"""
        if len(symbols) <= 1:
            return symbols
        
        validated = [symbols[0]]
        
        for i in range(1, len(symbols)):
            prev_symbol = validated[-1]
            curr_symbol = symbols[i]
            
            # Calculate distance between centers
            distance = np.sqrt((curr_symbol.center[0] - prev_symbol.center[0])**2 + 
                             (curr_symbol.center[1] - prev_symbol.center[1])**2)
            
            # Expected gap: 0.6-2.0 times radius
            min_gap = 0.6 * (prev_symbol.radius + curr_symbol.radius)
            max_gap = 2.0 * (prev_symbol.radius + curr_symbol.radius)
            
            if min_gap <= distance <= max_gap:
                validated.append(curr_symbol)
            else:
                # Gap is unreasonable, might be a false positive
                # Keep the one with higher confidence
                if curr_symbol.confidence > prev_symbol.confidence:
                    validated[-1] = curr_symbol
        
        return validated 
    
    def _get_border_reference(self, image: np.ndarray, mana_region: Optional[Tuple[int, int, int, int]]) -> float:
        """Get improved border reference using low-chroma pixels from card border and rules text"""
        h, w = image.shape[:2]
        
        # Create cache key
        cache_key = f"{w}x{h}"
        if cache_key in self.border_cache:
            return self.border_cache[cache_key]
        
        # Sample border and rules text regions for low-chroma pixels
        low_chroma_pixels = []
        
        # Sample card border (top, bottom, left, right edges)
        border_width = 20
        border_regions = [
            (0, 0, w, border_width),  # Top border
            (0, h-border_width, w, h),  # Bottom border
            (0, 0, border_width, h),  # Left border
            (w-border_width, 0, w, h)  # Right border
        ]
        
        # Sample rules text area (bottom portion of card)
        rules_height = int(h * 0.3)
        rules_region = (border_width, h-rules_height, w-border_width, h-border_width)
        
        # Collect low-chroma pixels from all regions
        for x1, y1, x2, y2 in border_regions + [rules_region]:
            region = image[y1:y2, x1:x2]
            if region.size > 0:
                # Convert to LAB
                lab_region = self._convert_to_lab_proper(region)
                lab_region = lab_region.reshape(-1, 3)
                
                # Calculate chroma for each pixel
                chroma_values = np.sqrt(lab_region[:, 1]**2 + lab_region[:, 2]**2)
                
                # Keep pixels with low chroma (C* < 6)
                low_chroma_mask = chroma_values < 6
                if np.any(low_chroma_mask):
                    low_chroma_pixels.extend(lab_region[low_chroma_mask, 0])  # L* values
        
        # Calculate 90th percentile of L* values
        if low_chroma_pixels:
            l_border_ref = np.percentile(low_chroma_pixels, 90)
        else:
            # Fallback to simplified estimation
            l_border_ref = 85.0
        
        # Cache the result
        self.border_cache[cache_key] = l_border_ref
        
        return l_border_ref
    
    def _delta_e00(self, l1: float, a1: float, b1: float, l2: float, a2: float, b2: float) -> float:
        """Calculate ΔE₀₀ (CIE2000) color difference between two LAB colors"""
        # Simplified ΔE₀₀ calculation (full implementation would be more complex)
        # For now, use weighted Euclidean distance as approximation
        
        # Weights for LAB components (L* is more important for white/green distinction)
        w_l = 1.0
        w_a = 2.0
        w_b = 2.0
        
        # Calculate weighted distance
        delta_l = l1 - l2
        delta_a = a1 - a2
        delta_b = b1 - b2
        
        delta_e = np.sqrt(w_l * delta_l**2 + w_a * delta_a**2 + w_b * delta_b**2)
        
        return delta_e
    
    def _compute_center_entropy(self, region: np.ndarray, radius: int) -> float:
        """Compute entropy of the center region for digit detection"""
        h, w = region.shape[:2]
        center_x, center_y = w // 2, h // 2
        crop_radius = int(radius * 0.55)
        
        x1 = max(0, center_x - crop_radius)
        y1 = max(0, center_y - crop_radius)
        x2 = min(w, center_x + crop_radius)
        y2 = min(h, center_y + crop_radius)
        
        center_crop = region[y1:y2, x1:x2]
        
        if center_crop.size == 0:
            return 0.0
        
        # Convert to grayscale
        if len(center_crop.shape) == 3:
            gray = cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = center_crop
        
        # Compute entropy
        return self._compute_entropy(gray)