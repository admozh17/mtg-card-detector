#!/usr/bin/env python3
"""
Enhanced Border Detection with Glare-Robust Edge Detection
Implements advanced techniques for reliable MTG card border detection
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import math

@dataclass
class CardBorders:
    top: int
    bottom: int
    left: int
    right: int

@dataclass
class Quadrilateral:
    corners: np.ndarray  # 4x2 array of corner points
    aspect_ratio: float
    area: float
    score: float

def canny_glare_robust(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build glare-robust edge map"""
    # Downscale to ~600px width for speed/precision balance
    h, w = image.shape[:2]
    scale_factor = 600.0 / w
    if scale_factor < 1.0:
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        image = cv2.resize(image, (new_w, new_h))
    
    # Convert to grayscale using min channel to suppress specular glare
    gray = np.min(image, axis=2).astype(np.uint8)
    
    # Light denoise
    gray = cv2.GaussianBlur(gray, (3, 3), 0.8)
    
    # Edge map with adaptive thresholds
    v = np.median(gray)
    low_thresh = int(0.66 * v)
    high_thresh = int(1.33 * v)
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    
    # Optional: black-hat to emphasize dark rim
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 11))
    
    blackhat_h = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_h)
    blackhat_v = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_v)
    rim = np.maximum(blackhat_h, blackhat_v)
    
    # Normalize and combine
    rim = cv2.normalize(rim, None, 0, 255, cv2.NORM_MINMAX)
    t_rim = np.mean(rim) + np.std(rim)
    rim_mask = (rim > t_rim).astype(np.uint8) * 255
    
    # Combine edges with rim
    edges = cv2.bitwise_or(edges, rim_mask)
    
    return edges, gray

def create_periphery_mask(h: int, w: int, p: float = 0.12) -> np.ndarray:
    """Create periphery mask for outer bands"""
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Define outer bands
    p_h = int(h * p)
    p_w = int(w * p)
    
    # Horizontal bands
    mask[:p_h, :] = 255
    mask[h-p_h:, :] = 255
    
    # Vertical bands
    mask[:, :p_w] = 255
    mask[:, w-p_w:] = 255
    
    return mask

def contours_in_periphery(edges: np.ndarray, p: float = 0.12) -> List[Tuple]:
    """Find contours in periphery with hierarchy"""
    h, w = edges.shape
    
    # Create periphery mask
    periphery_mask = create_periphery_mask(h, w, p)
    
    # Keep only edges in periphery
    edges_periph = cv2.bitwise_and(edges, periphery_mask)
    
    # Dilate lightly to bridge thin breaks
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    edges_periph = cv2.dilate(edges_periph, kernel_h, iterations=1)
    edges_periph = cv2.dilate(edges_periph, kernel_v, iterations=1)
    
    # Find contours with hierarchy
    contours, hierarchy = cv2.findContours(edges_periph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is None:
        return []
    
    # Filter contours
    valid_contours = []
    for i, contour in enumerate(contours):
        # Get bounding box
        x, y, w_rect, h_rect = cv2.boundingRect(contour)
        
        # Discard if bounding box leaves >=15% margin from any image side
        margin_left = x / w
        margin_right = (w - (x + w_rect)) / w
        margin_top = y / h
        margin_bottom = (h - (y + h_rect)) / h
        
        if min(margin_left, margin_right, margin_top, margin_bottom) >= 0.15:
            continue
        
        # Check if contour touches >=3 sides within δ=6-10px
        touches_sides = 0
        delta = 8
        
        if x <= delta: touches_sides += 1
        if x + w_rect >= w - delta: touches_sides += 1
        if y <= delta: touches_sides += 1
        if y + h_rect >= h - delta: touches_sides += 1
        
        # Keep parent contours (have children) or contours touching >=3 sides
        has_children = hierarchy[0][i][2] != -1
        if has_children or touches_sides >= 3:
            valid_contours.append((contour, hierarchy[0][i], x, y, w_rect, h_rect))
    
    return valid_contours

def fit_quadrilateral(contour: np.ndarray) -> Optional[np.ndarray]:
    """Fit quadrilateral to contour"""
    # Try approximation first
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) == 4:
        return approx.reshape(4, 2)
    
    # If approximation failed, use RANSAC line fitting
    # This is a simplified version - in practice you'd use a proper RANSAC implementation
    hull = cv2.convexHull(contour)
    if len(hull) < 4:
        return None
    
    # Find the 4 extreme points
    leftmost = tuple(hull[hull[:, :, 0].argmin()][0])
    rightmost = tuple(hull[hull[:, :, 0].argmax()][0])
    topmost = tuple(hull[hull[:, :, 1].argmin()][0])
    bottommost = tuple(hull[hull[:, :, 1].argmax()][0])
    
    corners = np.array([leftmost, topmost, rightmost, bottommost], dtype=np.float32)
    
    # Sort corners in clockwise order
    center = np.mean(corners, axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    corners = corners[sorted_indices]
    
    return corners

def aspect_snap_and_refine(quad: np.ndarray, target_aspect: float = 1.397, search_px: int = 6) -> np.ndarray:
    """Snap quadrilateral to target aspect ratio using side-aware adjustment"""
    # Calculate current aspect ratio using bounding rectangle
    x_coords = quad[:, 0]
    y_coords = quad[:, 1]
    width = np.max(x_coords) - np.min(x_coords)
    height = np.max(y_coords) - np.min(y_coords)
    current_aspect = height / width
    
    # If aspect is already close to target, just return
    if abs(current_aspect - target_aspect) < 0.05:
        return quad
    
    # Calculate adjustment needed
    aspect_ratio = target_aspect / current_aspect
    
    # Create a copy to avoid modifying the original
    refined_quad = quad.copy()
    
    # Side-aware adjustment: move each side along its normal
    # This avoids the center-scaling bias that pushes the top upward
    
    # Calculate center for reference
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)
    
    # Adjust height by moving top and bottom sides
    # Top side (lowest Y coordinates)
    top_indices = np.where(y_coords == np.min(y_coords))[0]
    bottom_indices = np.where(y_coords == np.max(y_coords))[0]
    
    # Calculate how much to move each side
    current_height = height
    target_height = width * target_aspect
    height_diff = target_height - current_height
    
    # Move top side up and bottom side down proportionally
    for idx in top_indices:
        refined_quad[idx, 1] -= height_diff * 0.5  # Move top up
    
    for idx in bottom_indices:
        refined_quad[idx, 1] += height_diff * 0.5  # Move bottom down
    
    return refined_quad

def border_uniformity(quad: np.ndarray, gray: np.ndarray, N: int = 40) -> Tuple[float, float, float]:
    """Measure border uniformity"""
    h, w = gray.shape
    widths = []
    invalid_samples = 0
    
    # Sample along each side
    for side_idx in range(4):
        p1 = quad[side_idx]
        p2 = quad[(side_idx + 1) % 4]
        
        # Sample N points along this side
        for i in range(N):
            t = i / (N - 1)
            sample_point = p1 + t * (p2 - p1)
            x, y = int(sample_point[0]), int(sample_point[1])
            
            if not (0 <= x < w and 0 <= y < h):
                invalid_samples += 1
                continue
            
            # Calculate normal direction (simplified)
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = np.sqrt(dx*dx + dy*dy)
            if length < 1e-6:  # Avoid division by zero
                invalid_samples += 1
                continue
            
            # Normal vector (perpendicular to side)
            nx = -dy / length
            ny = dx / length
            
            # Search for outer and inner rims
            outer_rim = None
            inner_rim = None
            
            # Search outward
            for d in range(1, 20):
                ox, oy = int(x + nx * d), int(y + ny * d)
                if not (0 <= ox < w and 0 <= oy < h):
                    break
                # Simplified: look for strong gradient
                if ox > 0 and ox < w-1:
                    grad = abs(int(gray[oy, ox+1]) - int(gray[oy, ox-1]))
                    if grad > 30:  # Threshold for strong edge
                        outer_rim = d
                        break
            
            # Search inward
            for d in range(1, 20):
                ix, iy = int(x - nx * d), int(y - ny * d)
                if not (0 <= ix < w and 0 <= iy < h):
                    break
                # Simplified: look for strong gradient
                if ix > 0 and ix < w-1:
                    grad = abs(int(gray[iy, ix+1]) - int(gray[iy, ix-1]))
                    if grad > 30:  # Threshold for strong edge
                        inner_rim = d
                        break
            
            if outer_rim is not None and inner_rim is not None:
                width = outer_rim + inner_rim
                widths.append(width)
            else:
                invalid_samples += 1
    
    if not widths:
        return 0.0, 0.0, 1.0
    
    mu_w = np.mean(widths)
    sigma_w = np.std(widths)
    U = 1.0 - (sigma_w / (mu_w + 1e-6))
    invalid_rate = invalid_samples / (4 * N)
    
    return U, mu_w, invalid_rate

def perimeter_energy(quad: np.ndarray, gray: np.ndarray, ring_width: int = 6) -> float:
    """Calculate edge energy along quadrilateral perimeter"""
    h, w = gray.shape
    energies = []
    
    # Calculate gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Sample along each side
    for side_idx in range(4):
        p1 = quad[side_idx]
        p2 = quad[(side_idx + 1) % 4]
        
        # Sample points along this side
        num_samples = 50
        side_energy = 0
        valid_samples = 0
        
        for i in range(num_samples):
            t = i / (num_samples - 1)
            sample_point = p1 + t * (p2 - p1)
            x, y = int(sample_point[0]), int(sample_point[1])
            
            if not (0 <= x < w and 0 <= y < h):
                continue
            
            # Sample in a ring around the point
            ring_energy = 0
            ring_count = 0
            
            for dx in range(-ring_width//2, ring_width//2 + 1):
                for dy in range(-ring_width//2, ring_width//2 + 1):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        ring_energy += grad_magnitude[ny, nx]
                        ring_count += 1
            
            if ring_count > 0:
                side_energy += ring_energy / ring_count
                valid_samples += 1
        
        if valid_samples > 0:
            energies.append(side_energy / valid_samples)
    
    return np.mean(energies) if energies else 0.0

def calculate_score(quad: np.ndarray, gray: np.ndarray, image_area: float) -> float:
    """Calculate composite score for quadrilateral"""
    h, w = gray.shape
    
    # Calculate basic properties
    x_coords = quad[:, 0]
    y_coords = quad[:, 1]
    quad_width = np.max(x_coords) - np.min(x_coords)
    quad_height = np.max(y_coords) - np.min(y_coords)
    aspect = quad_height / quad_width
    
    # Calculate area
    quad_area = cv2.contourArea(quad.astype(np.int32))
    area_ratio = quad_area / image_area
    
    # Aspect closeness score
    target_aspect = 1.397
    aspect_score = np.exp(-abs(np.log(aspect / target_aspect)))
    
    # Perimeter energy
    E = perimeter_energy(quad, gray)
    energy_score = E / (E + 30)
    
    # Distance to image edge
    center_x, center_y = np.mean(x_coords), np.mean(y_coords)
    dist_to_edge = min(center_x, center_y, w - center_x, h - center_y)
    dist_score = 1.0 / (1.0 + dist_to_edge)
    
    # Border uniformity
    U, mu_w, invalid_rate = border_uniformity(quad, gray)
    
    # Composite score
    score = (0.30 * area_ratio + 
             0.20 * aspect_score + 
             0.20 * energy_score + 
             0.15 * dist_score + 
             0.15 * U)
    
    return score

def detect_mtg_black_border(image: np.ndarray) -> Optional[np.ndarray]:
    """Specifically detect MTG card black border"""
    h, w = image.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply morphological operations to enhance black borders
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Use multiple approaches to find the black border
    
    # Method 1: Threshold for dark regions (black border)
    _, dark_thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    dark_thresh = cv2.bitwise_not(dark_thresh)  # Invert so black becomes white
    
    # Method 2: Edge detection with focus on strong edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Combine both methods
    combined = cv2.bitwise_or(dark_thresh, edges)
    
    # Clean up with morphological operations
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Filter contours by area and aspect ratio
    min_area = (w * h) * 0.3  # Card should be at least 30% of image
    valid_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
            
        # Check if contour has reasonable aspect ratio
        x, y, w_rect, h_rect = cv2.boundingRect(contour)
        aspect = h_rect / w_rect
        
        # MTG cards are roughly 1.4:1 ratio
        if 1.2 <= aspect <= 1.6:
            valid_contours.append((contour, area))
    
    if not valid_contours:
        return None
    
    # Get the largest valid contour
    largest_contour = max(valid_contours, key=lambda x: x[1])[0]
    
    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # If we get a quadrilateral, great! If not, get the bounding rect
    if len(approx) == 4:
        return approx.reshape(4, 2)
    else:
        # Use convex hull and find corners
        hull = cv2.convexHull(largest_contour)
        epsilon = 0.01 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        if len(approx) >= 4:
            # Get the 4 corner points
            points = approx.reshape(-1, 2)
            
            # Find corners by sorting points
            # Top-left: min(x+y), Top-right: max(x-y), Bottom-right: max(x+y), Bottom-left: min(x-y)
            sum_coords = points.sum(axis=1)
            diff_coords = np.diff(points, axis=1).flatten()
            
            top_left = points[np.argmin(sum_coords)]
            bottom_right = points[np.argmax(sum_coords)]
            top_right = points[np.argmax(diff_coords)]
            bottom_left = points[np.argmin(diff_coords)]
            
            return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    
    return None

def detect_card_contour(image: np.ndarray) -> Optional[np.ndarray]:
    """Legacy function - redirects to black border detection"""
    return detect_mtg_black_border(image)

def detect_borders_enhanced(image: np.ndarray) -> CardBorders:
    """Enhanced border detection - find actual card edges"""
    h, w = image.shape[:2]
    
    # First try to find the MTG black border specifically
    card_contour = detect_mtg_black_border(image)
    
    if card_contour is not None:
        # Get bounding box from the contour
        x_coords = card_contour[:, 0]
        y_coords = card_contour[:, 1]
        
        left = int(np.min(x_coords))
        right = int(np.max(x_coords))
        top = int(np.min(y_coords))
        bottom = int(np.max(y_coords))
        
        # Ensure we don't go outside image bounds
        left = max(0, left)
        right = min(w, right)
        top = max(0, top)
        bottom = min(h, bottom)
        
        print(f"    DEBUG: MTG black border found - borders: L={left}, R={right}, T={top}, B={bottom}")
        return CardBorders(top=top, bottom=bottom, left=left, right=right)
    
    # Fallback to the original complex detection
    print(f"    DEBUG: Card contour not found, using original detection")
    
    # Step A: Build glare-robust edge map
    edges, gray = canny_glare_robust(image)
    
    # Get the scale factor used in canny_glare_robust
    scale_factor = 600.0 / w if w > 600 else 1.0
    
    # Step B: Periphery-first candidate generation
    candidates = contours_in_periphery(edges, p=0.12)
    
    if not candidates:
        # Fallback to simple margins
        margin = int(min(w, h) * 0.05)
        print(f"    DEBUG: No candidates found, using simple fallback")
        return CardBorders(top=margin, bottom=h-margin, left=margin, right=w-margin)
    
    print(f"    DEBUG: Found {len(candidates)} candidates")
    
    # Step C & D: Quadrilateral fitting and aspect snapping
    scored_candidates = []
    image_area = h * w
    
    for i, (contour, hierarchy, x, y, w_rect, h_rect) in enumerate(candidates):
        # Fit quadrilateral
        quad = fit_quadrilateral(contour)
        if quad is None:
            print(f"    DEBUG: Candidate {i}: Failed to fit quadrilateral")
            continue
        
        # Check aspect ratio
        quad_width = np.max(quad[:, 0]) - np.min(quad[:, 0])
        quad_height = np.max(quad[:, 1]) - np.min(quad[:, 1])
        aspect = quad_height / quad_width
        
        if not (1.30 <= aspect <= 1.50):
            print(f"    DEBUG: Candidate {i}: Aspect ratio {aspect:.3f} out of range")
            continue
        
        # Aspect snap and refine
        quad = aspect_snap_and_refine(quad)
        
        # Check border uniformity
        U, mu_w, invalid_rate = border_uniformity(quad, gray)
        if mu_w < 2 or U < 0.6 or invalid_rate > 0.25:
            print(f"    DEBUG: Candidate {i}: Border uniformity failed (U={U:.3f}, mu_w={mu_w:.1f}, invalid={invalid_rate:.3f})")
            continue
        
        # Check area ratio
        quad_area = cv2.contourArea(quad.astype(np.int32))
        area_ratio = quad_area / image_area
        if area_ratio < 0.55:
            print(f"    DEBUG: Candidate {i}: Area ratio {area_ratio:.3f} too small")
            continue
        
        # Calculate score
        score = calculate_score(quad, gray, image_area)
        
        if score >= 0.62:
            print(f"    DEBUG: Candidate {i}: Score {score:.3f} >= 0.62, accepted")
            scored_candidates.append((score, quad))
        else:
            print(f"    DEBUG: Candidate {i}: Score {score:.3f} < 0.62, rejected")
    
    # If no candidates found, relax periphery and retry
    if not scored_candidates:
        print(f"    DEBUG: No scored candidates, relaxing periphery to 15%")
        candidates = contours_in_periphery(edges, p=0.15)
        for i, (contour, hierarchy, x, y, w_rect, h_rect) in enumerate(candidates):
            quad = fit_quadrilateral(contour)
            if quad is None:
                continue
            
            quad_width = np.max(quad[:, 0]) - np.min(quad[:, 0])
            quad_height = np.max(quad[:, 1]) - np.min(quad[:, 1])
            aspect = quad_height / quad_width
            
            if not (1.30 <= aspect <= 1.50):
                continue
            
            quad = aspect_snap_and_refine(quad)
            U, mu_w, invalid_rate = border_uniformity(quad, gray)
            if mu_w < 2 or U < 0.6 or invalid_rate > 0.25:
                continue
            
            quad_area = cv2.contourArea(quad.astype(np.int32))
            area_ratio = quad_area / image_area
            if area_ratio < 0.55:
                continue
            
            score = calculate_score(quad, gray, image_area)
            if score >= 0.62:
                print(f"    DEBUG: Relaxed candidate {i}: Score {score:.3f} >= 0.62, accepted")
                scored_candidates.append((score, quad))
    
    # Select best candidate
    if scored_candidates:
        best_score, best_quad = max(scored_candidates, key=lambda x: x[0])
        print(f"    DEBUG: Using best candidate with score {best_score:.3f}")
        
        # Convert back to original scale if needed
        if scale_factor < 1.0:
            best_quad = best_quad / scale_factor
        
        # Get bounding rectangle
        x_coords = best_quad[:, 0]
        y_coords = best_quad[:, 1]
        
        left = int(np.min(x_coords))
        right = int(np.max(x_coords))
        top = int(np.min(y_coords))
        bottom = int(np.max(y_coords))
        
        # Add small margin
        margin = 12
        return CardBorders(
            top=max(0, top + margin),
            bottom=min(h, bottom - margin),
            left=max(0, left + margin),
            right=min(w, right - margin)
        )
    
    # Final fallback to simple margins
    print(f"    DEBUG: No valid candidates, using simple fallback")
    margin = int(min(w, h) * 0.05)
    return CardBorders(top=margin, bottom=h-margin, left=margin, right=w-margin)

# Keep the original simple method for comparison
def detect_borders_simple(image: np.ndarray) -> CardBorders:
    """Simple border detection - just use fixed margins"""
    h, w = image.shape[:2]
    
    # Use fixed margins based on typical MTG card proportions
    margin_x = int(w * 0.05)  # 5% margin
    margin_y = int(h * 0.05)  # 5% margin
    
    return CardBorders(
        top=margin_y,
        bottom=h - margin_y,
        left=margin_x,
        right=w - margin_x
    ) 