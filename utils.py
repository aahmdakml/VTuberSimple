"""
Utility Functions for VTuber System
Contains helper functions for calculations, transformations, and utilities
"""

import cv2
import numpy as np
import math


def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def calculate_angle(point1, point2, point3):
    """
    Calculate angle at point2 formed by point1-point2-point3
    Returns angle in degrees
    """
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    
    ba = a - b
    bc = c - b
    
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    
    return np.degrees(angle)


def lerp(start, end, factor):
    """Linear interpolation between start and end"""
    return start + (end - start) * factor


def smooth_value(current, target, smoothing):
    """Smooth transition from current to target value"""
    return lerp(current, target, 1.0 - smoothing)


def rotate_point(point, center, angle):
    """Rotate a point around a center by angle (in degrees)"""
    angle_rad = math.radians(angle)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)
    
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    
    new_x = center[0] + dx * cos_angle - dy * sin_angle
    new_y = center[1] + dx * sin_angle + dy * cos_angle
    
    return (new_x, new_y)


def get_rotation_angle(point1, point2):
    """Get rotation angle from point1 to point2 in degrees"""
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    return math.degrees(math.atan2(dy, dx))


def rotate_image(image, angle, center=None):
    """Rotate an image around center point"""
    h, w = image.shape[:2]
    
    if center is None:
        center = (w // 2, h // 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), 
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))
    
    return rotated


def scale_image(image, scale):
    """Scale image by scale factor"""
    if scale == 1.0:
        return image
    
    h, w = image.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    if new_w == 0 or new_h == 0:
        return image
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def overlay_image(background, overlay, position, rotation=0):
    """
    Overlay an image with transparency onto background
    position: (x, y) center position
    rotation: rotation angle in degrees
    """
    if overlay is None or overlay.shape[0] == 0 or overlay.shape[1] == 0:
        return background
    
    # Rotate overlay if needed
    if rotation != 0:
        overlay = rotate_image(overlay, rotation)
    
    h, w = overlay.shape[:2]
    x, y = position
    
    # Calculate top-left position
    x1 = int(x - w // 2)
    y1 = int(y - h // 2)
    x2 = x1 + w
    y2 = y1 + h
    
    # Check bounds
    bg_h, bg_w = background.shape[:2]
    
    # Calculate overlay region
    overlay_x1 = max(0, -x1)
    overlay_y1 = max(0, -y1)
    overlay_x2 = w - max(0, x2 - bg_w)
    overlay_y2 = h - max(0, y2 - bg_h)
    
    # Calculate background region
    bg_x1 = max(0, x1)
    bg_y1 = max(0, y1)
    bg_x2 = min(bg_w, x2)
    bg_y2 = min(bg_h, y2)
    
    # Check if there's valid overlap
    if bg_x2 <= bg_x1 or bg_y2 <= bg_y1:
        return background
    
    if overlay_x2 <= overlay_x1 or overlay_y2 <= overlay_y1:
        return background
    
    # Extract regions
    overlay_crop = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    bg_crop = background[bg_y1:bg_y2, bg_x1:bg_x2]
    
    if overlay_crop.shape[0] == 0 or overlay_crop.shape[1] == 0:
        return background
    
    # Handle alpha channel
    if overlay_crop.shape[2] == 4:
        alpha = overlay_crop[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=2)
        
        overlay_rgb = overlay_crop[:, :, :3]
        
        # Blend
        blended = overlay_rgb * alpha + bg_crop * (1 - alpha)
        background[bg_y1:bg_y2, bg_x1:bg_x2] = blended.astype(np.uint8)
    else:
        background[bg_y1:bg_y2, bg_x1:bg_x2] = overlay_crop
    
    return background


def calculate_eye_aspect_ratio(eye_landmarks):
    """Calculate eye aspect ratio for blink detection"""
    if len(eye_landmarks) < 6:
        return 1.0
    
    # Vertical distances
    v1 = calculate_distance(eye_landmarks[1], eye_landmarks[5])
    v2 = calculate_distance(eye_landmarks[2], eye_landmarks[4])
    
    # Horizontal distance
    h = calculate_distance(eye_landmarks[0], eye_landmarks[3])
    
    # EAR formula
    ear = (v1 + v2) / (2.0 * h + 1e-6)
    return ear


def calculate_mouth_aspect_ratio(mouth_landmarks):
    """Calculate mouth aspect ratio for expression detection"""
    if len(mouth_landmarks) < 4:
        return 0.0, 0.0
    
    # Vertical distance (height)
    height = calculate_distance(mouth_landmarks[1], mouth_landmarks[3])
    
    # Horizontal distance (width)
    width = calculate_distance(mouth_landmarks[0], mouth_landmarks[2])
    
    # Ratios
    open_ratio = height / (width + 1e-6)
    smile_ratio = width / (height + 1e-6)
    
    return open_ratio, smile_ratio


def draw_fps(frame, fps):
    """Draw FPS counter on frame"""
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame


def create_blank_frame(width, height, color=(0, 0, 0)):
    """Create a blank frame with specified color"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = color
    return frame