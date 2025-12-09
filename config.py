"""
VTuber Configuration File
Contains all settings and constants for the VTuber system
"""

import os

# ==================== PATH SETTINGS ====================
ASSETS_PATH = "avatar"  # Folder containing sprite PNG files

# ==================== WINDOW SETTINGS ====================
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_NAME = "VTuber - Full Body Tracking"
FPS_TARGET = 60

# ==================== CAMERA SETTINGS ====================
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
MIRROR_MODE = True  # Mirror the webcam like a mirror

# ==================== CHARACTER SETTINGS ====================
# Base character scale (will be adjusted by distance)
BASE_SCALE = 1.2

# Distance-based scaling
MIN_SCALE = 0.7
MAX_SCALE = 2.0
REFERENCE_DISTANCE = 0.35  # Reference shoulder width for scaling

# Character vertical offset (to position character properly)
CHARACTER_Y_OFFSET = 0

# ==================== FLOATING JOINT SETTINGS ====================
# Maximum floating distance in pixels
FLOATING_DISTANCE = 6

# Floating speed/frequency for each body part (higher = faster floating)
FLOATING_SPEEDS = {
    'head': 2.0,
    'torso': 1.0,
    'left_upper_arm': 1.5,
    'right_upper_arm': 1.5,
    'left_lower_arm': 2.0,
    'right_lower_arm': 2.0,
    'left_hand': 2.5,
    'right_hand': 2.5,
    'left_upper_leg': 1.2,
    'right_upper_leg': 1.2,
    'left_lower_leg': 1.8,
    'right_lower_leg': 1.8,
    'tail': 1.5
}

# Smoothing factor for movement (0.0 - 1.0, higher = more smoothing)
SMOOTHING_FACTOR = 0.3

# ==================== EXPRESSION SETTINGS ====================
# Eye blink detection thresholds
EYE_ASPECT_RATIO_THRESHOLD = 0.18  # Below this = eyes closed (more sensitive)
BLINK_SMOOTHING = 0.1  # Faster blink response

# Mouth expression thresholds
MOUTH_OPEN_THRESHOLD = 0.025  # Mouth starts opening
MOUTH_SMILE_THRESHOLD = 0.48  # Width/height ratio for smile

# Expression smoothing
EXPRESSION_SMOOTHING = 0.15  # Faster expression response

# ==================== SPRITE FILE NAMES ====================
SPRITE_FILES = {
    # Head
    'head': 'head.png',
    
    # Eyes
    'eye_left_open': 'eyeLeftOpened.png',
    'eye_left_closed': 'eyeLeftClosed.png',
    'eye_right_open': 'eyeRightOpened.png',
    'eye_right_closed': 'eyeRightClosed.png',
    
    # Mouth
    'mouth_neutral': 'MouthNetral.png',
    'mouth_opened': 'MouthOpened.png',
    'mouth_smile': 'MouthSmile.png',
    
    # Arms - Left
    'arm_left_upper': 'ArmLeftUpper.png',
    'arm_left_middle': 'ArmLeftMiddle.png',
    'arm_left_lower': 'ArmLeftLower.png',
    
    # Arms - Right
    'arm_right_upper': 'ArmRightUpper.png',
    'arm_right_middle': 'ArmRightMiddle.png',
    'arm_right_lower': 'ArmRightLower.png',
    
    # Legs - Left
    'leg_left_upper': 'LegLeftUpper.png',
    'leg_left_middle': 'LegLeftMiddle.png',
    'leg_left_lower': 'LegLeftLower.png',
    
    # Legs - Right
    'leg_right_upper': 'LegRightUpper.png',
    'leg_right_middle': 'LegRightMiddle.png',
    'leg_right_lower': 'LegRightLower.png',
    
    # Body and tail
    'body_tail': 'bodyTails.png',
}

# ==================== MEDIAPIPE SETTINGS ====================
POSE_CONFIDENCE = 0.5
POSE_TRACKING_CONFIDENCE = 0.5
FACE_CONFIDENCE = 0.5
FACE_TRACKING_CONFIDENCE = 0.5

# ==================== DEBUG SETTINGS ====================
SHOW_SKELETON = False  # Show MediaPipe skeleton overlay
SHOW_FPS = True  # Show FPS counter
DEBUG_MODE = False  # Print debug information