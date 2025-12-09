"""
MediaPipe Tracking System
Handles pose and face tracking using MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
from config import *
from utils import calculate_distance, smooth_value


class BodyTracker:
    """Tracks full body pose using MediaPipe Pose"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=POSE_CONFIDENCE,
            min_tracking_confidence=POSE_TRACKING_CONFIDENCE
        )
        
        # Smoothed landmarks
        self.smoothed_landmarks = None
        
    def process(self, frame):
        """Process frame and return pose landmarks"""
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Smooth landmarks
            if self.smoothed_landmarks is None:
                self.smoothed_landmarks = results.pose_landmarks
            else:
                self._smooth_landmarks(results.pose_landmarks)
            
            return self.smoothed_landmarks
        
        return None
    
    def _smooth_landmarks(self, new_landmarks):
        """Apply smoothing to landmarks"""
        for i, landmark in enumerate(new_landmarks.landmark):
            old_landmark = self.smoothed_landmarks.landmark[i]
            
            old_landmark.x = smooth_value(old_landmark.x, landmark.x, SMOOTHING_FACTOR)
            old_landmark.y = smooth_value(old_landmark.y, landmark.y, SMOOTHING_FACTOR)
            old_landmark.z = smooth_value(old_landmark.z, landmark.z, SMOOTHING_FACTOR)
            old_landmark.visibility = landmark.visibility
    
    def get_landmark_position(self, landmarks, index, frame_width, frame_height):
        """Get pixel position of a landmark"""
        if landmarks is None:
            return None
        
        landmark = landmarks.landmark[index]
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        
        return (x, y)
    
    def calculate_scale_factor(self, landmarks, frame_width, frame_height):
        """Calculate scale factor based on shoulder width"""
        if landmarks is None:
            return 1.0
        
        # Get shoulder positions
        left_shoulder = self.get_landmark_position(landmarks, self.mp_pose.PoseLandmark.LEFT_SHOULDER, 
                                                    frame_width, frame_height)
        right_shoulder = self.get_landmark_position(landmarks, self.mp_pose.PoseLandmark.RIGHT_SHOULDER, 
                                                     frame_width, frame_height)
        
        if left_shoulder is None or right_shoulder is None:
            return 1.0
        
        # Calculate shoulder width
        shoulder_width = calculate_distance(left_shoulder, right_shoulder)
        
        # Normalize to reference distance
        reference_width = frame_width * REFERENCE_DISTANCE
        scale = shoulder_width / reference_width
        
        # Clamp scale
        scale = max(MIN_SCALE, min(MAX_SCALE, scale))
        
        return scale
    
    def close(self):
        """Release resources"""
        self.pose.close()


class FaceTracker:
    """Tracks face mesh using MediaPipe Face Mesh"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=FACE_CONFIDENCE,
            min_tracking_confidence=FACE_TRACKING_CONFIDENCE
        )
        
        # Smoothed values
        self.smoothed_left_eye_ratio = 1.0
        self.smoothed_right_eye_ratio = 1.0
        self.smoothed_mouth_open = 0.0
        self.smoothed_mouth_smile = 0.0
        
    def process(self, frame):
        """Process frame and return face landmarks"""
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        
        return None
    
    def get_eye_ratios(self, face_landmarks, frame_width, frame_height):
        """Get eye aspect ratios for blink detection"""
        if face_landmarks is None:
            return self.smoothed_left_eye_ratio, self.smoothed_right_eye_ratio
        
        # Left eye landmarks (simplified)
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        # Right eye landmarks (simplified)
        right_eye_indices = [362, 385, 387, 263, 373, 380]
        
        # Get left eye ratio
        left_eye_points = [self._get_landmark_pos(face_landmarks, i, frame_width, frame_height) 
                          for i in left_eye_indices]
        left_ratio = self._calculate_eye_ratio(left_eye_points)
        
        # Get right eye ratio
        right_eye_points = [self._get_landmark_pos(face_landmarks, i, frame_width, frame_height) 
                           for i in right_eye_indices]
        right_ratio = self._calculate_eye_ratio(right_eye_points)
        
        # Smooth ratios
        self.smoothed_left_eye_ratio = smooth_value(
            self.smoothed_left_eye_ratio, left_ratio, BLINK_SMOOTHING
        )
        self.smoothed_right_eye_ratio = smooth_value(
            self.smoothed_right_eye_ratio, right_ratio, BLINK_SMOOTHING
        )
        
        return self.smoothed_left_eye_ratio, self.smoothed_right_eye_ratio
    
    def get_mouth_expression(self, face_landmarks, frame_width, frame_height):
        """Get mouth expression parameters"""
        if face_landmarks is None:
            return self.smoothed_mouth_open, self.smoothed_mouth_smile
        
        # Mouth landmarks
        mouth_indices = [61, 291, 0, 17]  # Left, Right, Top, Bottom
        
        mouth_points = [self._get_landmark_pos(face_landmarks, i, frame_width, frame_height) 
                       for i in mouth_indices]
        
        # Calculate mouth metrics
        mouth_width = calculate_distance(mouth_points[0], mouth_points[1])
        mouth_height = calculate_distance(mouth_points[2], mouth_points[3])
        
        # Normalize
        face_width = frame_width * 0.15  # Approximate face width
        mouth_open = mouth_height / face_width
        mouth_smile = mouth_width / face_width
        
        # Smooth values
        self.smoothed_mouth_open = smooth_value(
            self.smoothed_mouth_open, mouth_open, EXPRESSION_SMOOTHING
        )
        self.smoothed_mouth_smile = smooth_value(
            self.smoothed_mouth_smile, mouth_smile, EXPRESSION_SMOOTHING
        )
        
        return self.smoothed_mouth_open, self.smoothed_mouth_smile
    
    def _get_landmark_pos(self, face_landmarks, index, width, height):
        """Get pixel position of face landmark"""
        landmark = face_landmarks.landmark[index]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        return (x, y)
    
    def _calculate_eye_ratio(self, eye_points):
        """Calculate eye aspect ratio"""
        if len(eye_points) < 6:
            return 1.0
        
        # Vertical distances
        v1 = calculate_distance(eye_points[1], eye_points[5])
        v2 = calculate_distance(eye_points[2], eye_points[4])
        
        # Horizontal distance
        h = calculate_distance(eye_points[0], eye_points[3])
        
        # EAR formula
        ear = (v1 + v2) / (2.0 * h + 1e-6)
        return ear
    
    def close(self):
        """Release resources"""
        self.face_mesh.close()