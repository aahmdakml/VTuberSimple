"""
Rendering Engine
Handles all character rendering with floating joints
Fixed for actual sprite dimensions
"""

import cv2
import numpy as np
import mediapipe as mp
from config import *
from utils import overlay_image, get_rotation_angle, calculate_distance
from floating_system import FloatingSystem, PositionSmoother, RotationSmoother, TailAnimator
from expression import ExpressionController


class CharacterRenderer:
    """Renders character with tracking data"""
    
    def __init__(self, character):
        self.character = character
        self.floating_system = FloatingSystem()
        self.position_smoother = PositionSmoother(SMOOTHING_FACTOR)
        self.rotation_smoother = RotationSmoother(SMOOTHING_FACTOR)
        self.tail_animator = TailAnimator()
        self.expression_controller = ExpressionController()
        
        # MediaPipe pose landmarks
        self.mp_pose = mp.solutions.pose.PoseLandmark
        
        # Previous position for velocity calculation
        self.prev_body_pos = None
        
        # Sprite-specific scale adjustments (relative to base scale)
        # Based on actual sprite dimensions vs body proportions
        self.sprite_scales = {
            'head': 0.45,  # Head is 744x872, increase from 0.25
            'body_tail': 0.35,  # Body is 1293x1170, increase from 0.18
            'arm': 0.8,  # Arms are ~211-316px
            'leg': 0.8,  # Legs are ~137-275px
            'eye': 2.2,  # Eyes are 58x50-84
            'mouth': 1.0,  # Mouth is 203-211x35-82
        }
        
        # Offset multipliers for facial features (relative to head scale)
        self.face_offsets = {
            'eye_left_x': -60,
            'eye_left_y': -30,
            'eye_right_x': 60,
            'eye_right_y': -30,
            'mouth_x': 0,
            'mouth_y': 55,
        }
    
    def render(self, frame, pose_landmarks, face_tracker, body_tracker):
        """Render character onto frame"""
        if pose_landmarks is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # Calculate scale based on distance
        distance_scale = body_tracker.calculate_scale_factor(pose_landmarks, w, h)
        self.character.set_scale(distance_scale * BASE_SCALE)
        
        # Get all body positions
        positions = self._extract_positions(pose_landmarks, body_tracker, w, h)
        
        # Update expression
        if face_tracker:
            left_eye = face_tracker.smoothed_left_eye_ratio
            right_eye = face_tracker.smoothed_right_eye_ratio
            mouth_open = face_tracker.smoothed_mouth_open
            mouth_smile = face_tracker.smoothed_mouth_smile
            self.expression_controller.update(left_eye, right_eye, mouth_open, mouth_smile)
        
        # Calculate body velocity
        body_velocity = self._calculate_body_velocity(positions.get('torso'))
        
        # Render in correct z-order (back to front)
        frame = self._render_back_leg(frame, positions)
        frame = self._render_back_arm(frame, positions)
        frame = self._render_body(frame, positions, body_velocity)
        frame = self._render_front_leg(frame, positions)
        frame = self._render_front_arm(frame, positions)
        frame = self._render_head(frame, positions)
        
        return frame
    
    def _extract_positions(self, landmarks, body_tracker, width, height):
        """Extract and smooth all body part positions"""
        positions = {}
        
        # Get raw positions
        raw = {
            'nose': body_tracker.get_landmark_position(landmarks, self.mp_pose.NOSE, width, height),
            'left_shoulder': body_tracker.get_landmark_position(landmarks, self.mp_pose.LEFT_SHOULDER, width, height),
            'right_shoulder': body_tracker.get_landmark_position(landmarks, self.mp_pose.RIGHT_SHOULDER, width, height),
            'left_elbow': body_tracker.get_landmark_position(landmarks, self.mp_pose.LEFT_ELBOW, width, height),
            'right_elbow': body_tracker.get_landmark_position(landmarks, self.mp_pose.RIGHT_ELBOW, width, height),
            'left_wrist': body_tracker.get_landmark_position(landmarks, self.mp_pose.LEFT_WRIST, width, height),
            'right_wrist': body_tracker.get_landmark_position(landmarks, self.mp_pose.RIGHT_WRIST, width, height),
            'left_hip': body_tracker.get_landmark_position(landmarks, self.mp_pose.LEFT_HIP, width, height),
            'right_hip': body_tracker.get_landmark_position(landmarks, self.mp_pose.RIGHT_HIP, width, height),
            'left_knee': body_tracker.get_landmark_position(landmarks, self.mp_pose.LEFT_KNEE, width, height),
            'right_knee': body_tracker.get_landmark_position(landmarks, self.mp_pose.RIGHT_KNEE, width, height),
            'left_ankle': body_tracker.get_landmark_position(landmarks, self.mp_pose.LEFT_ANKLE, width, height),
            'right_ankle': body_tracker.get_landmark_position(landmarks, self.mp_pose.RIGHT_ANKLE, width, height),
        }
        
        # Calculate derived positions
        if raw['left_shoulder'] and raw['right_shoulder']:
            sc_x = (raw['left_shoulder'][0] + raw['right_shoulder'][0]) / 2
            sc_y = (raw['left_shoulder'][1] + raw['right_shoulder'][1]) / 2
            raw['shoulder_center'] = (sc_x, sc_y)
        
        if raw['left_hip'] and raw['right_hip']:
            hc_x = (raw['left_hip'][0] + raw['right_hip'][0]) / 2
            hc_y = (raw['left_hip'][1] + raw['right_hip'][1]) / 2
            raw['hip_center'] = (hc_x, hc_y)
        
        # Torso at midpoint between shoulder and hip centers
        if raw.get('shoulder_center') and raw.get('hip_center'):
            tc_x = (raw['shoulder_center'][0] + raw['hip_center'][0]) / 2
            tc_y = (raw['shoulder_center'][1] + raw['hip_center'][1]) / 2
            raw['torso'] = (tc_x, tc_y)
        
        # Head position (nose with offset)
        if raw['nose']:
            raw['head'] = raw['nose']
        
        # Apply smoothing and floating
        for key, pos in raw.items():
            if pos is not None:
                smoothed = self.position_smoother.smooth_position(key, pos)
                floated = self.floating_system.apply_offset(smoothed, key)
                positions[key] = floated
        
        return positions
    
    def _calculate_body_velocity(self, current_pos):
        """Calculate body movement velocity"""
        if current_pos is None or self.prev_body_pos is None:
            velocity = 0
        else:
            dist = calculate_distance(current_pos, self.prev_body_pos)
            velocity = min(1.0, dist / 50.0)
        
        self.prev_body_pos = current_pos
        return velocity
    
    def _get_scaled_sprite(self, sprite_name, sprite_type='default'):
        """Get sprite with proper scaling"""
        sprite = self.character.get_sprite(sprite_name)
        if sprite is None:
            return None
        
        # Apply sprite-specific scale adjustment
        type_scale = self.sprite_scales.get(sprite_type, 1.0)
        adjusted_scale = self.character.current_scale * type_scale
        
        if adjusted_scale != 1.0:
            from utils import scale_image
            return scale_image(sprite, adjusted_scale)
        
        return sprite
    
    def _render_head(self, frame, positions):
        """Render head with facial features"""
        if 'head' not in positions:
            return frame
        
        head_pos = positions['head']
        base_scale = self.character.current_scale
        
        # Render head base
        head = self._get_scaled_sprite('head', 'head')
        if head is not None:
            frame = overlay_image(frame, head, head_pos)
        
        # Get expression sprites
        left_eye_name, right_eye_name = self.expression_controller.get_eye_sprites()
        mouth_name = self.expression_controller.get_mouth_sprite()
        
        # Render eyes with proper scale and offset
        left_eye = self._get_scaled_sprite(left_eye_name, 'eye')
        if left_eye is not None:
            eye_x = head_pos[0] + self.face_offsets['eye_left_x'] * base_scale
            eye_y = head_pos[1] + self.face_offsets['eye_left_y'] * base_scale
            frame = overlay_image(frame, left_eye, (eye_x, eye_y))
        
        right_eye = self._get_scaled_sprite(right_eye_name, 'eye')
        if right_eye is not None:
            eye_x = head_pos[0] + self.face_offsets['eye_right_x'] * base_scale
            eye_y = head_pos[1] + self.face_offsets['eye_right_y'] * base_scale
            frame = overlay_image(frame, right_eye, (eye_x, eye_y))
        
        # Render mouth
        mouth = self._get_scaled_sprite(mouth_name, 'mouth')
        if mouth is not None:
            mouth_x = head_pos[0] + self.face_offsets['mouth_x'] * base_scale
            mouth_y = head_pos[1] + self.face_offsets['mouth_y'] * base_scale
            frame = overlay_image(frame, mouth, (mouth_x, mouth_y))
        
        return frame
    
    def _render_body(self, frame, positions, body_velocity):
        """Render body/torso with tail"""
        if 'torso' not in positions:
            return frame
        
        torso_pos = positions['torso']
        tail_rotation = self.tail_animator.get_tail_rotation(body_velocity)
        
        body_tail = self._get_scaled_sprite('body_tail', 'body_tail')
        if body_tail is not None:
            frame = overlay_image(frame, body_tail, torso_pos, rotation=tail_rotation)
        
        return frame
    
    def _render_back_arm(self, frame, positions):
        """Render back arm (right in mirror)"""
        return self._render_arm(frame, positions, 'right')
    
    def _render_front_arm(self, frame, positions):
        """Render front arm (left in mirror)"""
        return self._render_arm(frame, positions, 'left')
    
    def _render_arm(self, frame, positions, side):
        """Render arm segments - each segment connects joints properly"""
        shoulder = positions.get(f'{side}_shoulder')
        elbow = positions.get(f'{side}_elbow')
        wrist = positions.get(f'{side}_wrist')
        
        if not (shoulder and elbow and wrist):
            return frame
        
        # Calculate rotations based on actual joint connections
        # Upper arm: shoulder pointing to elbow
        upper_rot = get_rotation_angle(shoulder, elbow)
        # Lower arm: elbow pointing to wrist
        lower_rot = get_rotation_angle(elbow, wrist)
        
        # Smooth rotations
        upper_rot = self.rotation_smoother.smooth_rotation(f'{side}_upper_arm', upper_rot)
        lower_rot = self.rotation_smoother.smooth_rotation(f'{side}_lower_arm', lower_rot)
        
        # Render upper arm - positioned AT shoulder, pointing towards elbow
        upper = self._get_scaled_sprite(f'arm_{side}_upper', 'arm')
        if upper is not None:
            frame = overlay_image(frame, upper, shoulder, rotation=upper_rot)
        
        # Render middle arm - positioned AT elbow
        middle = self._get_scaled_sprite(f'arm_{side}_middle', 'arm')
        if middle is not None:
            frame = overlay_image(frame, middle, elbow, rotation=lower_rot)
        
        # Render lower arm/hand - positioned AT wrist
        lower = self._get_scaled_sprite(f'arm_{side}_lower', 'arm')
        if lower is not None:
            frame = overlay_image(frame, lower, wrist, rotation=lower_rot)
        
        return frame
    
    def _render_back_leg(self, frame, positions):
        """Render back leg (right in mirror)"""
        return self._render_leg(frame, positions, 'right')
    
    def _render_front_leg(self, frame, positions):
        """Render front leg (left in mirror)"""
        return self._render_leg(frame, positions, 'left')
    
    def _render_leg(self, frame, positions, side):
        """Render leg segments - each segment connects joints properly"""
        hip = positions.get(f'{side}_hip')
        knee = positions.get(f'{side}_knee')
        ankle = positions.get(f'{side}_ankle')
        
        if not (hip and knee and ankle):
            return frame
        
        # Calculate rotations based on actual joint connections
        # Upper leg: hip pointing to knee
        upper_rot = get_rotation_angle(hip, knee)
        # Lower leg: knee pointing to ankle
        lower_rot = get_rotation_angle(knee, ankle)
        
        # Smooth rotations
        upper_rot = self.rotation_smoother.smooth_rotation(f'{side}_upper_leg', upper_rot)
        lower_rot = self.rotation_smoother.smooth_rotation(f'{side}_lower_leg', lower_rot)
        
        # Render upper leg - positioned AT hip, pointing towards knee
        upper = self._get_scaled_sprite(f'leg_{side}_upper', 'leg')
        if upper is not None:
            frame = overlay_image(frame, upper, hip, rotation=upper_rot)
        
        # Render middle leg - positioned AT knee
        middle = self._get_scaled_sprite(f'leg_{side}_middle', 'leg')
        if middle is not None:
            frame = overlay_image(frame, middle, knee, rotation=lower_rot)
        
        # Render lower leg/foot - positioned AT ankle
        lower = self._get_scaled_sprite(f'leg_{side}_lower', 'leg')
        if lower is not None:
            frame = overlay_image(frame, lower, ankle, rotation=lower_rot)
        
        return frame
    
    def _midpoint(self, p1, p2):
        """Calculate midpoint between two positions"""
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)