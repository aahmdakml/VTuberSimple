"""
Rendering Engine - Direct Skeleton Following
Each sprite attaches directly to skeleton joints and extends along skeleton lines
"""

import cv2
import numpy as np
import mediapipe as mp
import math
from config import *
from utils import overlay_image, get_rotation_angle, calculate_distance
from floating_system import FloatingSystem, PositionSmoother, RotationSmoother, TailAnimator
from expression import ExpressionController


class CharacterRenderer:
    """Renders character sprites directly onto skeleton structure"""
    
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
        
        # Sprite-specific scale adjustments
        self.sprite_scales = {
            'head': 0.4,
            'body_tail': 0.5,
            'arm': 0.6,
            'leg': 0.6,
            'eye': 1.0,
            'mouth': 1.0,
        }
        
        # Face feature offsets
        self.face_offsets = {
            'eye_left_x': -60,
            'eye_left_y': -30,
            'eye_right_x': 60,
            'eye_right_y': -30,
            'mouth_x': 0,
            'mouth_y': 55,
        }
        
        # Pivot offsets for extending sprites from joints
        # Negative = sprite extends downward/forward from pivot point
        self.pivot_offsets = {
            'upper': -0.2,    # Upper segments extend from shoulder/hip
            'middle': -0.2,   # Middle segments extend from elbow/knee
            'lower': -0.15,    # Hand/foot at wrist/ankle
        }
    
    def render(self, frame, pose_landmarks, face_tracker, body_tracker):
        """Render character onto frame"""
        if pose_landmarks is None:
            return frame
        
        h, w = frame.shape[:2]
        
        # Calculate scale
        distance_scale = body_tracker.calculate_scale_factor(pose_landmarks, w, h)
        self.character.set_scale(distance_scale * BASE_SCALE)
        
        # Extract positions with floating
        positions = self._extract_positions(pose_landmarks, body_tracker, w, h)
        
        # Update expression (default: eyes open, mouth neutral)
        if face_tracker:
            left_eye = face_tracker.smoothed_left_eye_ratio
            right_eye = face_tracker.smoothed_right_eye_ratio
            mouth_open = face_tracker.smoothed_mouth_open
            mouth_smile = face_tracker.smoothed_mouth_smile
            self.expression_controller.update(left_eye, right_eye, mouth_open, mouth_smile)
        
        # Calculate body velocity
        body_velocity = self._calculate_body_velocity(positions.get('torso'))
        
        # Render in z-order (back to front)
        frame = self._render_back_leg(frame, positions)
        frame = self._render_back_arm(frame, positions)
        frame = self._render_body(frame, positions, body_velocity)
        frame = self._render_front_leg(frame, positions)
        frame = self._render_front_arm(frame, positions)
        frame = self._render_head(frame, positions)
        
        return frame
    
    def _extract_positions(self, landmarks, body_tracker, width, height):
        """Extract and smooth positions"""
        positions = {}
        
        # Get raw positions from MediaPipe
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
        
        # Torso center
        if raw.get('shoulder_center') and raw.get('hip_center'):
            tc_x = (raw['shoulder_center'][0] + raw['hip_center'][0]) / 2
            tc_y = (raw['shoulder_center'][1] + raw['hip_center'][1]) / 2
            raw['torso'] = (tc_x, tc_y)
        
        # Head position
        if raw.get('shoulder_center'):
            head_offset = 80 * self.character.current_scale
            raw['head'] = (raw['shoulder_center'][0], raw['shoulder_center'][1] - head_offset)
        
        # Apply smoothing and floating
        for key, pos in raw.items():
            if pos is not None:
                smoothed = self.position_smoother.smooth_position(key, pos)
                floated = self.floating_system.apply_offset(smoothed, key)
                positions[key] = floated
        
        return positions
    
    def _calculate_body_velocity(self, current_pos):
        """Calculate body velocity"""
        if current_pos is None or self.prev_body_pos is None:
            velocity = 0
        else:
            dist = calculate_distance(current_pos, self.prev_body_pos)
            velocity = min(1.0, dist / 50.0)
        
        self.prev_body_pos = current_pos
        return velocity
    
    def _get_scaled_sprite(self, sprite_name, sprite_type='default'):
        """Get scaled sprite"""
        sprite = self.character.get_sprite(sprite_name)
        if sprite is None:
            return None
        
        type_scale = self.sprite_scales.get(sprite_type, 1.0)
        adjusted_scale = self.character.current_scale * type_scale
        
        if adjusted_scale != 1.0:
            from utils import scale_image
            return scale_image(sprite, adjusted_scale)
        
        return sprite
    
    def _render_segment_on_skeleton(self, frame, sprite_name, sprite_type, 
                                     start_joint, end_joint, segment_type):
        """
        Render sprite segment extending from start_joint towards end_joint
        Sprite pivot at start_joint, extends along skeleton line
        """
        sprite = self._get_scaled_sprite(sprite_name, sprite_type)
        if sprite is None or start_joint is None or end_joint is None:
            return frame
        
        # Calculate rotation angle from start to end
        rotation = get_rotation_angle(start_joint, end_joint)
        
        # Smooth rotation
        smooth_key = f'{sprite_name}_rotation'
        rotation = self.rotation_smoother.smooth_rotation(smooth_key, rotation)
        
        h, w = sprite.shape[:2]
        
        # Get pivot offset for this segment type
        pivot_offset_pct = self.pivot_offsets.get(segment_type, 0)
        offset_y = h * pivot_offset_pct
        offset_x = 0
        
        # Rotate offset vector
        angle_rad = math.radians(rotation)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        rotated_offset_x = offset_x * cos_a - offset_y * sin_a
        rotated_offset_y = offset_x * sin_a + offset_y * cos_a
        
        # Apply offset to start joint position
        render_x = start_joint[0] + rotated_offset_x
        render_y = start_joint[1] + rotated_offset_y
        render_pos = (render_x, render_y)
        
        # Render sprite
        frame = overlay_image(frame, sprite, render_pos, rotation=rotation)
        
        return frame
    
    def _render_head(self, frame, positions):
        """Render head with expressions"""
        if 'head' not in positions:
            return frame
        
        head_pos = positions['head']
        base_scale = self.character.current_scale
        
        # Render head base
        head = self._get_scaled_sprite('head', 'head')
        if head is not None:
            frame = overlay_image(frame, head, head_pos)
        
        # Get expression sprites (default: open eyes, neutral mouth)
        left_eye_name, right_eye_name = self.expression_controller.get_eye_sprites()
        mouth_name = self.expression_controller.get_mouth_sprite()
        
        # Render left eye
        left_eye = self._get_scaled_sprite(left_eye_name, 'eye')
        if left_eye is not None:
            eye_x = head_pos[0] + self.face_offsets['eye_left_x'] * base_scale
            eye_y = head_pos[1] + self.face_offsets['eye_left_y'] * base_scale
            frame = overlay_image(frame, left_eye, (eye_x, eye_y))
        
        # Render right eye
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
        """Render body with tail"""
        if 'torso' not in positions:
            return frame
        
        torso_pos = positions['torso']
        tail_rotation = self.tail_animator.get_tail_rotation(body_velocity)
        
        body_tail = self._get_scaled_sprite('body_tail', 'body_tail')
        if body_tail is not None:
            frame = overlay_image(frame, body_tail, torso_pos, rotation=tail_rotation)
        
        return frame
    
    def _render_back_arm(self, frame, positions):
        return self._render_arm(frame, positions, 'right')
    
    def _render_front_arm(self, frame, positions):
        return self._render_arm(frame, positions, 'left')
    
    def _render_arm(self, frame, positions, side):
        """
        Render arm following skeleton structure:
        - Upper: extends from shoulder (12/11) to elbow (14/13)
        - Middle: extends from elbow (14/13) to wrist (16/15)
        - Lower (hand): at wrist (16/15), rotation follows forearm
        """
        shoulder = positions.get(f'{side}_shoulder')
        elbow = positions.get(f'{side}_elbow')
        wrist = positions.get(f'{side}_wrist')
        
        if not (shoulder and elbow and wrist):
            return frame
        
        # Render upper arm: shoulder → elbow
        frame = self._render_segment_on_skeleton(
            frame, f'arm_{side}_upper', 'arm',
            shoulder, elbow, 'upper'
        )
        
        # Render middle arm (forearm): elbow → wrist
        frame = self._render_segment_on_skeleton(
            frame, f'arm_{side}_middle', 'arm',
            elbow, wrist, 'middle'
        )
        
        # Render lower arm (hand): at wrist, rotation follows forearm
        forearm_rotation = get_rotation_angle(elbow, wrist)
        forearm_rotation = self.rotation_smoother.smooth_rotation(
            f'{side}_forearm_rot', forearm_rotation
        )
        
        hand_sprite = self._get_scaled_sprite(f'arm_{side}_lower', 'arm')
        if hand_sprite is not None:
            h, w = hand_sprite.shape[:2]
            
            # Hand pivot offset (slightly adjusted for natural wrist position)
            pivot_offset = self.pivot_offsets['lower']
            offset_y = h * pivot_offset
            
            angle_rad = math.radians(forearm_rotation)
            rotated_offset_x = -offset_y * math.sin(angle_rad)
            rotated_offset_y = offset_y * math.cos(angle_rad)
            
            hand_x = wrist[0] + rotated_offset_x
            hand_y = wrist[1] + rotated_offset_y
            
            frame = overlay_image(frame, hand_sprite, (hand_x, hand_y), rotation=forearm_rotation)
        
        return frame
    
    def _render_back_leg(self, frame, positions):
        return self._render_leg(frame, positions, 'right')
    
    def _render_front_leg(self, frame, positions):
        return self._render_leg(frame, positions, 'left')
    
    def _render_leg(self, frame, positions, side):
        """
        Render leg following skeleton structure:
        - Upper: extends from hip (24/23) to knee (26/25)
        - Middle: extends from knee (26/25) to ankle (28/27)
        - Lower (foot): at ankle (28/27), rotation follows shin
        """
        hip = positions.get(f'{side}_hip')
        knee = positions.get(f'{side}_knee')
        ankle = positions.get(f'{side}_ankle')
        
        if not (hip and knee and ankle):
            return frame
        
        # Render upper leg: hip → knee
        frame = self._render_segment_on_skeleton(
            frame, f'leg_{side}_upper', 'leg',
            hip, knee, 'upper'
        )
        
        # Render middle leg (shin): knee → ankle
        frame = self._render_segment_on_skeleton(
            frame, f'leg_{side}_middle', 'leg',
            knee, ankle, 'middle'
        )
        
        # Render lower leg (foot): at ankle, rotation follows shin
        shin_rotation = get_rotation_angle(knee, ankle)
        shin_rotation = self.rotation_smoother.smooth_rotation(
            f'{side}_shin_rot', shin_rotation
        )
        
        foot_sprite = self._get_scaled_sprite(f'leg_{side}_lower', 'leg')
        if foot_sprite is not None:
            h, w = foot_sprite.shape[:2]
            
            # Foot pivot offset
            pivot_offset = self.pivot_offsets['lower']
            offset_y = h * pivot_offset
            
            angle_rad = math.radians(shin_rotation)
            rotated_offset_x = -offset_y * math.sin(angle_rad)
            rotated_offset_y = offset_y * math.cos(angle_rad)
            
            foot_x = ankle[0] + rotated_offset_x
            foot_y = ankle[1] + rotated_offset_y
            
            frame = overlay_image(frame, foot_sprite, (foot_x, foot_y), rotation=shin_rotation)
        
        return frame