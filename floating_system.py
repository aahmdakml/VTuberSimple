"""
Floating Joint System
Creates organic floating/disconnected joint effect like Agatha from Double King
"""

import numpy as np
import time
import math
from config import FLOATING_DISTANCE, FLOATING_SPEEDS


class FloatingSystem:
    """Manages floating behavior for character joints"""
    
    def __init__(self):
        self.start_time = time.time()
        self.offsets = {}
        
        # Initialize offset trackers for each body part
        self.body_parts = [
            'head', 'torso',
            'left_upper_arm', 'right_upper_arm',
            'left_lower_arm', 'right_lower_arm',
            'left_hand', 'right_hand',
            'left_upper_leg', 'right_upper_leg',
            'left_lower_leg', 'right_lower_leg',
            'tail'
        ]
        
        # Initialize with random phases for organic movement
        for part in self.body_parts:
            self.offsets[part] = {
                'x_phase': np.random.uniform(0, 2 * np.pi),
                'y_phase': np.random.uniform(0, 2 * np.pi),
                'current_x': 0,
                'current_y': 0
            }
    
    def get_floating_offset(self, body_part):
        """
        Get current floating offset for a body part
        Returns: (offset_x, offset_y)
        """
        if body_part not in self.offsets:
            return (0, 0)
        
        current_time = time.time() - self.start_time
        
        # Get speed multiplier for this body part
        speed = FLOATING_SPEEDS.get(body_part, 1.0)
        
        offset_data = self.offsets[body_part]
        
        # Calculate smooth floating using sine waves with different phases
        # This creates organic, lissajous-like patterns
        x_offset = FLOATING_DISTANCE * math.sin(
            current_time * speed + offset_data['x_phase']
        )
        
        y_offset = FLOATING_DISTANCE * math.cos(
            current_time * speed * 1.3 + offset_data['y_phase']
        )
        
        # Add some variation with secondary waves
        x_offset += FLOATING_DISTANCE * 0.3 * math.sin(
            current_time * speed * 2.1 + offset_data['x_phase'] * 0.7
        )
        
        y_offset += FLOATING_DISTANCE * 0.3 * math.cos(
            current_time * speed * 1.7 + offset_data['y_phase'] * 1.3
        )
        
        # Store current offsets
        offset_data['current_x'] = x_offset
        offset_data['current_y'] = y_offset
        
        return (x_offset, y_offset)
    
    def apply_offset(self, position, body_part):
        """Apply floating offset to a position"""
        if position is None:
            return None
        
        offset = self.get_floating_offset(body_part)
        
        new_x = position[0] + offset[0]
        new_y = position[1] + offset[1]
        
        return (new_x, new_y)
    
    def reset(self):
        """Reset floating system"""
        self.start_time = time.time()
        for part in self.body_parts:
            self.offsets[part]['x_phase'] = np.random.uniform(0, 2 * np.pi)
            self.offsets[part]['y_phase'] = np.random.uniform(0, 2 * np.pi)


class PositionSmoother:
    """Smooths position transitions for fluid movement"""
    
    def __init__(self, smoothing_factor=0.3):
        self.smoothing_factor = smoothing_factor
        self.previous_positions = {}
    
    def smooth_position(self, key, new_position):
        """
        Apply smoothing to a position
        key: unique identifier for this position
        new_position: (x, y) tuple
        """
        if new_position is None:
            return self.previous_positions.get(key, None)
        
        if key not in self.previous_positions:
            self.previous_positions[key] = new_position
            return new_position
        
        prev_pos = self.previous_positions[key]
        
        # Linear interpolation
        smoothed_x = prev_pos[0] + (new_position[0] - prev_pos[0]) * (1.0 - self.smoothing_factor)
        smoothed_y = prev_pos[1] + (new_position[1] - prev_pos[1]) * (1.0 - self.smoothing_factor)
        
        smoothed_pos = (smoothed_x, smoothed_y)
        self.previous_positions[key] = smoothed_pos
        
        return smoothed_pos
    
    def reset(self):
        """Reset all smoothed positions"""
        self.previous_positions.clear()


class RotationSmoother:
    """Smooths rotation transitions"""
    
    def __init__(self, smoothing_factor=0.3):
        self.smoothing_factor = smoothing_factor
        self.previous_rotations = {}
    
    def smooth_rotation(self, key, new_rotation):
        """
        Apply smoothing to rotation angle
        key: unique identifier
        new_rotation: angle in degrees
        """
        if key not in self.previous_rotations:
            self.previous_rotations[key] = new_rotation
            return new_rotation
        
        prev_rot = self.previous_rotations[key]
        
        # Handle angle wrapping
        diff = new_rotation - prev_rot
        
        # Normalize to [-180, 180]
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        
        smoothed_rot = prev_rot + diff * (1.0 - self.smoothing_factor)
        
        self.previous_rotations[key] = smoothed_rot
        
        return smoothed_rot
    
    def reset(self):
        """Reset all smoothed rotations"""
        self.previous_rotations.clear()


class TailAnimator:
    """Handles semi-dynamic tail animation"""
    
    def __init__(self):
        self.base_rotation = 0
        self.rotation_speed = 0.5
        self.max_rotation = 15  # Maximum rotation in degrees
        
    def get_tail_rotation(self, body_velocity=0):
        """
        Get tail rotation based on body movement
        body_velocity: movement speed (0-1)
        """
        current_time = time.time()
        
        # Base idle animation
        idle_rotation = self.max_rotation * 0.3 * math.sin(current_time * self.rotation_speed)
        
        # Add movement-based rotation
        movement_rotation = body_velocity * self.max_rotation * 0.5
        
        total_rotation = idle_rotation + movement_rotation
        
        return total_rotation