"""
Facial Expression Controller
Manages facial expressions based on tracking data
"""

from config import EYE_ASPECT_RATIO_THRESHOLD, MOUTH_OPEN_THRESHOLD, MOUTH_SMILE_THRESHOLD


class ExpressionController:
    """Controls character facial expressions"""
    
    def __init__(self):
        self.current_expression = {
            'left_eye': 'open',
            'right_eye': 'open',
            'mouth': 'neutral'
        }
        
    def update(self, left_eye_ratio, right_eye_ratio, mouth_open, mouth_smile):
        """
        Update expression state based on tracking data
        
        Args:
            left_eye_ratio: Left eye aspect ratio (lower = more closed)
            right_eye_ratio: Right eye aspect ratio
            mouth_open: Mouth open ratio
            mouth_smile: Mouth smile ratio
        """
        # Update eye states
        self.current_expression['left_eye'] = self._determine_eye_state(left_eye_ratio)
        self.current_expression['right_eye'] = self._determine_eye_state(right_eye_ratio)
        
        # Update mouth state
        self.current_expression['mouth'] = self._determine_mouth_state(mouth_open, mouth_smile)
        
        return self.current_expression
    
    def _determine_eye_state(self, eye_ratio):
        """Determine if eye is open or closed"""
        if eye_ratio < EYE_ASPECT_RATIO_THRESHOLD:
            return 'closed'
        else:
            return 'open'
    
    def _determine_mouth_state(self, mouth_open, mouth_smile):
        """
        Determine mouth expression state
        Returns: 'neutral', 'opened', or 'smile'
        """
        # Priority: smile > opened > neutral
        
        # Check for smile (wide mouth)
        if mouth_smile > MOUTH_SMILE_THRESHOLD and mouth_open < MOUTH_OPEN_THRESHOLD * 2:
            return 'smile'
        
        # Check for open mouth
        if mouth_open > MOUTH_OPEN_THRESHOLD:
            return 'opened'
        
        # Default neutral
        return 'neutral'
    
    def get_expression(self):
        """Get current expression state"""
        return self.current_expression
    
    def get_eye_sprites(self):
        """Get sprite names for current eye state"""
        left_eye = 'eye_left_open' if self.current_expression['left_eye'] == 'open' else 'eye_left_closed'
        right_eye = 'eye_right_open' if self.current_expression['right_eye'] == 'open' else 'eye_right_closed'
        
        return left_eye, right_eye
    
    def get_mouth_sprite(self):
        """Get sprite name for current mouth state"""
        mouth_state = self.current_expression['mouth']
        
        if mouth_state == 'opened':
            return 'mouth_opened'
        elif mouth_state == 'smile':
            return 'mouth_smile'
        else:
            return 'mouth_neutral'
    
    def reset(self):
        """Reset expression to default"""
        self.current_expression = {
            'left_eye': 'open',
            'right_eye': 'open',
            'mouth': 'neutral'
        }