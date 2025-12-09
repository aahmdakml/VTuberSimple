"""
Character Model and Sprite Manager
Loads and manages character sprites
"""

import cv2
import os
import numpy as np
from config import ASSETS_PATH, SPRITE_FILES, BASE_SCALE
from utils import scale_image


class Character:
    """Manages character sprites and properties"""
    
    def __init__(self):
        self.sprites = {}
        self.base_scale = BASE_SCALE
        self.current_scale = BASE_SCALE
        
        # Load all sprites
        self._load_sprites()
        
        # Character dimensions (will be calculated from sprites)
        self.head_size = None
        self.torso_size = None
        self.calculate_dimensions()
        
    def _load_sprites(self):
        """Load all sprite images"""
        print("Loading character sprites...")
        
        for sprite_name, filename in SPRITE_FILES.items():
            filepath = os.path.join(ASSETS_PATH, filename)
            
            if not os.path.exists(filepath):
                print(f"Warning: Sprite not found: {filepath}")
                self.sprites[sprite_name] = None
                continue
            
            # Load with alpha channel
            sprite = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            
            if sprite is None:
                print(f"Warning: Failed to load sprite: {filepath}")
                self.sprites[sprite_name] = None
                continue
            
            # Ensure BGRA format
            if sprite.shape[2] == 3:
                sprite = cv2.cvtColor(sprite, cv2.COLOR_BGR2BGRA)
            
            self.sprites[sprite_name] = sprite
            print(f"Loaded: {sprite_name} ({sprite.shape[1]}x{sprite.shape[0]})")
        
        print(f"Loaded {len([s for s in self.sprites.values() if s is not None])} sprites")
    
    def calculate_dimensions(self):
        """Calculate character dimensions from sprites"""
        if self.sprites['head'] is not None:
            h, w = self.sprites['head'].shape[:2]
            self.head_size = (w, h)
        
        if self.sprites['body_tail'] is not None:
            h, w = self.sprites['body_tail'].shape[:2]
            self.torso_size = (w, h)
    
    def get_sprite(self, sprite_name, scale=None):
        """
        Get a sprite with optional scaling
        
        Args:
            sprite_name: Name of the sprite
            scale: Scale factor (uses current_scale if None)
        
        Returns:
            Scaled sprite image or None
        """
        if sprite_name not in self.sprites:
            return None
        
        sprite = self.sprites[sprite_name]
        
        if sprite is None:
            return None
        
        if scale is None:
            scale = self.current_scale
        
        if scale != 1.0:
            return scale_image(sprite, scale)
        
        return sprite.copy()
    
    def set_scale(self, scale):
        """Set current character scale"""
        self.current_scale = scale
    
    def get_scaled_dimensions(self):
        """Get character dimensions with current scale applied"""
        scaled_head = None
        scaled_torso = None
        
        if self.head_size:
            scaled_head = (
                int(self.head_size[0] * self.current_scale),
                int(self.head_size[1] * self.current_scale)
            )
        
        if self.torso_size:
            scaled_torso = (
                int(self.torso_size[0] * self.current_scale),
                int(self.torso_size[1] * self.current_scale)
            )
        
        return scaled_head, scaled_torso


class SpriteCache:
    """Cache for scaled sprites to improve performance"""
    
    def __init__(self, max_cache_size=100):
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.access_count = {}
    
    def get(self, key):
        """Get cached sprite"""
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def put(self, key, sprite):
        """Cache a sprite"""
        # Clean cache if too large
        if len(self.cache) >= self.max_cache_size:
            self._clean_cache()
        
        self.cache[key] = sprite
        self.access_count[key] = 1
    
    def _clean_cache(self):
        """Remove least accessed items"""
        if not self.access_count:
            return
        
        # Remove 20% of least accessed items
        sorted_items = sorted(self.access_count.items(), key=lambda x: x[1])
        num_to_remove = max(1, len(sorted_items) // 5)
        
        for key, _ in sorted_items[:num_to_remove]:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_count:
                del self.access_count[key]
    
    def clear(self):
        """Clear all cached sprites"""
        self.cache.clear()
        self.access_count.clear()