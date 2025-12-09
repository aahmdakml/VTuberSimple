"""
VTuber Full Body Tracking - Main Application
Real-time character tracking using MediaPipe and OpenCV
"""

import cv2
import time
import numpy as np
from config import *
from tracker import BodyTracker, FaceTracker
from character import Character
from renderer import CharacterRenderer
from utils import draw_fps, create_blank_frame


class VTuberApp:
    """Main VTuber application"""
    
    def __init__(self):
        print("=" * 60)
        print("VTuber Full Body Tracking System")
        print("=" * 60)
        
        # Initialize components
        print("\n[1/5] Initializing character...")
        self.character = Character()
        
        print("\n[2/5] Initializing body tracker...")
        self.body_tracker = BodyTracker()
        
        print("\n[3/5] Initializing face tracker...")
        self.face_tracker = FaceTracker()
        
        print("\n[4/5] Initializing renderer...")
        self.renderer = CharacterRenderer(self.character)
        
        print("\n[5/5] Initializing camera...")
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera!")
        
        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        # Display toggles
        self.show_skeleton = SHOW_SKELETON
        self.show_fps = SHOW_FPS
        
        print("\n" + "=" * 60)
        print("Initialization complete!")
        print("=" * 60)
        print("\nControls:")
        print("  ESC or Q - Quit")
        print("  S - Toggle skeleton overlay")
        print("  F - Toggle FPS display")
        print("  R - Reset floating system")
        print("=" * 60)
    
    def run(self):
        """Main application loop"""
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # Mirror frame
                if MIRROR_MODE:
                    frame = cv2.flip(frame, 1)
                
                # Process tracking
                pose_landmarks = self.body_tracker.process(frame)
                face_landmarks = self.face_tracker.process(frame)
                
                # Update face tracker values
                if face_landmarks:
                    self.face_tracker.get_eye_ratios(face_landmarks, 
                                                     frame.shape[1], 
                                                     frame.shape[0])
                    self.face_tracker.get_mouth_expression(face_landmarks, 
                                                           frame.shape[1], 
                                                           frame.shape[0])
                
                # Create output frame (black background)
                output_frame = create_blank_frame(WINDOW_WIDTH, WINDOW_HEIGHT)
                
                # Render character
                if pose_landmarks:
                    output_frame = self.renderer.render(
                        output_frame, 
                        pose_landmarks,
                        self.face_tracker,
                        self.body_tracker
                    )
                
                # Show skeleton overlay if enabled
                if self.show_skeleton and pose_landmarks:
                    output_frame = self._draw_skeleton(output_frame, pose_landmarks)
                
                # Calculate and display FPS
                self._update_fps()
                if self.show_fps:
                    output_frame = draw_fps(output_frame, self.fps)
                
                # Display
                cv2.imshow(WINDOW_NAME, output_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or Q
                    print("\nShutting down...")
                    break
                elif key == ord('s'):  # Toggle skeleton
                    self.show_skeleton = not self.show_skeleton
                    print(f"Skeleton overlay: {'ON' if self.show_skeleton else 'OFF'}")
                elif key == ord('f'):  # Toggle FPS
                    self.show_fps = not self.show_fps
                    print(f"FPS display: {'ON' if self.show_fps else 'OFF'}")
                elif key == ord('r'):  # Reset floating
                    self.renderer.floating_system.reset()
                    print("Floating system reset!")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def _update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        
        if self.frame_count >= 30:
            current_time = time.time()
            elapsed = current_time - self.fps_start_time
            self.fps = self.frame_count / elapsed
            
            self.frame_count = 0
            self.fps_start_time = current_time
    
    def _draw_skeleton(self, frame, landmarks):
        """Draw MediaPipe skeleton overlay for debugging"""
        h, w = frame.shape[:2]
        
        # Draw landmarks
        for landmark in landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        # Draw connections
        connections = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Arms
            (11, 23), (12, 24), (23, 24),  # Torso
            (23, 25), (25, 27), (24, 26), (26, 28),  # Legs
            (0, 1), (0, 4), (1, 2), (2, 3), (3, 7), (4, 5), (5, 6), (6, 8)  # Face
        ]
        
        for connection in connections:
            start_idx, end_idx = connection
            
            if start_idx < len(landmarks.landmark) and end_idx < len(landmarks.landmark):
                start = landmarks.landmark[start_idx]
                end = landmarks.landmark[end_idx]
                
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        return frame
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        
        if self.cap:
            self.cap.release()
        
        self.body_tracker.close()
        self.face_tracker.close()
        
        cv2.destroyAllWindows()
        
        print("Cleanup complete!")


def main():
    """Entry point"""
    try:
        app = VTuberApp()
        app.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()