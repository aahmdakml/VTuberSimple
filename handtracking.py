import cv2
import time
import mediapipe as mp

# --- MediaPipe setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Camera index: 0 = default webcam. Change if needed.
CAM_INDEX = 0
CAP_WIDTH, CAP_HEIGHT = 1280, 720  # try 640x480 if your cam is slow

def draw_info(frame, fps, results):
    h, w = frame.shape[:2]
    # FPS overlay
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # Handedness overlay (Left/Right)
    if results.multi_handedness:
        for idx, hand in enumerate(results.multi_handedness):
            label = hand.classification[0].label  # 'Left' or 'Right'
            score = hand.classification[0].score
            cv2.putText(frame, f"{label} ({score:.2f})",
                        (10, 60 + idx*30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 0), 2)

def count_raised_fingers(hand_landmarks, handed_label):
    """
    Simple 5-finger counter using tip-vs-pip y for fingers, and x for thumb.
    Landmarks index reference: https://google.github.io/mediapipe/solutions/hands.html
    """
    lm = hand_landmarks.landmark
    # Indices for tips and PIP joints
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]

    fingers = 0

    # Thumb: compare x depending on left/right hand (image is mirrored later)
    if handed_label == "Right":
        fingers += int(lm[tips[0]].x < lm[pips[0]].x)
    else:  # Left
        fingers += int(lm[tips[0]].x > lm[pips[0]].x)

    # Other fingers: tip higher (smaller y) than PIP
    for ti, pi in zip(tips[1:], pips[1:]):
        fingers += int(lm[ti].y < lm[pi].y)

    return fingers

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

    prev = time.time()

    # Hands config:
    # max_num_hands=2, detection_confidence=0.6, tracking_confidence=0.6
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to grab frame. Try another CAM_INDEX.")
                break

            # Flip for more natural selfie view
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb)

            # Draw landmarks & count fingers
            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

                    # Handedness for this hand (align with multi_handedness list)
                    handed_label = "Unknown"
                    if results.multi_handedness and i < len(results.multi_handedness):
                        handed_label = results.multi_handedness[i].classification[0].label

                    # Finger count (quick demo)
                    fingers = count_raised_fingers(hand_landmarks, handed_label)
                    cv2.putText(frame, f"{handed_label}: {fingers}",
                                (10, 120 + i*30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 255), 2)

                    # Example: print landmarks for the first detected hand
                    if i == 0:
                        # 21 landmarks with normalized coords (x,y in [0,1], z in image depth)
                        lms = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                        # Uncomment to see them in console
                        # print("Hand0 landmarks:", lms[:3], "...")  # first 3 as preview

            # FPS
            now = time.time()
            fps = 1.0 / (now - prev) if now != prev else 0.0
            prev = now
            draw_info(frame, fps, results)

            cv2.imshow("Hand Tracking - OpenCV + MediaPipe", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or q to quit
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
