import cv2
import numpy as np
import mediapipe as mp
import time
import os

# ---- Config ----
CAM_INDEX = 0
CAP_W, CAP_H = 1920, 1280
MODEL_SELECTION = 1  # 0=general (close-up), 1=landscape (full body farther away)
MASK_THRESHOLD = 0.5  # higher = stricter person mask

# Background modes: "live_blur", "solid", "image", "mask_view"
BG_MODE = "live_blur"
BG_SOLID_COLOR = (0, 255, 255)  # BGR (used when BG_MODE="solid")
BG_IMAGE_PATH = "BG.jpg"  # used when BG_MODE="image"

mp_selfie = mp.solutions.selfie_segmentation

def ensure_bg_image(path, target_shape):
    if not os.path.exists(path):
        return None
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)

def main():

    global BG_MODE

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)

    prev = time.time()

    with mp_selfie.SelfieSegmentation(model_selection=MODEL_SELECTION) as seg:
        bg_img_cache = None

        print("[Controls] q/ESC: quit | 1: blur | 2: solid color | 3: image | 4: mask view")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed. Try another CAM_INDEX.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Run segmentation
            res = seg.process(rgb)
            # res.segmentation_mask is float32 [0..1], higher = foreground
            if res.segmentation_mask is None:
                cv2.imshow("Body Segmentation", frame)
            else:
                mask = res.segmentation_mask
                # Smooth & threshold the mask
                mask = cv2.GaussianBlur(mask, (5,5), 0)
                fg = (mask > MASK_THRESHOLD).astype(np.uint8)

                if BG_MODE == "mask_view":
                    # Visualize mask grayscale
                    mask_vis = (mask * 255).astype(np.uint8)
                    cv2.imshow("Body Segmentation", mask_vis)
                else:
                    h, w = frame.shape[:2]

                    if BG_MODE == "live_blur":
                        # Blur the background, keep person sharp
                        blurred = cv2.GaussianBlur(frame, (35, 35), 0)
                        out = frame * fg[..., None] + blurred * (1 - fg[..., None])

                    elif BG_MODE == "solid":
                        solid = np.full_like(frame, BG_SOLID_COLOR, dtype=np.uint8)
                        out = frame * fg[..., None] + solid * (1 - fg[..., None])

                    elif BG_MODE == "image":
                        if bg_img_cache is None or bg_img_cache.shape[:2] != (h, w):
                            bg_img_cache = ensure_bg_image(BG_IMAGE_PATH, frame.shape)
                            if bg_img_cache is None:
                                print(f"[Warn] Could not load BG image: {BG_IMAGE_PATH}, falling back to blur.")
                                BG_MODE = "live_blur"
                                blurred = cv2.GaussianBlur(frame, (35, 35), 0)
                                out = frame * fg[..., None] + blurred * (1 - fg[..., None])
                            else:
                                out = frame * fg[..., None] + bg_img_cache * (1 - fg[..., None])
                        else:
                            out = frame * fg[..., None] + bg_img_cache * (1 - fg[..., None])

                    else:
                        out = frame  # fallback

                    out = out.astype(np.uint8)
                    # FPS overlay
                    now = time.time()
                    fps = 1.0 / (now - prev) if now != prev else 0.0
                    prev = now
                    cv2.putText(out, f"FPS: {fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                    cv2.putText(out, f"Mode: {BG_MODE}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                    cv2.imshow("Body Segmentation", out)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('1'):
                BG_MODE = "live_blur"
            elif key == ord('2'):
                BG_MODE = "solid"
            elif key == ord('3'):
                BG_MODE = "image"
            elif key == ord('4'):
                BG_MODE = "mask_view"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
