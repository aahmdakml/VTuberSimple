import cv2
import numpy as np
import time

# =========================
# Config
# =========================
CAM_INDEX = 0
CAP_W, CAP_H = 1280, 720

# Filters (Tugas 1)
MODE_NORMAL      = 0
MODE_AVG_5       = 1
MODE_AVG_9       = 2
MODE_GAUSS_2D    = 3  # Using custom kernel + filter2D (required)
MODE_SHARPEN     = 4

# HSV Detection (Tugas 2)
HSV_OFF          = 0
HSV_ON           = 1

# Morphology kernel
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# =========================
# Gaussian kernel builder (2D) for filter2D
# =========================
def gaussian_kernel_2d(ksize=9, sigma=2.0):
    """Build a normalized 2D Gaussian kernel (outer product)"""
    g1d = cv2.getGaussianKernel(ksize, sigma)              # column vector
    g2d = g1d @ g1d.T                                      # outer product -> 2D
    g2d = g2d / g2d.sum()                                  # normalize (important)
    return g2d

GAUSS_KSIZE = 9
GAUSS_SIGMA = 2.0
GAUSS_KERNEL_2D = gaussian_kernel_2d(GAUSS_KSIZE, GAUSS_SIGMA)

# Sharpen kernel from spec
K_SHARPEN = np.array([[0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]], dtype=np.float32)

# =========================
# HSV Color Ranges (OpenCV Hue: 0..179)
# You can adjust if lighting differs.
# =========================
# Blue
BLUE_LO = np.array([100, 100, 50], dtype=np.uint8)
BLUE_HI = np.array([130, 255, 255], dtype=np.uint8)
# Green
GREEN_LO = np.array([40,  80,  50], dtype=np.uint8)
GREEN_HI = np.array([85, 255, 255], dtype=np.uint8)

def detect_color_and_action(frame_bgr, which="blue", area_thresh=3000):
    """
    Tugas 2:
      - Convert BGR->HSV
      - inRange with selected color
      - Morphological opening then closing
      - findContours and trigger an action if big object is present
    Returns: (vis_frame, action_triggered_bool)
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    if which == "green":
        lo, hi = GREEN_LO, GREEN_HI
        label = "GREEN"
    else:
        lo, hi = BLUE_LO, BLUE_HI
        label = "BLUE"

    mask = cv2.inRange(hsv, lo, hi)

    # Opening: remove small noisy blobs; Closing: fill small holes
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=1)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=1)

    # Find contours on cleaned mask
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    action = False
    vis = frame_bgr.copy()
    if contours:
        # Take the largest contour
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > area_thresh:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,255), 2)
            cv2.putText(vis, f"Objek {label} Terdeteksi (area={int(area)})",
                        (x, max(30, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            action = True

    # Side-by-side small mask preview
    mask_vis = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR)
    mask_vis = cv2.resize(mask_vis, (vis.shape[1]//3, vis.shape[0]//3))
    vis[0:mask_vis.shape[0], 0:mask_vis.shape[1]] = mask_vis

    return vis, action

def draw_hud(img, fps, mode, hsv_state, hsv_color):
    lines = [
        "Controls: 0=Normal | 1=Avg5 | 2=Avg9 | 3=Gaussian(filter2D) | 4=Sharpen | H=HSV toggle | B/G=Color | Q/Esc=Quit",
        f"Mode: {mode}   HSV: {'ON' if hsv_state==HSV_ON else 'OFF'} ({hsv_color})   FPS: {fps:.1f}"
    ]
    y = 25
    for ln in lines:
        cv2.putText(img, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 255, 10), 2, cv2.LINE_AA)
        y += 24

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)

    mode = MODE_NORMAL
    hsv_state = HSV_OFF
    hsv_color = "blue"   # default color
    prev = time.time()

    print("[Controls] 0=Normal | 1=Avg5 | 2=Avg9 | 3=Gaussian(filter2D) | 4=Sharpen | H=HSV toggle | B/G=Color | Q/Esc=Quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to grab frame. Try another camera index.")
            break

        frame = cv2.flip(frame, 1)  # mirror for webcam use

        # ---------- Tugas 1: apply selected filter ----------
        if mode == MODE_AVG_5:
            out = cv2.blur(frame, (5,5))
            mode_name = "Average Blur 5x5"
        elif mode == MODE_AVG_9:
            out = cv2.blur(frame, (9,9))
            mode_name = "Average Blur 9x9"
        elif mode == MODE_GAUSS_2D:
            # REQUIRED: custom Gaussian with filter2D
            out = cv2.filter2D(frame, ddepth=-1, kernel=GAUSS_KERNEL_2D)
            mode_name = f"Gaussian {GAUSS_KSIZE}x{GAUSS_KSIZE} (filter2D, sigma={GAUSS_SIGMA})"
        elif mode == MODE_SHARPEN:
            out = cv2.filter2D(frame, ddepth=-1, kernel=K_SHARPEN)
            mode_name = "Sharpen"
        else:
            out = frame.copy()
            mode_name = "Normal"

        # ---------- Tugas 2: HSV detection & action ----------
        action = False
        if hsv_state == HSV_ON:
            out, action = detect_color_and_action(out, which=hsv_color)

        # HUD + FPS
        now = time.time()
        fps = 1.0 / (now - prev) if now != prev else 0.0
        prev = now
        draw_hud(out, fps, mode_name, hsv_state, hsv_color.upper())

        # Optional: show a simple “action” banner if triggered
        if action:
            cv2.putText(out, "ACTION: EVENT TRIGGERED!", (10, out.shape[0]-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 3, cv2.LINE_AA)

        cv2.imshow("Final Projek - PCV", out)

        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q'), ord('Q')):
            break
        elif k == ord('0'):
            mode = MODE_NORMAL
        elif k == ord('1'):
            mode = MODE_AVG_5
        elif k == ord('2'):
            mode = MODE_AVG_9
        elif k == ord('3'):
            mode = MODE_GAUSS_2D
        elif k == ord('4'):
            mode = MODE_SHARPEN
        elif k in (ord('h'), ord('H')):
            hsv_state = HSV_OFF if hsv_state == HSV_ON else HSV_ON
        elif k in (ord('b'), ord('B')):
            hsv_color = "blue"
        elif k in (ord('g'), ord('G')):
            hsv_color = "green"

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
