import cv2
import numpy as np
import mediapipe as mp
import time

# ---------- Config ----------
CAM_INDEX = 0
CAP_W, CAP_H = 1280, 720

MIN_DET_CONF = 0.6
MIN_TRK_CONF = 0.6
MAX_FACES = 1
REFINE_LANDMARKS = True   # enables iris + around lips/eyes; more detail

MASK_THRESHOLD = 0.5       # used only for binary ops; here we directly fill poly
BG_SOLID_COLOR = (0, 255, 255)  # BGR for solid mode

# Modes
MODE_BLUR  = "blur"
MODE_SOLID = "solid"
MODE_MASK  = "mask_view"

# Drawing toggles (you can change defaults)
DRAW_TESSEL = True
DRAW_CONTOUR = True
DRAW_IRIS = True

# ---------- Mediapipe setup ----------
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles
mp_face    = mp.solutions.face_mesh

# Convenience aliases for connection sets
FACEMESH_TESSEL = mp_face.FACEMESH_TESSELATION
FACEMESH_CONTOURS = mp_face.FACEMESH_CONTOURS
FACEMESH_IRISES = mp_face.FACEMESH_IRISES
FACEMESH_FACE_OVAL = mp_face.FACEMESH_FACE_OVAL   # we'll build mask from this

def draw_hud(img, lines, start_y=24, scale=0.6, color=(50,255,50)):
    y = start_y
    for t in lines:
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        y += int(26*scale + 6)

def face_oval_mask(frame_shape, landmarks, img_w, img_h):
    """
    Build a binary mask (uint8 0/255) for the face using FACEMESH_FACE_OVAL vertices.
    We take all vertices that appear in FACE_OVAL connections, then compute a convex hull
    to ensure a valid polygon order (simple & robust).
    """
    pts = []
    for a, b in FACEMESH_FACE_OVAL:
        pa = landmarks[a]
        pb = landmarks[b]
        pts.append((int(pa.x * img_w), int(pa.y * img_h)))
        pts.append((int(pb.x * img_w), int(pb.y * img_h)))

    if not pts:
        return np.zeros(frame_shape[:2], dtype=np.uint8)

    pts_np = np.array(pts, dtype=np.int32)
    hull = cv2.convexHull(pts_np)
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255)
    return mask

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)

    prev = time.time()

    # UI state
    apply_mask = True
    seg_mode = MODE_SOLID  # start solid so the effect is obvious

    draw_tessel = DRAW_TESSEL
    draw_contour = DRAW_CONTOUR
    draw_iris = DRAW_IRIS

    print("[Controls] q/ESC quit | M toggle apply | 1 blur | 2 solid | 3 mask view | T tessellation | O contours | I iris")

    with mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=MAX_FACES,
        refine_landmarks=REFINE_LANDMARKS,   # enables iris/detail
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRK_CONF,
    ) as face_mesh:

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed. Try another CAM_INDEX.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            out = frame.copy()
            h, w = out.shape[:2]
            face_mask = None

            if res.multi_face_landmarks:
                for face_landmarks in res.multi_face_landmarks:
                    # Drawing
                    if draw_tessel:
                        mp_drawing.draw_landmarks(
                            image=out,
                            landmark_list=face_landmarks,
                            connections=FACEMESH_TESSEL,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                        )
                    if draw_contour:
                        mp_drawing.draw_landmarks(
                            image=out,
                            landmark_list=face_landmarks,
                            connections=FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
                        )
                    if draw_iris:
                        mp_drawing.draw_landmarks(
                            image=out,
                            landmark_list=face_landmarks,
                            connections=FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_styles.get_default_face_mesh_iris_connections_style()
                        )

                    # Build a binary mask from the face oval (0 background, 255 face)
                    face_mask = face_oval_mask(out.shape, face_landmarks.landmark, w, h)
                    # Only one face (MAX_FACES=1) by default; if multiple, you can OR masks.

                    break  # process only the first face for now

            # Apply composite AFTER drawing, so the mesh stays visible
            if apply_mask and face_mask is not None:
                if seg_mode == MODE_MASK:
                    vis = cv2.cvtColor(face_mask, cv2.COLOR_GRAY2BGR)
                    out = cv2.addWeighted(out, 0.35, vis, 0.65, 0)  # overlay mask for visibility
                elif seg_mode == MODE_BLUR:
                    bg = cv2.GaussianBlur(out, (35, 35), 0)
                    # keep face region sharp
                    mask3 = face_mask[..., None] / 255.0
                    out = (out * mask3 + bg * (1 - mask3)).astype(np.uint8)
                elif seg_mode == MODE_SOLID:
                    solid = np.full_like(out, BG_SOLID_COLOR, dtype=np.uint8)
                    mask3 = face_mask[..., None] / 255.0
                    out = (out * mask3 + solid * (1 - mask3)).astype(np.uint8)
            else:
                # If no face or mask not applied, just show out
                pass

            # HUD
            now = time.time()
            fps = 1.0 / (now - prev) if now != prev else 0.0
            prev = now

            hud = [
                f"FPS: {fps:.1f}",
                f"Mask apply: {'ON' if apply_mask else 'OFF'} | Mode: {seg_mode} (M toggle, 1/2/3 select)",
                f"Draw: Tessel={'ON' if draw_tessel else 'OFF'} (T)  Contours={'ON' if draw_contour else 'OFF'} (O)  Iris={'ON' if draw_iris else 'OFF'} (I)",
                "Quit: q / ESC"
            ]
            draw_hud(out, hud, 24)

            cv2.imshow("Face Tracking + Mesh + Face Mask", out)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q'), ord('Q')):
                break
            elif k in (ord('m'), ord('M')):
                apply_mask = not apply_mask
            elif k == ord('1'):
                seg_mode = MODE_BLUR
            elif k == ord('2'):
                seg_mode = MODE_SOLID
            elif k == ord('3'):
                seg_mode = MODE_MASK
            elif k in (ord('t'), ord('T')):
                draw_tessel = not draw_tessel
            elif k in (ord('o'), ord('O')):
                draw_contour = not draw_contour
            elif k in (ord('i'), ord('I')):
                draw_iris = not draw_iris

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

