import cv2, time, math, os, numpy as np
import mediapipe as mp

CAM_INDEX = 0
CAP_W, CAP_H = 1280, 720
MIN_DET_CONF = 0.6
MIN_TRK_CONF = 0.6
MODEL_COMPLEXITY = 1
SMOOTH_LANDMARKS = True

# Segmentation settings
MASK_THRESHOLD = 0.4  # a bit looser to make effect obvious
SEG_MODE_BLUR = "blur"
SEG_MODE_SOLID = "solid"
SEG_MODE_IMAGE = "image"
SEG_MODE_MASK = "mask_view"
BG_SOLID_COLOR = (0, 255, 255)
BG_IMAGE_PATH = "BG.png"

# Counters
ENABLE_CURL_COUNTER = True
ENABLE_SQUAT_COUNTER = True

mp_pose  = mp.solutions.pose
mp_draw  = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles
L = mp_pose.PoseLandmark

def calc_angle(a,b,c):
    ax,ay=a; bx,by=b; cx,cy=c
    ab=(ax-bx, ay-by); cb=(cx-bx, cy-by)
    dot=ab[0]*cb[0]+ab[1]*cb[1]
    nab=(ab[0]**2+ab[1]**2)**0.5+1e-6
    ncb=(cb[0]**2+cb[1]**2)**0.5+1e-6
    cosang=max(-1,min(1,dot/(nab*ncb)))
    import math
    return math.degrees(math.acos(cosang))

def get_xy(w,h,lm): return int(lm.x*w), int(lm.y*h)

def draw_info(img, lines, y0=24, scale=0.6, color=(50,255,50)):
    y=y0
    for t in lines:
        cv2.putText(img, t, (10,y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        y += int(26*scale+6)

def apply_segmentation(out_bgr, seg_mask, mode, cache):
    if seg_mask is None:
        # Show a hint so you can *see* if mask is missing
        cv2.putText(out_bgr, "No segmentation mask (check mediapipe version)", (10, out_bgr.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        return out_bgr
    m = cv2.GaussianBlur(seg_mask, (5,5), 0)
    fg = (m > MASK_THRESHOLD).astype(np.uint8)
    if mode == SEG_MODE_MASK:
        return cv2.cvtColor((m*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    h,w = out_bgr.shape[:2]
    if mode == SEG_MODE_BLUR:
        bg = cv2.GaussianBlur(out_bgr, (35,35), 0)
    elif mode == SEG_MODE_SOLID:
        bg = np.full_like(out_bgr, BG_SOLID_COLOR, dtype=np.uint8)
    elif mode == SEG_MODE_IMAGE:
        key=(w,h,BG_IMAGE_PATH)
        if key not in cache:
            if not os.path.exists(BG_IMAGE_PATH):
                cache[key] = cv2.GaussianBlur(out_bgr, (35,35), 0)
            else:
                raw = cv2.imread(BG_IMAGE_PATH)
                cache[key] = cv2.resize(raw, (w,h)) if raw is not None else cv2.GaussianBlur(out_bgr, (35,35), 0)
        bg = cache[key]
    else:
        return out_bgr
    comp = out_bgr * fg[...,None] + bg * (1-fg[...,None])
    return comp.astype(np.uint8)

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)

    prev=time.time()
    # UI state
    # IMPORTANT: keep model segmentation ENABLED; we only toggle whether we *apply* it.
    seg_apply = True
    seg_mode  = SEG_MODE_SOLID  # start with obvious solid
    bg_cache  = {}

    curl_count=0; curlL="down"; curlR="down"
    squat_count=0; squatPhase="up"
    enableCurl=ENABLE_CURL_COUNTER
    enableSquat=ENABLE_SQUAT_COUNTER

    print("[Controls] q/ESC quit | M toggle apply | 1 blur | 2 solid | 3 image | 4 mask view | C curl | S squat")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=MODEL_COMPLEXITY,
        enable_segmentation=True,          # <— ALWAYS TRUE
        smooth_landmarks=SMOOTH_LANDMARKS,
        min_detection_confidence=MIN_DET_CONF,
        min_tracking_confidence=MIN_TRK_CONF
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.flip(frame,1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)

            out = frame.copy()
            h,w = out.shape[:2]

            if res.pose_landmarks:
                mp_draw.draw_landmarks(out, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                       landmark_drawing_spec=mp_style.get_default_pose_landmarks_style())
                lm = res.pose_landmarks.landmark
                ls=get_xy(w,h,lm[L.LEFT_SHOULDER.value]); le=get_xy(w,h,lm[L.LEFT_ELBOW.value]); lw=get_xy(w,h,lm[L.LEFT_WRIST.value])
                rs=get_xy(w,h,lm[L.RIGHT_SHOULDER.value]); re=get_xy(w,h,lm[L.RIGHT_ELBOW.value]); rw=get_xy(w,h,lm[L.RIGHT_WRIST.value])
                lh=get_xy(w,h,lm[L.LEFT_HIP.value]); lk=get_xy(w,h,lm[L.LEFT_KNEE.value]); la=get_xy(w,h,lm[L.LEFT_ANKLE.value])
                rh=get_xy(w,h,lm[L.RIGHT_HIP.value]); rk=get_xy(w,h,lm[L.RIGHT_KNEE.value]); ra=get_xy(w,h,lm[L.RIGHT_ANKLE.value])

                angLE=calc_angle(ls,le,lw); angRE=calc_angle(rs,re,rw)
                angLK=calc_angle(lh,lk,la); angRK=calc_angle(rh,rk,ra)

                # simple counters
                if enableCurl:
                    if angLE<50 and curlL=="down": curlL="up"
                    if angLE>160 and curlL=="up": curlL="down"; curl_count+=1
                    if angRE<50 and curlR=="down": curlR="up"
                    if angRE>160 and curlR=="up": curlR="down"; curl_count+=1
                if enableSquat:
                    avgK=0.5*(angLK+angRK)
                    if avgK<90 and squatPhase=="up": squatPhase="down"
                    if avgK>160 and squatPhase=="down": squatPhase="up"; squat_count+=1

                # apply segmentation composite AFTER drawing
                if seg_apply:
                    out = apply_segmentation(out, res.segmentation_mask, seg_mode, bg_cache)

                # angles overlay
                cv2.putText(out, f"Elbow L:{angLE:5.1f} R:{angRE:5.1f}  Knee L:{angLK:5.1f} R:{angRK:5.1f}",
                            (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

            # HUD
            now=time.time(); fps=1.0/(now-prev) if now!=prev else 0.0; prev=now
            hud=[
                f"FPS: {fps:.1f}",
                f"Segmentation APPLY: {'ON' if seg_apply else 'OFF'} | Mode: {seg_mode} (M toggle, 1/2/3/4 select)",
                f"Curl: {'ON' if enableCurl else 'OFF'} (C) Count={curl_count} | Squat: {'ON' if enableSquat else 'OFF'} (S) Count={squat_count}",
                "Quit: q / ESC"
            ]
            draw_info(out, hud, 24)

            cv2.imshow("Body Tracking - Pose + Segmentation (Fixed)", out)
            k=cv2.waitKey(1)&0xFF
            if k in (27, ord('q'), ord('Q')): break
            elif k in (ord('m'), ord('M')): seg_apply = not seg_apply
            elif k == ord('1'): seg_mode = SEG_MODE_BLUR
            elif k == ord('2'): seg_mode = SEG_MODE_SOLID
            elif k == ord('3'): seg_mode = SEG_MODE_IMAGE
            elif k == ord('4'): seg_mode = SEG_MODE_MASK
            elif k in (ord('c'), ord('C')): enableCurl = not enableCurl
            elif k in (ord('s'), ord('S')): enableSquat = not enableSquat

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
