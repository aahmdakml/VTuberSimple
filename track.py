import cv2
import mediapipe as mp
import numpy as np
from pythonosc import udp_client
import math
import time

# --- KONFIGURASI UTAMA ---
OSC_IP = "127.0.0.1"
OSC_PORT = 39539
WEBCAM_ID = 0
TARGET_FPS = 30

# ==========================================
# === DATA KALIBRASI FINAL ANDA ===
# ==========================================

# 1. LENGAN (1, 1, 1)
ARM_INVERT_X = 1.0  
ARM_INVERT_Y = 1.0 
ARM_INVERT_Z = 1.0 
ARM_GAIN_XY  = 1.2
ARM_GAIN_Z   = 0.5 

# 2. JARI (Finger: L=Z, R=Z | Sign: L=1, R=-1)
# Axis Index: 0=X, 1=Y, 2=Z
FINGER_AXIS_L = 2 
FINGER_AXIS_R = 2
FINGER_SIGN_L = 1.0
FINGER_SIGN_R = -1.0
FINGER_SENSITIVITY = 1.3

# 3. JEMPOL (Thumb: L=Y, R=Y | Sign: L=-1, R=-1)
THUMB_AXIS_L = 1
THUMB_AXIS_R = 1
THUMB_SIGN_L = -1.0
THUMB_SIGN_R = 1.0

# ==========================================

# --- TUNING LAINNYA ---
EYE_Y_OFFSET = 0.02
GAZE_SENSITIVITY = 2.0   
PITCH_CORRECTION_FACTOR = 0.015
DEADZONE = 0.3      
NECK_RATIO = 0.5
EAR_THRESH_CLOSE, EAR_THRESH_OPEN = 0.15, 0.25
MOUTH_OPEN_MIN, MOUTH_OPEN_MAX = 5.0, 40.0  

# --- HELPER FUNCTIONS ---
def euler_to_quaternion(pitch, yaw, roll):
    qx = np.sin(pitch/2) * np.cos(yaw/2) * np.cos(roll/2) - np.cos(pitch/2) * np.sin(yaw/2) * np.sin(roll/2)
    qy = np.cos(pitch/2) * np.sin(yaw/2) * np.cos(roll/2) + np.sin(pitch/2) * np.cos(yaw/2) * np.sin(roll/2)
    qz = np.cos(pitch/2) * np.cos(yaw/2) * np.sin(roll/2) - np.sin(pitch/2) * np.sin(yaw/2) * np.cos(roll/2)
    qw = np.cos(pitch/2) * np.cos(yaw/2) * np.cos(roll/2) + np.sin(pitch/2) * np.sin(yaw/2) * np.sin(roll/2)
    return [qx, qy, qz, qw]

# Fungsi Rotasi Jari Custom (Bisa pilih Axis X/Y/Z)
def get_finger_quat(angle, axis_idx):
    s = math.sin(angle / 2)
    c = math.cos(angle / 2)
    if axis_idx == 0:   return [s, 0, 0, c] # X
    elif axis_idx == 1: return [0, s, 0, c] # Y
    elif axis_idx == 2: return [0, 0, s, c] # Z
    return [0, 0, 0, 1]

def get_limb_rotation(start, end, rest_vector):
    v_curr = np.array(end) - np.array(start)
    norm = np.linalg.norm(v_curr)
    if norm < 1e-6: return [0,0,0,1]
    v_curr = v_curr / norm
    v_rest = np.array(rest_vector)
    v_rest = v_rest / np.linalg.norm(v_rest)
    dot = np.dot(v_rest, v_curr)
    dot = max(-1.0, min(1.0, dot)) 
    angle = math.acos(dot)
    axis = np.cross(v_rest, v_curr)
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-6: return [0, 0, 0, 1]
    axis = axis / axis_len
    sin_half = math.sin(angle / 2)
    qx = axis[0] * sin_half
    qy = axis[1] * sin_half
    qz = axis[2] * sin_half
    qw = math.cos(angle / 2)
    return [qx, qy, qz, qw]

def calculate_ear(face_landmarks, indices, img_w, img_h):
    coords = []
    for idx in indices:
        lm = face_landmarks.landmark[idx]
        coords.append(np.array([lm.x * img_w, lm.y * img_h]))
    v1 = np.linalg.norm(coords[1] - coords[5])
    v2 = np.linalg.norm(coords[2] - coords[4])
    h  = np.linalg.norm(coords[0] - coords[3])
    return (v1 + v2) / (2.0 * h + 1e-6)

def get_relative_iris(face_landmarks, iris_idx, inner_idx, outer_idx, img_w, img_h):
    iris = np.array([face_landmarks.landmark[iris_idx].x * img_w, face_landmarks.landmark[iris_idx].y * img_h])
    inner = np.array([face_landmarks.landmark[inner_idx].x * img_w, face_landmarks.landmark[inner_idx].y * img_h])
    outer = np.array([face_landmarks.landmark[outer_idx].x * img_w, face_landmarks.landmark[outer_idx].y * img_h])
    eye_width = np.linalg.norm(outer - inner)
    eye_vec = outer - inner
    eye_vec_norm = eye_vec / (np.linalg.norm(eye_vec) + 1e-6)
    iris_vec = iris - inner
    proj_x = np.dot(iris_vec, eye_vec_norm)
    norm_x = (proj_x / eye_width) * 2.0 - 1.0
    cross_prod = (eye_vec[0] * (iris[1] - inner[1])) - (eye_vec[1] * (iris[0] - inner[0]))
    dist_y = cross_prod / eye_width
    norm_y = dist_y / (eye_width * 0.3)
    return norm_x, norm_y

def map_range(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def get_finger_curl(landmarks, tip_idx, knuckle_idx, wrist_idx):
    tip = np.array([landmarks.landmark[tip_idx].x, landmarks.landmark[tip_idx].y])
    wrist = np.array([landmarks.landmark[wrist_idx].x, landmarks.landmark[wrist_idx].y])
    dist_tip_wrist = np.linalg.norm(tip - wrist)
    knuckle = np.array([landmarks.landmark[knuckle_idx].x, landmarks.landmark[knuckle_idx].y])
    dist_palm = np.linalg.norm(knuckle - wrist)
    ratio = dist_tip_wrist / (dist_palm + 1e-6)
    curl = (ratio - 1.9) / (0.8 - 1.9)
    return max(0.0, min(1.0, curl)) * FINGER_SENSITIVITY

# --- CLASS STABILIZER ---
class Stabilizer:
    def __init__(self, state_num=2, measure_num=1, cov_process=0.0001, cov_measure=0.1):
        self.filter = cv2.KalmanFilter(state_num, measure_num, 0)
        self.state = np.zeros((state_num, 1), dtype=np.float32)
        self.filter.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        self.filter.measurementMatrix = np.array([[1, 1]], np.float32)
        self.filter.processNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * cov_process
        self.filter.measurementNoiseCov = np.array([[1]], np.float32) * cov_measure
    def update(self, measurement):
        self.filter.predict()
        self.filter.correct(np.array([[np.float32(measurement)]]))
        self.state = self.filter.statePost
        return self.state[0][0]

# --- INIT ---
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_face_landmarks=True, model_complexity=1)
client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

# Stabilizers
stab_pitch = Stabilizer(cov_process=0.02, cov_measure=0.1)
stab_yaw   = Stabilizer(cov_process=0.02, cov_measure=0.1)
stab_roll  = Stabilizer(cov_process=0.02, cov_measure=0.1)
stab_eye_x = Stabilizer(cov_process=0.005, cov_measure=0.1)
stab_eye_y = Stabilizer(cov_process=0.005, cov_measure=0.1)
stab_spine_roll = Stabilizer(cov_process=0.02, cov_measure=0.1)
stab_spine_yaw  = Stabilizer(cov_process=0.02, cov_measure=0.1)

# Stabilizer Jari (10 Jari)
stab_fingers_L = [Stabilizer(cov_process=0.1, cov_measure=0.1) for _ in range(5)]
stab_fingers_R = [Stabilizer(cov_process=0.1, cov_measure=0.1) for _ in range(5)]

model_points = np.array([(0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)], dtype=np.float64)
LEFT_EYE_IDXS, RIGHT_EYE_IDXS = [33, 160, 158, 133, 153, 144], [362, 385, 387, 263, 373, 380]
L_IRIS_C, L_IN, L_OUT = 468, 133, 33  
R_IRIS_C, R_IN, R_OUT = 473, 362, 263 
last_raw_pitch, last_raw_yaw, last_raw_roll = 0, 0, 0
blink_l_state, blink_r_state = 0.0, 0.0
prev_time = 0

# Config Jari
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Little"]
FINGER_INDICES = [(4, 2), (8, 5), (12, 9), (16, 13), (20, 17)] 
BONE_SUFFIXES = ["Proximal", "Intermediate", "Distal"]

# --- CAMERA ---
cap = cv2.VideoCapture(WEBCAM_ID)
cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) 

print("=== VTuber READY TO STREAM ===")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    image = cv2.flip(image, 1)
    img_h, img_w, _ = image.shape
    
    image.flags.writeable = False
    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image.flags.writeable = True

    # 1. FACE LOGIC
    if results.face_landmarks:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, None, mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))
        fl = results.face_landmarks
        image_points = np.array([
            (fl.landmark[1].x * img_w, fl.landmark[1].y * img_h),
            (fl.landmark[152].x * img_w, fl.landmark[152].y * img_h),
            (fl.landmark[263].x * img_w, fl.landmark[263].y * img_h),
            (fl.landmark[33].x * img_w, fl.landmark[33].y * img_h),
            (fl.landmark[287].x * img_w, fl.landmark[287].y * img_h),
            (fl.landmark[57].x * img_w, fl.landmark[57].y * img_h)
        ], dtype=np.float64)
        focal_length = img_w
        cam_matrix = np.array([[focal_length, 0, img_w/2], [0, focal_length, img_h/2], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))
        success_pnp, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        dY = (fl.landmark[263].y * img_h) - (fl.landmark[33].y * img_h)
        dX = (fl.landmark[263].x * img_w) - (fl.landmark[33].x * img_w)
        curr_pitch, curr_yaw, curr_roll = angles[0], angles[1], math.degrees(math.atan2(dY, dX))

        if abs(curr_pitch - last_raw_pitch) < DEADZONE: curr_pitch = last_raw_pitch
        else: last_raw_pitch = curr_pitch
        if abs(curr_yaw - last_raw_yaw) < DEADZONE: curr_yaw = last_raw_yaw
        else: last_raw_yaw = curr_yaw
        if abs(curr_roll - last_raw_roll) < DEADZONE: curr_roll = last_raw_roll
        else: last_raw_roll = curr_roll

        s_pitch = stab_pitch.update(curr_pitch)
        s_yaw, s_roll = stab_yaw.update(curr_yaw), stab_roll.update(curr_roll)

        neck_pitch, neck_yaw, neck_roll = s_pitch * NECK_RATIO, s_yaw * NECK_RATIO, s_roll * NECK_RATIO
        head_pitch, head_yaw, head_roll = s_pitch - neck_pitch, s_yaw - neck_yaw, s_roll - neck_roll
        
        raw_ear_l = calculate_ear(fl, LEFT_EYE_IDXS, img_w, img_h)
        raw_ear_r = calculate_ear(fl, RIGHT_EYE_IDXS, img_w, img_h)
        if raw_ear_l < EAR_THRESH_CLOSE: blink_l_state = 1.0
        elif raw_ear_l > EAR_THRESH_OPEN: blink_l_state = 0.0
        if raw_ear_r < EAR_THRESH_CLOSE: blink_r_state = 1.0
        elif raw_ear_r > EAR_THRESH_OPEN: blink_r_state = 0.0
        if s_yaw > 20.0: blink_r_state = blink_l_state 
        elif s_yaw < -20.0: blink_l_state = blink_r_state

        lx, ly = get_relative_iris(fl, L_IRIS_C, L_IN, L_OUT, img_w, img_h)
        rx, ry = get_relative_iris(fl, R_IRIS_C, R_IN, R_OUT, img_w, img_h)
        avg_x, avg_y = (lx + rx)/2.0, ((ly + ry)/2.0) - (s_pitch * PITCH_CORRECTION_FACTOR) + EYE_Y_OFFSET
        if not (blink_l_state > 0.5 or blink_r_state > 0.5):
            smooth_eye_x, smooth_eye_y = stab_eye_x.update(avg_x), stab_eye_y.update(avg_y)
        else:
            smooth_eye_x, smooth_eye_y = stab_eye_x.state[0][0], stab_eye_y.state[0][0]
        
        mouth_dist = np.linalg.norm(np.array([fl.landmark[13].x*img_w, fl.landmark[13].y*img_h]) - np.array([fl.landmark[14].x*img_w, fl.landmark[14].y*img_h]))
        mouth_open = max(0.0, min(1.0, (mouth_dist - 5.0) * (1.0/(35.0))))

        nqx, nqy, nqz, nqw = euler_to_quaternion(math.radians(neck_pitch), math.radians(neck_yaw), math.radians(neck_roll))
        client.send_message("/VMC/Ext/Bone/Pos", ["Neck", 0.0, 0.0, 0.0, float(nqx), float(nqy), float(nqz), float(nqw)])
        hqx, hqy, hqz, hqw = euler_to_quaternion(math.radians(head_pitch), math.radians(head_yaw), math.radians(head_roll))
        client.send_message("/VMC/Ext/Bone/Pos", ["Head", 0.0, 0.0, 0.0, float(hqx), float(hqy), float(hqz), float(hqw)])
        client.send_message("/VMC/Ext/Root/Pos", ["Root", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        client.send_message("/VMC/Ext/Blend/Val", ["Blink_L", float(blink_l_state)])
        client.send_message("/VMC/Ext/Blend/Val", ["Blink_R", float(blink_r_state)])
        client.send_message("/VMC/Ext/Blend/Val", ["A", float(mouth_open)])
        eqx, eqy, eqz, eqw = euler_to_quaternion(math.radians(smooth_eye_y*70.0), math.radians(smooth_eye_x*70.0), 0)
        client.send_message("/VMC/Ext/Bone/Pos", ["LeftEye", 0.0, 0.0, 0.0, float(eqx), float(eqy), float(eqz), float(eqw)])
        client.send_message("/VMC/Ext/Bone/Pos", ["RightEye", 0.0, 0.0, 0.0, float(eqx), float(eqy), float(eqz), float(eqw)])

    # 2. BODY & ARM TRACKING
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        lm = results.pose_landmarks.landmark

        def get_vec(idx): return [lm[idx].x, lm[idx].y, lm[idx].z]
        def to_unity_vec(mp_vec): 
            return np.array([ mp_vec[0] * ARM_INVERT_X * ARM_GAIN_XY, mp_vec[1] * ARM_INVERT_Y * ARM_GAIN_XY, mp_vec[2] * ARM_INVERT_Z * ARM_GAIN_Z ])

        l_sh, r_sh = get_vec(11), get_vec(12)
        spine_roll = stab_spine_roll.update((l_sh[1] - r_sh[1]) * -120.0) 
        spine_yaw = stab_spine_yaw.update((l_sh[2] - r_sh[2]) * -80.0)
        sqx, sqy, sqz, sqw = euler_to_quaternion(0, math.radians(spine_yaw), math.radians(spine_roll))
        client.send_message("/VMC/Ext/Bone/Pos", ["Spine", 0.0, 0.0, 0.0, float(sqx), float(sqy), float(sqz), float(sqw)])

        if lm[11].visibility > 0.5 and lm[13].visibility > 0.5:
            start, end = to_unity_vec(get_vec(11)), to_unity_vec(get_vec(13))
            q_lu = get_limb_rotation(start, end, [1.0, 0.0, 0.0])
            client.send_message("/VMC/Ext/Bone/Pos", ["LeftUpperArm", 0.0, 0.0, 0.0, float(q_lu[0]), float(q_lu[1]), float(q_lu[2]), float(q_lu[3])])
            if lm[15].visibility > 0.5:
                start, end = to_unity_vec(get_vec(13)), to_unity_vec(get_vec(15))
                q_ll = get_limb_rotation(start, end, [1.0, 0.0, 0.0])
                client.send_message("/VMC/Ext/Bone/Pos", ["LeftLowerArm", 0.0, 0.0, 0.0, float(q_ll[0]), float(q_ll[1]), float(q_ll[2]), float(q_ll[3])])

        if lm[12].visibility > 0.5 and lm[14].visibility > 0.5:
            start, end = to_unity_vec(get_vec(12)), to_unity_vec(get_vec(14))
            q_ru = get_limb_rotation(start, end, [-1.0, 0.0, 0.0])
            client.send_message("/VMC/Ext/Bone/Pos", ["RightUpperArm", 0.0, 0.0, 0.0, float(q_ru[0]), float(q_ru[1]), float(q_ru[2]), float(q_ru[3])])
            if lm[16].visibility > 0.5:
                start, end = to_unity_vec(get_vec(14)), to_unity_vec(get_vec(16))
                q_rl = get_limb_rotation(start, end, [-1.0, 0.0, 0.0])
                client.send_message("/VMC/Ext/Bone/Pos", ["RightLowerArm", 0.0, 0.0, 0.0, float(q_rl[0]), float(q_rl[1]), float(q_rl[2]), float(q_rl[3])])

        # --- 3. FINGER TRACKING (LOGIC BARU SESUAI DATA ANDA) ---
        
        # LEFT HAND (Finger=Z, Thumb=Y)
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            for i, (name, (tip, knuckle)) in enumerate(zip(FINGER_NAMES, FINGER_INDICES)):
                raw_curl = get_finger_curl(results.left_hand_landmarks, tip, knuckle, 0)
                curl = stab_fingers_L[i].update(raw_curl)
                
                # Gunakan Axis & Sign dari Kalibrasi Anda
                if name == "Thumb":
                    angle = curl * (math.pi / 2.0) * THUMB_SIGN_L
                    axis = THUMB_AXIS_L
                else:
                    angle = curl * (math.pi / 1.5) * FINGER_SIGN_L
                    axis = FINGER_AXIS_L
                
                fqx, fqy, fqz, fqw = get_finger_quat(angle, axis)
                
                for suffix in BONE_SUFFIXES: 
                    client.send_message("/VMC/Ext/Bone/Pos", [f"Left{name}{suffix}", 0.0, 0.0, 0.0, float(fqx), float(fqy), float(fqz), float(fqw)])

        # RIGHT HAND (Finger=Z, Thumb=Y)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            for i, (name, (tip, knuckle)) in enumerate(zip(FINGER_NAMES, FINGER_INDICES)):
                raw_curl = get_finger_curl(results.right_hand_landmarks, tip, knuckle, 0)
                curl = stab_fingers_R[i].update(raw_curl)
                
                if name == "Thumb":
                    angle = curl * (math.pi / 2.0) * THUMB_SIGN_R
                    axis = THUMB_AXIS_R
                else:
                    angle = curl * (math.pi / 1.5) * FINGER_SIGN_R
                    axis = FINGER_AXIS_R
                
                fqx, fqy, fqz, fqw = get_finger_quat(angle, axis)
                
                for suffix in BONE_SUFFIXES: 
                    client.send_message("/VMC/Ext/Bone/Pos", [f"Right{name}{suffix}", 0.0, 0.0, 0.0, float(fqx), float(fqy), float(fqz), float(fqw)])

    cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('VTuber ULTIMATE', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()