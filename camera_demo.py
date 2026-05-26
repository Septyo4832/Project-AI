import cv2
import time
import numpy as np
import mediapipe as mp

from yolo_detector import YOLODetector
from production_system import classify_focus, get_color


# =========================
# KONFIG
# =========================

MODEL_PATH = "model/bestv3.pt"

EAR_THRESHOLD = 0.20

# Threshold arah kepala (nilai 0.0 - 1.0, relatif lebar/tinggi frame)
LOOKING_AWAY_THRESHOLD_X = 0.15   # jarak dari tengah horizontal
LOOKING_DOWN_THRESHOLD_Y = 0.12   # jarak dahi ke hidung secara vertikal


# =========================
# LOAD YOLO
# =========================

detector = YOLODetector(MODEL_PATH)


# =========================
# LOAD MEDIAPIPE
# =========================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)


# =========================
# BUKA KAMERA
# =========================

cap = cv2.VideoCapture(0)


# =========================
# TIMER
# =========================

eyes_closed_start = None
eyes_closed_duration = 0


# =========================
# LANDMARK INDEX MATA
# =========================

# Mata kiri  : p1, p2, p3, p4, p5, p6
LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]

# Mata kanan : p1, p2, p3, p4, p5, p6
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Hidung tip & dahi untuk head pose
NOSE_TIP_IDX  = 1
FOREHEAD_IDX  = 10


# =========================
# FUNGSI HITUNG EAR
# =========================

def calculate_ear(landmarks, w, h):
    """
    Hitung Eye Aspect Ratio (EAR) dari landmark MediaPipe.

    Rumus EAR:
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Keterangan:
        p1, p4 = sudut kiri & kanan mata (horizontal)
        p2, p3, p5, p6 = titik atas & bawah mata (vertikal)

    Return rata-rata EAR kiri dan kanan.
    Semakin kecil nilai EAR → mata semakin tertutup.
    """

    def get_point(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    # --- Mata kiri ---
    lp = [get_point(i) for i in LEFT_EYE_IDX]
    ear_left = (
        np.linalg.norm(lp[1] - lp[5]) +
        np.linalg.norm(lp[2] - lp[4])
    ) / (2.0 * np.linalg.norm(lp[0] - lp[3]) + 1e-6)

    # --- Mata kanan ---
    rp = [get_point(i) for i in RIGHT_EYE_IDX]
    ear_right = (
        np.linalg.norm(rp[1] - rp[5]) +
        np.linalg.norm(rp[2] - rp[4])
    ) / (2.0 * np.linalg.norm(rp[0] - rp[3]) + 1e-6)

    return (ear_left + ear_right) / 2.0


# =========================
# FUNGSI CEK ARAH KEPALA
# =========================

def check_head_direction(landmarks):
    """
    Cek apakah kepala menoleh ke samping atau menunduk.

    Menoleh (looking_away):
        Posisi hidung (x) jauh dari tengah frame (0.5).
        Misal hidung di x=0.3 → menoleh kiri.

    Menunduk (head_down):
        Posisi hidung (y) jauh di bawah dahi (y).
        Nilai y MediaPipe bertambah ke bawah.

    Return:
        looking_away (bool), head_down (bool)
    """

    nose     = landmarks[NOSE_TIP_IDX]
    forehead = landmarks[FOREHEAD_IDX]

    # Menoleh ke samping: hidung jauh dari tengah horizontal
    nose_center_offset = abs(nose.x - 0.5)
    looking_away = nose_center_offset > LOOKING_AWAY_THRESHOLD_X

    # Menunduk: hidung turun jauh dari dahi
    # (y bertambah ke bawah, jadi nose.y > forehead.y saat normal)
    vertical_diff = nose.y - forehead.y
    head_down = vertical_diff > (0.18 + LOOKING_DOWN_THRESHOLD_Y)

    return looking_away, head_down


# =========================
# LOOP UTAMA
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))

    h, w, c = frame.shape

    # =========================
    # YOLO DETECT
    # =========================

    result = detector.detect(frame)

    detected_classes = detector.get_detected_classes(result)

    # =========================
    # MEDIAPIPE
    # =========================

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_result = face_mesh.process(rgb)

    # =========================
    # DEFAULT FACT
    # =========================

    face_detected = False
    eyes_open     = False
    looking_away  = False
    head_down_mp  = False   # dari MediaPipe

    # =========================
    # CEK FACE LANDMARK
    # =========================

    if mp_result.multi_face_landmarks:

        face_detected = True

        for face_landmarks in mp_result.multi_face_landmarks:

            # =========================
            # HITUNG EAR
            # =========================

            ear = calculate_ear(
                face_landmarks.landmark,
                w,
                h
            )

            # =========================
            # CEK MATA
            # =========================

            if ear < EAR_THRESHOLD:

                if eyes_closed_start is None:
                    eyes_closed_start = time.time()

                eyes_closed_duration = (
                    time.time() - eyes_closed_start
                )

            else:

                eyes_open         = True
                eyes_closed_start = None
                eyes_closed_duration = 0

            # =========================
            # CEK ARAH KEPALA
            # =========================

            looking_away, head_down_mp = check_head_direction(
                face_landmarks.landmark
            )

    else:

        eyes_closed_start    = None
        eyes_closed_duration = 0

    # =========================
    # GABUNG HEAD_DOWN:
    # prioritas YOLO, fallback MediaPipe
    # =========================

    head_down = (
        "head_down" in detected_classes
        or head_down_mp
    )

    # =========================
    # FACTS
    # =========================

    facts = {

        "face_detected": face_detected,

        "eyes_open": eyes_open,

        "eyes_closed": not eyes_open,

        "eyes_closed_duration":
            eyes_closed_duration,

        "phone_detected":
            "phone" in detected_classes,

        "head_down": head_down,

        "looking_away": looking_away
    }

    # =========================
    # PRODUCTION SYSTEM
    # =========================

    status = classify_focus(facts)

    color = get_color(status)

    # =========================
    # DISPLAY
    # =========================

    output_frame = result.plot()

    cv2.putText(
        output_frame,
        f"Status: {status}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.putText(
        output_frame,
        f"EAR: {ear:.2f}" if face_detected else "EAR: -",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    cv2.putText(
        output_frame,
        f"Eyes Closed: {eyes_closed_duration:.1f}s",
        (30, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    cv2.putText(
        output_frame,
        f"Looking Away: {looking_away}",
        (30, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    cv2.putText(
        output_frame,
        f"Head Down: {head_down}",
        (30, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    cv2.imshow(
        "Deteksi Fokus Mahasiswa",
        output_frame
    )

    # keluar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()