import cv2
import time
import mediapipe as mp

from yolo_detector import YOLODetector
from production_system import classify_focus, get_color


# =========================
# KONFIG
# =========================

MODEL_PATH = "model/bestv3.pt"

EAR_THRESHOLD = 0.20

# TODO:
# sesuaikan threshold EAR
# semakin kecil = mata dianggap tertutup


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
# FUNGSI HITUNG EAR
# =========================

def calculate_ear(landmarks, w, h):

    """
    TODO:
    Hitung Eye Aspect Ratio.
    
    EAR dipakai untuk menentukan:
    mata terbuka / tertutup

    Kalian perlu:
    - ambil landmark mata kiri/kanan
    - hitung jarak vertikal
    - hitung jarak horizontal
    """

    # sementara dummy
    return 0.3


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
    eyes_open = False
    looking_away = False

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

                eyes_open = True
                eyes_closed_start = None
                eyes_closed_duration = 0

            # =========================
            # TODO:
            # CEK ARAH KEPALA
            # =========================

            """
            Bisa pakai:
            - nose landmark
            - face angle
            - solvePnP OpenCV
            """

            # dummy sementara
            looking_away = False

    else:

        eyes_closed_start = None
        eyes_closed_duration = 0

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

        "head_down":
            "head_down" in detected_classes,

        "looking_away":
            looking_away
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
        f"Eyes Closed: {eyes_closed_duration:.1f}s",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
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