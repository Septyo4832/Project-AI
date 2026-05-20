# note: jalankan "pip install -r requirements.txt" 
# sebelum mulai program untuk menginstal pip terlebih dahulu


"""
camera_demo.py

TODO:
- Buka kamera laptop.
- Ambil frame dari kamera.
- Kirim frame ke YOLO.
- Ambil hasil deteksi.
- Ubah hasil deteksi menjadi fakta.
- Kirim fakta ke Production System.
- Tampilkan status di layar.
"""

import cv2
import time

from yolo_detector import YOLODetector
from production_system import classify_focus, get_color


# =========================
# TODO: KONFIGURASI AWAL
# =========================

MODEL_PATH = "model/best.pt"
CAMERA_INDEX = 0


def main():
    # TODO: Load YOLO detector
    detector = YOLODetector(MODEL_PATH)

    # TODO: Buka kamera
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Kamera tidak bisa dibuka.")
        print("TODO: Coba ubah CAMERA_INDEX menjadi 1 atau 2.")
        return

    # TODO:
    # Variabel untuk menghitung durasi mata tertutup
    eyes_closed_start = None
    eyes_closed_duration = 0

    while True:
        # TODO: Baca frame dari kamera
        ret, frame = cap.read()

        if not ret:
            print("Frame tidak terbaca.")
            break

        # TODO: Resize frame jika diperlukan
        frame = cv2.resize(frame, (960, 540))

        # TODO: Deteksi YOLO
        result = detector.detect(frame)

        # TODO: Ambil nama class yang terdeteksi
        detected_classes = detector.get_detected_classes(result)

        # =========================
        # TODO: HITUNG DURASI MATA TERTUTUP
        # =========================

        if "face_eyes_closed" in detected_classes:
            if eyes_closed_start is None:
                eyes_closed_start = time.time()

            eyes_closed_duration = time.time() - eyes_closed_start

        else:
            eyes_closed_start = None
            eyes_closed_duration = 0

        # =========================
        # TODO: BUAT FAKTA
        # =========================

        facts = detector.make_facts(
            detected_classes,
            eyes_closed_duration
        )

        # =========================
        # TODO: PRODUCTION SYSTEM
        # =========================

        status = classify_focus(facts)
        color = get_color(status)

        # =========================
        # TODO: TAMPILKAN HASIL
        # =========================

        # Gambar bounding box dari YOLO
        output_frame = result.plot()

        # Tampilkan status
        cv2.putText(
            output_frame,
            "Status: " + status,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2
        )

        # Tampilkan durasi mata tertutup
        cv2.putText(
            output_frame,
            "Eyes Closed: " + str(round(eyes_closed_duration, 1)) + "s",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        # Tampilkan class yang terdeteksi
        cv2.putText(
            output_frame,
            "Detected: " + str(detected_classes),
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Tampilkan window
        cv2.imshow("Demo Deteksi Fokus Mahasiswa", output_frame)

        # TODO:
        # Tekan q untuk keluar
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # TODO: Tutup kamera
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()