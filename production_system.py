"""
production_system.py

TODO:
- File ini digunakan untuk menentukan status mahasiswa.
- Input-nya berupa fakta dari hasil deteksi YOLO.
- Output-nya berupa status:
  Focused, Distracted, Drowsy, Microsleep, atau Undetected.
"""


def classify_focus(facts):

    # TODO: Buat aturan IF-THEN berdasarkan fakta yang diterima.

    """
    Contoh facts:
    {
        "face_detected": True,
        "eyes_open": True,
        "eyes_closed": False,
        "phone_detected": False,
        "head_down": False,
        "eyes_closed_duration": 0
    }
    """

    # TODO: Atur batas durasi sesuaikan dengan laporan
    drowsy_limit = 5
    microsleep_limit = 10

    # TODO: Rule 1 - Jika wajah tidak terdeteksi
    if facts["face_detected"] == False:
        return "Undetected"

    # TODO: Rule 2 - Jika mata tertutup terlalu lama
    if facts["eyes_closed_duration"] >= microsleep_limit:
        return "Microsleep"

    # TODO: Rule 3 - Jika mata tertutup beberapa detik
    if facts["eyes_closed_duration"] >= drowsy_limit:
        return "Drowsy"

    # TODO: Rule 4 - Jika HP terdeteksi
    if facts["phone_detected"] == True:
        return "Distracted"

    # TODO: Rule 5 - Jika kepala menunduk
    if facts["head_down"] == True:
        return "Drowsy"

    # TODO: Rule 6 - Jika mata terbuka dan tidak ada gangguan
    if facts["eyes_open"] == True:
        return "Focused"

    # TODO: Jika tidak ada kondisi yang cocok
    return "Undetected"


def get_color(status):

    # TODO: Warna untuk tampilan teks di OpenCV. Format warna OpenCV adalah BGR.


    if status == "Focused":
        return (0, 255, 0)      # hijau
    elif status == "Distracted":
        return (0, 165, 255)    # oranye
    elif status == "Drowsy":
        return (0, 255, 255)    # kuning
    elif status == "Microsleep":
        return (0, 0, 255)      # merah
    else:
        return (128, 128, 128)  # abu-abu