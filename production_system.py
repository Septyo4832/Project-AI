"""
production_system.py
"""

def classify_focus(facts):

    drowsy_limit      = 5
    microsleep_limit  = 10

    # =========================
    # RULE 1 - wajah tidak ada
    # =========================
    if not facts["face_detected"]:
        return "Undetected"

    # =========================
    # RULE 2 - microsleep
    # =========================
    if facts["eyes_closed_duration"] >= microsleep_limit:
        return "Microsleep"

    # =========================
    # RULE 3 - drowsy (mata tertutup 5-10 detik)
    # =========================
    if facts["eyes_closed_duration"] >= drowsy_limit:
        return "Drowsy"

    # =========================
    # RULE 4 - kepala menunduk & mata terbuka
    # =========================
    if facts["head_down"] and facts["eyes_open"]:
        return "Drowsy"

    # =========================
    # RULE 5 - main HP & mata terbuka
    # =========================
    if facts["phone_detected"] and facts["eyes_open"]:
        return "Distracted"

    # =========================
    # RULE 5b - melihat ke samping
    # =========================
    if facts["looking_away"]:
        return "Distracted"

    # =========================
    # RULE 6 - fokus
    # =========================
    if (
        facts["eyes_open"]
        and not facts["phone_detected"]
        and not facts["head_down"]
    ):
        return "Focused"

    return "Undetected"


def get_color(status):

    if status == "Focused":
        return (0, 255, 0)       # Hijau

    elif status == "Distracted":
        return (0, 165, 255)     # Oranye

    elif status == "Drowsy":
        return (0, 255, 255)     # Kuning

    elif status == "Microsleep":
        return (0, 0, 255)       # Merah

    return (128, 128, 128)       # Abu-abu (Undetected)