"""
production_system.py
"""

def classify_focus(facts):

    drowsy_limit = 5
    microsleep_limit = 10

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
    # RULE 3 - drowsy
    # =========================
    if facts["eyes_closed_duration"] >= drowsy_limit:
        return "Drowsy"

    # =========================
    # RULE 4 - main HP
    # =========================
    if facts["phone_detected"]:
        return "Distracted"

    # =========================
    # RULE 5 - melihat ke samping
    # =========================
    if facts["looking_away"]:
        return "Distracted"

    # =========================
    # RULE 6 - kepala menunduk
    # =========================
    if facts["head_down"]:
        return "Drowsy"

    # =========================
    # RULE 7 - fokus
    # =========================
    if facts["eyes_open"]:
        return "Focused"

    return "Undetected"


def get_color(status):

    if status == "Focused":
        return (0, 255, 0)

    elif status == "Distracted":
        return (0, 165, 255)

    elif status == "Drowsy":
        return (0, 255, 255)

    elif status == "Microsleep":
        return (0, 0, 255)

    return (128, 128, 128)