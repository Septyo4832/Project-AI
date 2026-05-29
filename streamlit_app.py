

import streamlit as st
import cv2
import numpy as np
import time
import threading
from collections import deque
import os
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from production_system import classify_focus, get_color

try:
    from yolo_detector import YOLODetector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False



MODEL_PATH               = "model/bestv3.pt"
EAR_THRESHOLD            = 0.20
LOOKING_AWAY_THRESHOLD_X = 0.15
LOOKING_DOWN_THRESHOLD_Y = 0.12
HISTORY_SIZE             = 60

LEFT_EYE_IDX  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
NOSE_TIP_IDX  = 1
FOREHEAD_IDX  = 10


RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})



@st.cache_resource(show_spinner="Memuat model YOLO...")
def load_detector(model_path: str):
    if not YOLO_AVAILABLE:
        return None
    if not os.path.exists(model_path):
        return None
    try:
        return YOLODetector(model_path)
    except Exception:
        return None


@st.cache_resource(show_spinner="Memuat MediaPipe...")
def load_face_mesh():
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )



def calculate_ear(landmarks, w, h):
    def get_point(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    lp = [get_point(i) for i in LEFT_EYE_IDX]
    ear_left = (
        np.linalg.norm(lp[1] - lp[5]) +
        np.linalg.norm(lp[2] - lp[4])
    ) / (2.0 * np.linalg.norm(lp[0] - lp[3]) + 1e-6)

    rp = [get_point(i) for i in RIGHT_EYE_IDX]
    ear_right = (
        np.linalg.norm(rp[1] - rp[5]) +
        np.linalg.norm(rp[2] - rp[4])
    ) / (2.0 * np.linalg.norm(rp[0] - rp[3]) + 1e-6)

    return (ear_left + ear_right) / 2.0


def check_head_direction(landmarks, look_thresh, down_thresh):
    nose     = landmarks[NOSE_TIP_IDX]
    forehead = landmarks[FOREHEAD_IDX]
    nose_center_offset = abs(nose.x - 0.5)
    looking_away  = nose_center_offset > look_thresh
    vertical_diff = nose.y - forehead.y
    head_down     = vertical_diff > (0.18 + down_thresh)
    return looking_away, head_down


def process_frame(frame, detector, face_mesh,
                  eyes_closed_start, eyes_closed_duration,
                  ear_thresh, look_thresh, down_thresh):

    frame = cv2.resize(frame, (640, 360))
    h, w, _ = frame.shape

    detected_classes = []
    if detector is not None:
        try:
            result           = detector.detect(frame)
            detected_classes = detector.get_detected_classes(result)
            annotated        = np.ascontiguousarray(result.plot())
        except Exception:
            annotated = frame.copy()
    else:
        annotated = frame.copy()

    rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_result = face_mesh.process(rgb)

    face_detected = False
    eyes_open     = False
    looking_away  = False
    head_down_mp  = False
    ear_val       = 0.0

    if mp_result.multi_face_landmarks:
        face_detected = True
        for face_landmarks in mp_result.multi_face_landmarks:
            ear_val = calculate_ear(face_landmarks.landmark, w, h)
            if ear_val < ear_thresh:
                if eyes_closed_start is None:
                    eyes_closed_start = time.time()
                eyes_closed_duration = time.time() - eyes_closed_start
            else:
                eyes_open            = True
                eyes_closed_start    = None
                eyes_closed_duration = 0
            looking_away, head_down_mp = check_head_direction(
                face_landmarks.landmark, look_thresh, down_thresh
            )
    else:
        eyes_closed_start    = None
        eyes_closed_duration = 0

    head_down = ("head_down" in detected_classes) or head_down_mp

    facts = {
        "face_detected":        face_detected,
        "eyes_open":            eyes_open,
        "eyes_closed":          not eyes_open,
        "eyes_closed_duration": eyes_closed_duration,
        "phone_detected":       "phone" in detected_classes,
        "head_down":            head_down,
        "looking_away":         looking_away,
    }

    status    = classify_focus(facts)
    color_bgr = get_color(status)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

    overlay = annotated.copy()
    cv2.rectangle(overlay, (20, 15), (300, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)
    cv2.putText(annotated, f"Status: {status}",                         (30, 45),  cv2.FONT_HERSHEY_SIMPLEX, 0.8,  color_rgb,     2)
    cv2.putText(annotated, f"EAR: {ear_val:.3f}",                       (30, 75),  cv2.FONT_HERSHEY_SIMPLEX, 0.6,  (200,200,200), 1)
    cv2.putText(annotated, f"Eyes Closed: {eyes_closed_duration:.1f}s", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,  (200,200,200), 1)
    cv2.putText(annotated, f"Looking Away: {looking_away}",             (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    cv2.putText(annotated, f"Head Down: {head_down}",                   (30, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

    return annotated, facts, status, ear_val, eyes_closed_start, eyes_closed_duration




class FocusDetector:

    def __init__(self):
        self.detector   = load_detector(MODEL_PATH)
        self.face_mesh  = load_face_mesh()
        self.lock       = threading.Lock()

       
        self.eyes_closed_start    = None
        self.eyes_closed_duration = 0.0
        self.last_status          = "Undetected"
        self.last_ear             = 0.0
        self.last_facts           = {}
        self.ear_history          = deque(maxlen=HISTORY_SIZE)
        self.status_history       = deque(maxlen=HISTORY_SIZE)
        self.session_stats        = {
            "total_frames": 0,
            "focused":      0,
            "distracted":   0,
            "drowsy":       0,
            "microsleep":   0,
            "undetected":   0,
            "alerts":       0,
            "start_time":   time.time(),
        }

      
        self.ear_thresh  = EAR_THRESHOLD
        self.look_thresh = LOOKING_AWAY_THRESHOLD_X
        self.down_thresh = LOOKING_DOWN_THRESHOLD_Y

    def update_thresholds(self, ear, look, down):
        with self.lock:
            self.ear_thresh  = ear
            self.look_thresh = look
            self.down_thresh = down

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
       
        img = frame.to_ndarray(format="bgr24")

        with self.lock:
            ear_thresh  = self.ear_thresh
            look_thresh = self.look_thresh
            down_thresh = self.down_thresh
            eyes_closed_start    = self.eyes_closed_start
            eyes_closed_duration = self.eyes_closed_duration

        annotated, facts, status, ear_val, eyes_closed_start, eyes_closed_duration = process_frame(
            img,
            self.detector,
            self.face_mesh,
            eyes_closed_start,
            eyes_closed_duration,
            ear_thresh,
            look_thresh,
            down_thresh,
        )

        with self.lock:
            self.eyes_closed_start    = eyes_closed_start
            self.eyes_closed_duration = eyes_closed_duration
            self.last_status          = status
            self.last_ear             = ear_val
            self.last_facts           = facts
            self.ear_history.append(ear_val)
            self.status_history.append(status)

            s   = self.session_stats
            s["total_frames"] += 1
            key = status.lower()
            if key in s:
                s[key] += 1
            if status in ("Drowsy", "Microsleep", "Distracted"):
                s["alerts"] += 1

       
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")




st.set_page_config(
    page_title="Focus Detection System",
    page_icon="👁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .main { background-color: #0d0f14; }
    .status-box {
        border-radius: 8px; padding: 16px 20px;
        font-family: 'Courier New', monospace;
        font-size: 1.6rem; font-weight: 700;
        text-align: center; letter-spacing: 0.05em; margin-bottom: 10px;
    }
    .status-focused    { background: #0a2e1a; border: 2px solid #00ff7f; color: #00ff7f; }
    .status-distracted { background: #2e1a00; border: 2px solid #ff8c00; color: #ff8c00; }
    .status-drowsy     { background: #2e2a00; border: 2px solid #ffd700; color: #ffd700; }
    .status-microsleep { background: #2e0000; border: 2px solid #ff2020; color: #ff2020; }
    .status-undetected { background: #1a1a1a; border: 2px solid #888888; color: #888888; }
    .metric-card { background: #161a24; border: 1px solid #2a2f3d; border-radius: 8px; padding: 14px; text-align: center; }
    .metric-label { font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px; }
    .metric-value { font-family: 'Courier New', monospace; font-size: 1.4rem; font-weight: 700; color: #e2e8f0; }
    .rule-card { background: #0f1420; border-left: 3px solid #3b82f6; border-radius: 0 6px 6px 0; padding: 8px 14px; margin-bottom: 6px; font-size: 0.85rem; color: #94a3b8; }
    .stButton>button { background: #1d4ed8; color: white; border: none; border-radius: 6px; font-weight: 600; padding: 10px 24px; width: 100%; }
    .stButton>button:hover { background: #2563eb; }
    div[data-testid="stSidebarContent"] { background: #0d0f14; }
</style>
""", unsafe_allow_html=True)



if "detector_instance" not in st.session_state:
    st.session_state.detector_instance = None



with st.sidebar:
    st.markdown("## Konfigurasi")
    st.markdown("---")

    st.markdown("**Threshold Mata**")
    ear_thresh = st.slider("EAR Threshold", 0.10, 0.35, EAR_THRESHOLD, 0.01)

    st.markdown("**Threshold Kepala**")
    look_thresh = st.slider("Looking Away X", 0.05, 0.30, LOOKING_AWAY_THRESHOLD_X, 0.01)
    down_thresh = st.slider("Head Down Y",    0.05, 0.25, LOOKING_DOWN_THRESHOLD_Y, 0.01)

    st.markdown("---")
    st.markdown("**Timer Alert**")
    drowsy_limit     = st.number_input("Drowsy (detik)",     1, 30, 5)
    microsleep_limit = st.number_input("Microsleep (detik)", 2, 60, 10)

    st.markdown("---")
    st.markdown("**Basis Pengetahuan**")
    rules = [
        "RULE 1: Wajah tidak terdeteksi -> Undetected",
        f"RULE 2: Mata tutup >= {microsleep_limit}s -> Microsleep",
        f"RULE 3: Mata tutup >= {drowsy_limit}s -> Drowsy",
        "RULE 4: Kepala menunduk & mata buka -> Drowsy",
        "RULE 5: HP terdeteksi & mata buka -> Distracted",
        "RULE 5b: Melihat ke samping -> Distracted",
        "RULE 6: Semua normal -> Focused",
    ]
    for r in rules:
        st.markdown(f'<div class="rule-card">{r}</div>', unsafe_allow_html=True)



st.markdown("# Driver Focus Detection System")
st.markdown("Sistem deteksi fokus berbasis **YOLOv8 + MediaPipe + Production System (Rule-Based)**")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Live Detection", "Session Statistics", "Tentang Sistem"])




with tab1:

    col_vid, col_info = st.columns([2, 1])

    with col_vid:
        st.markdown("### Feed Kamera")

       
        ctx = webrtc_streamer(
            key="focus-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_processor_factory=FocusDetector,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

       
        if ctx.video_processor:
            st.session_state.detector_instance = ctx.video_processor
           
            ctx.video_processor.update_thresholds(ear_thresh, look_thresh, down_thresh)

    with col_info:
        st.markdown("### Status Real-time")
        status_placeholder = st.empty()
        st.markdown("### Metrics")
        m1, m2 = st.columns(2)
        ear_placeholder   = m1.empty()
        close_placeholder = m2.empty()
        st.markdown("### Facts")
        facts_placeholder = st.empty()
        st.markdown("### EAR History")
        chart_placeholder = st.empty()

    
    if ctx.state.playing and ctx.video_processor:
        proc = ctx.video_processor

        with proc.lock:
            status   = proc.last_status
            ear_val  = proc.last_ear
            facts    = dict(proc.last_facts)
            eye_dur  = proc.eyes_closed_duration
            ear_hist = list(proc.ear_history)

        
        css_class = f"status-{status.lower()}"
        status_placeholder.markdown(
            f'<div class="status-box {css_class}">{status.upper()}</div>',
            unsafe_allow_html=True
        )

        
        ear_placeholder.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">EAR</div>'
            f'<div class="metric-value">{ear_val:.3f}</div></div>',
            unsafe_allow_html=True
        )
        close_placeholder.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Eyes Closed</div>'
            f'<div class="metric-value">{eye_dur:.1f}s</div></div>',
            unsafe_allow_html=True
        )

  
        if facts:
            facts_md = "\n".join([f"| `{k}` | `{v}` |" for k, v in facts.items()])
            facts_placeholder.markdown("| Fact | Value |\n|---|---|\n" + facts_md)

       
        if len(ear_hist) > 1:
            chart_placeholder.line_chart(ear_hist, height=120)

        
        time.sleep(0.5)
        st.rerun()

    else:
        status_placeholder.markdown(
            '<div class="status-box status-undetected">WAITING</div>',
            unsafe_allow_html=True
        )




with tab2:

    st.markdown("### Statistik Sesi")

    proc = st.session_state.get("detector_instance")

    if proc is not None:
        with proc.lock:
            s = dict(proc.session_stats)
    else:
        s = {
            "total_frames": 0, "focused": 0, "distracted": 0,
            "drowsy": 0, "microsleep": 0, "undetected": 0,
            "alerts": 0, "start_time": None,
        }

    total = max(s["total_frames"], 1)

    if s["start_time"]:
        elapsed = time.time() - s["start_time"]
        st.markdown(f"Durasi sesi: **{elapsed:.0f} detik**")
    else:
        st.markdown("Belum ada sesi berjalan.")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Frame",  s["total_frames"])
    c2.metric("Focused",      s["focused"],    f"{s['focused']    / total * 100:.1f}%")
    c3.metric("Distracted",   s["distracted"], f"{s['distracted'] / total * 100:.1f}%")
    c4.metric("Drowsy",       s["drowsy"],     f"{s['drowsy']     / total * 100:.1f}%")
    c5.metric("Microsleep",   s["microsleep"], f"{s['microsleep'] / total * 100:.1f}%")

    st.markdown("---")
    st.markdown("### Distribusi Status")

    if s["total_frames"] > 0:
        import pandas as pd
        dist_data = {
            "Status": ["Focused", "Distracted", "Drowsy", "Microsleep", "Undetected"],
            "Frames": [s["focused"], s["distracted"], s["drowsy"], s["microsleep"], s["undetected"]],
            "Persen": [
                round(s["focused"]    / total * 100, 1),
                round(s["distracted"] / total * 100, 1),
                round(s["drowsy"]     / total * 100, 1),
                round(s["microsleep"] / total * 100, 1),
                round(s["undetected"] / total * 100, 1),
            ]
        }
        df = pd.DataFrame(dist_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.bar_chart(df.set_index("Status")["Frames"])




with tab3:

    st.markdown("### Arsitektur Sistem")
    st.markdown("""
**1. YOLOv8 (Object Detection)**
- Model custom `bestv3.pt` untuk mendeteksi objek relevan: `phone`, `head_down`
- Dijalankan pada setiap frame video

**2. MediaPipe Face Mesh**
- 468 landmark wajah untuk analisis detail
- Menghitung Eye Aspect Ratio (EAR) untuk deteksi mata tertutup
- Estimasi arah kepala dari posisi hidung dan dahi

**3. Production System (Rule-Based Inference)**
- Knowledge base berisi 7 aturan produksi
- Input: facts dictionary dari YOLO + MediaPipe
- Output: status klasifikasi fokus
    """)

    st.markdown("---")
    st.markdown("### Rumus EAR")
    st.latex(r"EAR = \frac{||p2-p6|| + ||p3-p5||}{2 \cdot ||p1-p4||}")
    st.markdown("Nilai EAR < threshold maka mata dianggap tertutup")

    st.markdown("---")
    st.markdown("### Kelas Output")
    classes = {
        "Focused":    "Mata terbuka, tidak ada HP, kepala normal",
        "Distracted": "HP terdeteksi atau melihat ke samping",
        "Drowsy":     f"Kepala menunduk atau mata tertutup > {drowsy_limit}s",
        "Microsleep": f"Mata tertutup > {microsleep_limit}s",
        "Undetected": "Wajah tidak terdeteksi",
    }
    for k, v in classes.items():
        st.markdown(f"**{k}**: {v}")

    st.markdown("---")
    st.markdown("### Dependencies")
    st.code("""
ultralytics==8.3.0
mediapipe==0.10.14
opencv-contrib-python==4.10.0.84
streamlit==1.36.0
streamlit-webrtc
numpy==1.26.4
scipy==1.14.1
torch==2.2.2
torchvision==0.17.2
    """, language="text")