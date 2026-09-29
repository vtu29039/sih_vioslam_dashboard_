import os
import time

import cv2
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from src.camera_stream import CameraStream
from src.dynafusion_engine import DynaFusionEngine
from src.telemetry_controller import TelemetryController
from src.webrtc_processor import VIOProcessor

st.set_page_config(page_title="SIH VIO-SLAM Vehicle Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .stMetric { background-color: #1f2937; padding: 12px; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚁 Zero-GNSS VIO-SLAM Navigation & Risk Dashboard")
st.caption("Live Field Demonstration — DynaFusion-VIO Dynamic Outlier Rejection Engine")


def ice_servers():
    servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    # Optional TURN relay (needed on some mobile/college networks). Set as Space secrets.
    turn_url = os.environ.get("TURN_URL")
    if turn_url:
        servers.append(
            {
                "urls": [turn_url],
                "username": os.environ.get("TURN_USERNAME", ""),
                "credential": os.environ.get("TURN_CREDENTIAL", ""),
            }
        )
    return servers


mode = st.radio(
    "Camera source",
    ["Browser camera (works online)", "Server camera (vehicle / local machine)"],
    horizontal=True,
)

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Live Camera Feed (FAST Feature Anchors)")
    feed_area = st.container()
with col2:
    st.subheader("2. DynaFusion Dynamic Masking (Dynamic Foliage Red)")
    mask_spot = st.empty()

st.divider()
c3, c4, c5, c6 = st.columns(4)
risk_metric, steer_metric = c3.empty(), c4.empty()
dist_metric, lat_metric = c5.empty(), c6.empty()


def show_metrics(t, latency_ms):
    risk_metric.metric("Risk Level", t["risk_level"])
    steer_metric.metric(
        "Steering Angle",
        f"{t['steering_angle']}°",
        delta="Turn Right" if t["steering_angle"] > 0 else "Turn Left",
    )
    dist_metric.metric("Obstacle Proximity", f"{t['distance_m']} m")
    lat_metric.metric("Loop Latency", f"{latency_ms} ms", delta="50-65 ms target")


if mode.startswith("Browser"):
    with feed_area:
        ctx = webrtc_streamer(
            key="vio-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": ice_servers()}),
            video_processor_factory=VIOProcessor,
            media_stream_constraints={
                "video": {"width": 640, "height": 480, "frameRate": {"ideal": 20}},
                "audio": False,
            },
            async_processing=True,
        )
    if not ctx.state.playing:
        st.info("Click START above and allow camera access to begin the live analysis.")
    while ctx.state.playing:
        proc = ctx.video_processor
        if proc is not None:
            mask, telemetry, latency = proc.snapshot()
            if mask is not None:
                mask_spot.image(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB), channels="RGB")
                show_metrics(telemetry, latency)
        time.sleep(0.1)
else:
    run = st.checkbox("Initialize Vehicle Camera Loop", value=False)
    if run:
        cam = CameraStream()
        if not cam.is_open():
            st.error("No server-side camera found. Use 'Browser camera' when hosted online.")
        else:
            engine, controller = DynaFusionEngine(), TelemetryController()
            frame_spot = feed_area.empty()
            while run:
                t0 = time.time()
                frame, gray = cam.get_frame()
                if frame is None:
                    st.error("Camera acquisition error.")
                    break
                annotated, mask, _s, dyn_pixels, cx = engine.process_frame(frame, gray)
                telemetry = controller.compute_telemetry(dyn_pixels, cx, frame.shape[1])
                latency_ms = round((time.time() - t0) * 1000, 1)
                frame_spot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
                mask_spot.image(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB), channels="RGB")
                show_metrics(telemetry, latency_ms)
        cam.release()
