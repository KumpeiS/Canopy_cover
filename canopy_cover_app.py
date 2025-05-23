import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io

st.set_page_config(page_title="Canopy Cover Estimation", layout="wide")
st.title("🌿 Canopy Cover Estimation from Image")

# --- 1. Upload image ---
uploaded_file = st.file_uploader("Upload an image with a 20cm black square for scale", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(img)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]

    st.image(img, caption='Uploaded Image', use_column_width=True)

    # --- 2. Scale selection ---
    st.subheader("Step 1: Define 20cm Scale")
    st.markdown("Input the coordinates of the top-left and bottom-right corners of the 20cm square:")
    col1, col2 = st.columns(2)
    with col1:
        x1 = st.number_input("Top-Left X", value=100)
        y1 = st.number_input("Top-Left Y", value=100)
    with col2:
        x2 = st.number_input("Bottom-Right X", value=300)
        y2 = st.number_input("Bottom-Right Y", value=300)

    scale_length = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    pixels_per_cm = scale_length / 20
    side_px = round(20 * pixels_per_cm)
    st.write(f"Pixels per cm: {pixels_per_cm:.2f}, 20cm = {side_px} px")

    # --- 3. ROI selection ---
    st.subheader("Step 2: Select Center Point for ROI")
    cx = st.number_input("Center X", value=200)
    cy = st.number_input("Center Y", value=200)

    half = side_px // 2
    x_start, x_end = max(0, cx - half), min(w, cx + half)
    y_start, y_end = max(0, cy - half), min(h, cy + half)
    roi = img_cv[y_start:y_end, x_start:x_end]

    # --- 4. Green mask extraction ---
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # --- 5. Display ---
    st.subheader("Step 3: Result")
    col3, col4 = st.columns(2)
    with col3:
        st.image(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB), caption='20cm x 20cm ROI', use_column_width=True)
    with col4:
        st.image(mask, caption='Green Area Mask', clamp=True, use_column_width=True)

    # --- 6. Canopy cover calculation ---
    canopy_cover = (np.count_nonzero(mask) / mask.size) * 100
    st.success(f"🌿 Canopy Cover: {canopy_cover:.2f}%")
