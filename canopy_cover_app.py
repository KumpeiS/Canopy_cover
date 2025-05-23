import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Canopy Cover with Click", layout="wide")
st.title("🌿 Canopy Cover Estimation with Clickable Scale")

# 1. Upload image
uploaded_file = st.file_uploader("Upload an image (with 20cm square black paper)", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # 🔧 Resize image to max 1000px width
    MAX_WIDTH = 1000
    if image.width > MAX_WIDTH:
        ratio = MAX_WIDTH / image.width
        new_height = int(image.height * ratio)
        image = image.resize((MAX_WIDTH, new_height))

    img_np = np.array(image)
    h, w = img_np.shape[:2]
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # 2. Draw rectangle for scale
    st.subheader("Step 1: Draw a rectangle over the 20cm black square")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=2,
        background_image=image,
        update_streamlit=True,
        height=h,
        width=w,
        drawing_mode="rect",
        key="scale",
    )

    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        obj = canvas_result.json_data["objects"][-1]
        x = obj["left"]
        y = obj["top"]
        width_box = obj["width"]
        height_box = obj["height"]
        pixel_length = (width_box**2 + height_box**2) ** 0.5
        pixels_per_cm = pixel_length / 20
        side_px = round(20 * pixels_per_cm)
        st.success(f"Scale length: {pixel_length:.2f} px → {pixels_per_cm:.2f} px/cm → 20cm = {side_px} px")

        # 3. Click to select center point
        st.subheader("Step 2: Click the center of 20cm square ROI")
        canvas_center = st_canvas(
            fill_color="rgba(255, 255, 0, 0.3)",
            stroke_width=2,
            background_image=image,
            update_streamlit=True,
            height=h,
            width=w,
            drawing_mode="circle",
            key="center",
        )

        if canvas_center.json_data and len(canvas_center.json_data["objects"]) > 0:
            center_obj = canvas_center.json_data["objects"][-1]
            cx = int(center_obj["left"])
            cy = int(center_obj["top"])
            st.success(f"ROI center selected: ({cx}, {cy})")

            # 4. ROI extraction
            half = side_px // 2
            x1, x2 = max(0, int(cx - half)), min(w, int(cx + half))
            y1, y2 = max(0, int(cy - half)), min(h, int(cy + half))

            roi = img_np[y1:y2, x1:x2]
            roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

            lower_green = np.array([35, 40, 40])
            upper_green = np.array([90, 255, 255])
            mask = cv2.inRange(hsv, lower_green, upper_green)

            # 5. Results
            st.subheader("Step 3: Results")
            col1, col2 = st.columns(2)
            with col1:
                st.image(roi, caption="20cm x 20cm ROI", use_container_width=True)
            with col2:
                st.image(mask, caption="Green Area Mask", clamp=True, use_container_width=True)

            canopy_cover = (np.count_nonzero(mask) / mask.size) * 100
            st.success(f"🌿 Canopy Cover: {canopy_cover:.2f}%")
        else:
            st.warning("Please click on the center point of the ROI.")
    else:
        st.warning("Please draw a rectangle over the 20cm black square.")
