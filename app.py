import streamlit as st
import tempfile
from plate_recognition import process_video, process_image

st.set_page_config(page_title="License Plate Recognition", layout="centered")
st.title("🚗 License Plate Number Recognition")
st.write("Upload a video or an image and get only the detected license plate number.")

uploaded_file = st.file_uploader(
    "Upload a car video or image",
    type=["mp4", "avi", "mov", "jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    if st.button("Start Detection"):
        st.info("Processing... Please wait.")

        file_type = uploaded_file.name.split(".")[-1].lower()

        # Save uploaded file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.flush()

        if file_type in ["mp4", "avi", "mov"]:
            frame_placeholder = st.empty()
            text_placeholder = st.empty()

            texts = process_video(tfile.name)

        elif file_type in ["jpg", "jpeg", "png"]:
            texts = process_image(tfile.name)

        st.success("✅ Processing completed!")

        if texts:
            st.subheader("🔍 Detected License Plate Number(s):")
            for t in texts:
                st.success(t)
        else:
            st.warning("❌ No license plate number detected.")
