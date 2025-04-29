import streamlit as st
import time
import cv2
import numpy as np
import json
import urllib.parse
import base64
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
import tempfile
import webbrowser

# Globals
SMS_THRESHOLD = 0.98
NOTIFICATION_PHONE_NUMBER = "917842624422"
model = None
class_labels = None

# Setup
st.set_page_config(page_title="Pest Detection", layout="wide")

# Utility functions
def send_sms_alert(pest_type, confidence):
    message = f"🚨 ALERT: {pest_type} detected with {confidence:.1%} confidence! Take action immediately."
    encoded_message = urllib.parse.quote(message)
    url = f"https://wa.me/{NOTIFICATION_PHONE_NUMBER}?text={encoded_message}"
    webbrowser.open(url)
    st.success("Opened WhatsApp Web. Please click 'Send' manually.")

def load_class_labels():
    try:
        with open('class_labels.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return ["aphids", "armyworm", "beetle", "bollworm", "grasshopper",
                "mites", "mosquito", "sawfly", "stem_borer"]

def load_detection_model():
    global model, class_labels
    try:
        model = load_model('mobilenetv2_model.h5')
    except:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), include_top=False, weights='imagenet')
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = tf.keras.layers.Dense(9, activation='softmax')(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=output)
    class_labels = load_class_labels()

def predict_from_image(img_data):
    img = cv2.resize(img_data, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    predicted_class = class_labels[predicted_idx]
    confidence = float(predictions[0][predicted_idx])

    all_probs = sorted(
        [{"class": label, "probability": float(prob)}
         for label, prob in zip(class_labels, predictions[0])],
        key=lambda x: x["probability"],
        reverse=True
    )

    return predicted_class, confidence, all_probs

# Load model
load_detection_model()

# UI
st.title("🐛 Pest Detection with MobileNetV2")
tab1, tab2, tab3 = st.tabs(["📷 Live Capture", "📁 Upload Image", "🎞️ Upload Video"])

# Tab 1: Live Capture
with tab1:
    run = st.checkbox('Start Camera')
    frame_window = st.image([])
    pred_text = st.empty()  # Placeholder for predictions

    cap = cv2.VideoCapture(0)
    while run:
        success, frame = cap.read()
        if not success:
            st.error("Failed to capture from camera.")
            break

        pred_class, conf, probs = predict_from_image(frame)
        cv2.putText(frame, f"{pred_class}: {conf:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if conf > SMS_THRESHOLD:
            cv2.putText(frame, "ALERT SENT!", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            send_sms_alert(pred_class, conf)
            time.sleep(2)

        frame_window.image(frame, channels="BGR")

        # Show prediction probabilities
        pred_display = "### Prediction Probabilities:\n"
        for item in probs:
            pred_display += f"- **{item['class']}**: {item['probability']:.2%}\n"
        pred_text.markdown(pred_display)

    cap.release()

# Tab 2: Upload Image
with tab2:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img_pil = Image.open(uploaded).convert("RGB")
        img_np = np.array(img_pil)
        pred_class, conf, probs = predict_from_image(img_np)
        st.image(img_np, caption=f"{pred_class} ({conf:.2f})", use_column_width=True)
        st.write("### All Predictions:")
        for item in probs:
            st.write(f"{item['class']}: {item['probability']:.2f}")
        if conf > SMS_THRESHOLD:
            send_sms_alert(pred_class, conf)

# Tab 3: Upload Video
with tab3:
    st.subheader("Detect Pests in Uploaded Video")
    video_file = st.file_uploader("Upload a video (mp4)", type=["mp4"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        frame_window = st.image([])
        pred_text = st.empty()  # Placeholder for displaying predictions

        alerted_classes = set()  # Track classes already alerted

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            pred_class, conf, probs = predict_from_image(frame)
            cv2.putText(frame, f"{pred_class}: {conf:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Send alert once per pest type if confidence > 95%
            if conf > 0.95 and pred_class not in alerted_classes:
                send_sms_alert(pred_class, conf)
                alerted_classes.add(pred_class)

            frame_window.image(frame, channels="BGR")

            # Show all class predictions
            pred_display = "### Prediction Probabilities:\n"
            for item in probs:
                pred_display += f"- **{item['class']}**: {item['probability']:.2%}\n"
            pred_text.markdown(pred_display)

        cap.release()
