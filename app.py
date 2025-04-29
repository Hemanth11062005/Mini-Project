# app.py - Flask application for pest detection
from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import time
import json
import os
import base64
from io import BytesIO

app = Flask(__name__)

# Global variables
model = None
class_labels = None

def load_class_labels():
    """Load class labels from file or use default labels"""
    try:
        with open('class_labels.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Default pest labels if file not found
        return [
            "aphids", "armyworm", "beetle", "bollworm", "grasshopper",
            "mites", "mosquito", "sawfly", "stem_borer"
        ]

def load_detection_model():
    """Load and return the MobileNetV2 pest detection model"""
    global model, class_labels
    try:
        model = load_model('mobilenetv2_model.h5')
        print("Loaded MobileNetV2 model successfully")
    except:
        print("Error: MobileNetV2 model not found. Using placeholder for demo.")
        # For demo purposes, create a basic model that returns random values
        # In production, ensure the actual model is available
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = tf.keras.layers.Dense(9, activation='softmax')(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=output)
    
    class_labels = load_class_labels()
    print(f"Loaded {len(class_labels)} class labels: {class_labels}")
    return model

def predict_from_image(img_data):
    """
    Predict pest class from image data
    
    Args:
        img_data: Image data in numpy array format
        
    Returns:
        Dictionary with predicted class, confidence score, and all class probabilities
    """
    global model, class_labels
    
    # Ensure model is loaded
    if model is None:
        load_detection_model()
    
    # Resize image to match MobileNetV2 input size (224x224)
    img_height, img_width = 224, 224
    processed_img = cv2.resize(img_data, (img_width, img_height))
    
    # Convert from BGR to RGB (OpenCV uses BGR by default)
    processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    
    # Preprocess for MobileNetV2 input
    img_array = image.img_to_array(processed_img_rgb)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    
    # Get predicted class and confidence
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_labels[predicted_class_idx]
    confidence = float(predictions[0][predicted_class_idx])
    
    # Get all class probabilities
    all_probs = []
    for i, (label, prob) in enumerate(zip(class_labels, predictions[0])):
        all_probs.append({
            "class": label,
            "probability": float(prob)
        })
    
    # Sort probabilities by value (highest first)
    all_probs = sorted(all_probs, key=lambda x: x["probability"], reverse=True)
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": all_probs
    }

def generate_frames():
    """Generator function for streaming video frames"""
    camera = cv2.VideoCapture(0)
    
    # Set camera resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    last_prediction_time = 0
    prediction_interval = 1.0  # Seconds between predictions
    current_prediction = "No prediction yet"
    current_confidence = 0.0
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Make a copy of the frame for display
        display_frame = frame.copy()
        
        # Get current time
        current_time = time.time()
        
        # Process every prediction_interval seconds
        if current_time - last_prediction_time > prediction_interval:
            try:
                result = predict_from_image(frame)
                current_prediction = result["predicted_class"]
                current_confidence = result["confidence"]
                last_prediction_time = current_time
            except Exception as e:
                print(f"Prediction error: {e}")
        
        # Draw a bounding box to indicate where to position the pest
        h, w = display_frame.shape[:2]
        cv2.rectangle(display_frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
        
        # Display the prediction on the frame
        text = f"{current_prediction}: {current_confidence:.2f}"
        cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 255, 0), 2)
        
        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in the required format for Flask's Response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Small delay to reduce CPU usage
        time.sleep(0.03)
    
    camera.release()

@app.route('/')
def index():
    """Route for the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Route for streaming video"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Route for analyzing a captured image"""
    try:
        # Get image data from request
        data = request.get_json()
        image_data = data['image'].split(',')[1]  # Remove the "data:image/jpeg;base64," part
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Save the captured image if needed
        if not os.path.exists('static/captures'):
            os.makedirs('static/captures')
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"pest_capture_{timestamp}.jpg"
        filepath = f"static/captures/{filename}"
        cv2.imwrite(filepath, img)
        
        # Get prediction
        result = predict_from_image(img)
        
        # Add the saved image path to the result
        result["image_path"] = filepath
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Load model at startup
    load_detection_model()
    
    # Run the Flask app
    app.run(debug=True)