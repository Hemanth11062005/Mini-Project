# Pest Detection Web Application

A Streamlit-based web application for detecting and classifying agricultural pests using a webcam and TensorFlow model.

## Features

- Real-time video streaming from webcam
- On-demand image capture and analysis
- Detailed probability breakdown for pest classification
- Visual guidance for positioning pests in frame
- Responsive web interface built with Bootstrap

## Requirements

- Python 3.8+
- Webcam/camera
- Required Python packages (see requirements.txt)

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/pest-detection-webapp.git
   cd pest-detection-webapp
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Place your trained MobileNetV2 model in the project directory:
   - The model should be named `mobilenetv2_model.h5`
   - If you want to use custom class labels, create a `class_labels.json` file

5. Run the application:
   ```
   streamlit run app1.py
   ```

## Project Structure

```
pest-detection-webapp/
├── app1.py                # Main streanlit application
├── mobilenetv2_model.h5  # Your trained model file
├── class_labels.json     # Optional: custom class labels
├── requirements.txt      # Python dependencies
```

## Usage

1. Allow camera access when prompted by your browser
2. Position the pest inside the green box on the screen
3. Click "Capture & Analyze" to process the current frame
4. View the detailed analysis results below the video feed

## Notes

- The default class labels are: aphids, armyworm, beetle, bollworm, grasshopper, mites, mosquito, sawfly, and stem_borer
- For production use, ensure you have a properly trained model specific to your pest identification needs
- The application will create a directory for storing captured images