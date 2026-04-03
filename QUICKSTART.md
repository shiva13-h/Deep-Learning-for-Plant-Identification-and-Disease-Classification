# Quick Start Guide
# AgriVision AI - Plant Disease Detection System

## Setup (5 minutes)

1. Open Terminal/Command Prompt in project folder

2. Create virtual environment:
   Windows:     python -m venv venv
   Mac/Linux:   python3 -m venv venv

3. Activate virtual environment:
   Windows:     venv\Scripts\activate
   Mac/Linux:   source venv/bin/activate

4. Install dependencies:
   pip install -r requirements.txt
   (This may take 5-10 minutes for TensorFlow)

5. Run the application:
   python app.py

6. Open browser:
   http://127.0.0.1:5000

## Usage

1. Click "Start Detection" or "Detect" in navigation
2. Upload a plant leaf image (JPG/PNG)
3. View results with disease name and confidence
4. Check "History" to see all past predictions

## Troubleshooting

- Port in use? Change port in app.py: app.run(debug=True, port=5001)
- TensorFlow error? Use: pip install tensorflow-cpu (faster for CPU)
- Database error? Delete instance/database.db and restart

## Notes

- Mock predictions enabled (no trained model required)
- To add real model: Place plant_model.h5 in model/ folder
- Max upload size: 16MB
- Supported formats: JPG, JPEG, PNG

Enjoy using AgriVision AI!
