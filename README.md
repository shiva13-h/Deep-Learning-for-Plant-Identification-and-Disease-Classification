# AgriVision AI - Plant Disease Detection System

A professional, production-ready Flask web application that uses deep learning to detect plant diseases from leaf images. This system provides instant AI-powered diagnosis with treatment recommendations to help farmers and agricultural professionals protect their crops.

## Features

- **AI-Powered Detection**: Leverages deep learning models for accurate disease identification
- **Instant Results**: Get predictions within seconds with confidence scores
- **Treatment Recommendations**: Receive actionable advice for each detected disease
- **Prediction History**: Track all your analyses with detailed records and thumbnails
- **Professional UI**: Clean, modern interface with Bootstrap 5 and Font Awesome icons
- **Image Preview**: See your uploaded image before analysis
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Database Integration**: SQLite database for persistent storage of predictions

## Technology Stack

### Backend
- **Python 3.10+**
- **Flask** - Web framework
- **Flask-SQLAlchemy** - ORM for database operations
- **SQLite** - Lightweight database
- **TensorFlow/Keras** - Machine learning model
- **Pillow** - Image preprocessing

### Frontend
- **HTML5** - Structure
- **CSS3** - Styling with custom animations
- **Bootstrap 5** - Responsive UI framework
- **Font Awesome 6** - Professional icons
- **JavaScript** - Client-side validation and image preview
- **Jinja2** - Template engine

## Project Structure

```
plant-disease-detection/
│
├── app.py                      # Main Flask application
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
│
├── model/
│   ├── plant_model.h5         # Trained ML model (to be added)
│   └── README.md              # Model documentation
│
├── static/
│   ├── css/
│   │   └── styles.css         # Custom stylesheet
│   ├── uploads/               # Uploaded images storage
│   └── images/                # Static images
│
├── templates/
│   ├── base.html              # Base template
│   ├── index.html             # Landing page
│   ├── upload.html            # Upload page
│   ├── result.html            # Results display
│   └── history.html           # Prediction history
│
└── instance/
    └── database.db            # SQLite database (auto-generated)
```

## Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step-by-Step Setup

1. **Clone or Download the Project**
   ```bash
   cd plant-disease-detection
   ```

2. **Create Virtual Environment**
   
   **Windows:**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   
   **Note:** TensorFlow installation might take several minutes. For CPU-only version, edit `requirements.txt` and replace `tensorflow` with `tensorflow-cpu`.

4. **Verify Installation**
   ```bash
   python -c "import flask; import tensorflow; print('Installation successful!')"
   ```

## Database Setup

The database is automatically initialized when you first run the application. However, you can manually initialize it:

```bash
python
>>> from app import app, db
>>> with app.app_context():
...     db.create_all()
>>> exit()
```

## Running the Application

1. **Activate Virtual Environment** (if not already activated)
   ```bash
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

2. **Start the Flask Server**
   ```bash
   python app.py
   ```

3. **Access the Application**
   
   Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

4. **Stop the Server**
   
   Press `Ctrl+C` in the terminal

## Usage Guide

### 1. Home Page
- View system overview and features
- Click "Start Detection" to begin

### 2. Upload Image
- Click "Select Image" and choose a leaf photo
- Preview the image before submission
- Supported formats: JPG, JPEG, PNG (max 16MB)
- Click "Analyze Image" to process

### 3. View Results
- See detected disease name
- Check confidence percentage
- Read treatment recommendations
- Analyze another image or view history

### 4. Prediction History
- View all past analyses
- See statistics and trends
- Access image thumbnails
- Track confidence scores over time

## Training Your Own Model

The application currently uses mock predictions. To use a real model:

1. **Prepare Dataset**
   - Collect labeled images of plant diseases
   - Organize into class folders
   - Split into train/validation/test sets

2. **Train Model**
   ```python
   # Example training script
   import tensorflow as tf
   from tensorflow.keras.applications import MobileNetV2
   from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
   from tensorflow.keras.models import Model
   
   # Load base model
   base_model = MobileNetV2(weights='imagenet', include_top=False, 
                            input_shape=(224, 224, 3))
   
   # Add custom layers
   x = base_model.output
   x = GlobalAveragePooling2D()(x)
   x = Dense(128, activation='relu')(x)
   predictions = Dense(7, activation='softmax')(x)  # 7 classes
   
   model = Model(inputs=base_model.input, outputs=predictions)
   
   # Compile and train
   model.compile(optimizer='adam', 
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
   
   # Train with your data
   # model.fit(train_data, validation_data, epochs=20)
   
   # Save model
   model.save('model/plant_model.h5')
   ```

3. **Update Class Names**
   
   Edit the `class_names` list in `app.py` to match your trained classes.

4. **Place Model File**
   ```
   model/plant_model.h5
   ```

## Configuration

Edit `config.py` to customize:

- **SECRET_KEY**: Change in production for security
- **MAX_CONTENT_LENGTH**: Adjust maximum upload size
- **IMAGE_SIZE**: Modify input image dimensions
- **ALLOWED_EXTENSIONS**: Add/remove file types

## Production Deployment

### Using Gunicorn (Linux/macOS)
```bash
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

### Using Waitress (Windows)
```bash
waitress-serve --host=0.0.0.0 --port=8000 app:app
```

### Environment Variables
```bash
export SECRET_KEY='your-secret-key-here'
export DATABASE_URL='your-database-url'
```

## Troubleshooting

### TensorFlow Installation Issues
- **GPU Version**: Requires CUDA and cuDNN
- **CPU Version**: Use `tensorflow-cpu` for faster installation
- **M1/M2 Mac**: Use `tensorflow-macos` and `tensorflow-metal`

### Database Errors
```bash
# Delete and recreate database
rm instance/database.db
python app.py
```

### Port Already in Use
```bash
# Change port in app.py
app.run(debug=True, port=5001)
```

## Future Improvements

- [ ] User authentication and profiles
- [ ] Multiple language support
- [ ] Mobile application (React Native/Flutter)
- [ ] Real-time disease tracking dashboard
- [ ] Integration with weather APIs
- [ ] PDF report generation
- [ ] Email notifications
- [ ] API endpoints for third-party integration
- [ ] Advanced analytics and insights
- [ ] Multi-crop support with crop-specific models
- [ ] Treatment product recommendations
- [ ] Expert consultation booking system

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available for educational and commercial use.

## Support

For issues, questions, or suggestions:
- Create an issue in the repository
- Contact: support@agrivision-ai.example.com

## Acknowledgments

- **PlantVillage Dataset** - For training data
- **TensorFlow Team** - For the ML framework
- **Bootstrap Team** - For the UI framework
- **Font Awesome** - For professional icons

## Version History

- **v1.0.0** (2026-02-13)
  - Initial release
  - Core disease detection functionality
  - Prediction history tracking
  - Professional UI/UX
  - Mock prediction fallback

---

**Made with care for the agricultural community**

*Protecting crops, one leaf at a time.*
