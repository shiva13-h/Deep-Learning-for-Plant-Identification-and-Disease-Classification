"""
Plant Disease Detection Application
Main Flask application with routes and business logic
"""
import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import json

# Initialize Flask app
app = Flask(__name__)
app.config.from_object('config.Config')

# Debug: Print GPT configuration on startup
print(f"\n=== GPT Configuration ===")
print(f"ENABLE_GPT_INSIGHTS: {app.config.get('ENABLE_GPT_INSIGHTS')}")
print(f"OPENAI_API_KEY present: {bool(app.config.get('OPENAI_API_KEY'))}")
print(f"========================\n")

# Debug: Print database URI
print(f"Database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")

# Ensure critical directories exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'instance'), exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
db = SQLAlchemy(app)


# ======================== DATABASE MODELS ========================

class Prediction(db.Model):
    """Model for storing prediction history"""
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    image_name = db.Column(db.String(255), nullable=False)
    predicted_class = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f'<Prediction {self.predicted_class} - {self.confidence}%>'


# ======================== HELPER FUNCTIONS ========================

def allowed_file(filename):
    """Check if uploaded file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def preprocess_image(image_path):
    """
    Preprocess image for model prediction
    - Resize to 150x150 (model input size)
    - Normalize pixel values to [0, 1]
    - Add batch dimension
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(app.config['IMAGE_SIZE'])
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def load_model():
    """
    Load trained model from model folder
    Tries .keras format first (TF 2.x), then falls back to .h5
    Returns None if model file doesn't exist (uses mock prediction instead)
    """
    import tensorflow as tf
    
    # Get base directory
    basedir = os.path.abspath(os.path.dirname(__file__))
    
    # Try .keras format first (newer, recommended)
    keras_path = os.path.join(basedir, 'model', 'disease_model_tf2.keras')
    h5_path = os.path.join(basedir, 'model', 'my_cnn_model.h5')
    
    if os.path.exists(keras_path):
        try:
            print(f"Loading model from: {keras_path}")
            model = tf.keras.models.load_model(keras_path)
            print("✓ Model loaded successfully (.keras format)")
            return model
        except Exception as e:
            print(f"Error loading .keras model: {e}")
    
    # Fall back to .h5 format
    if os.path.exists(h5_path):
        try:
            print(f"Loading model from: {h5_path}")
            model = tf.keras.models.load_model(h5_path)
            print("✓ Model loaded successfully (.h5 format)")
            return model
        except Exception as e:
            print(f"Error loading .h5 model: {e}")
    
    print("⚠ No model file found. Using mock predictions.")
    return None


def mock_prediction():
    """
    Mock prediction function for testing when model is not available
    Returns random disease with confidence
    """
    import random
    
    diseases = [
        ('Tomato Late Blight', 'Remove infected leaves and apply fungicide. Ensure proper spacing for air circulation.'),
        ('Potato Early Blight', 'Apply copper-based fungicide. Practice crop rotation and remove plant debris.'),
        ('Corn Common Rust', 'Use resistant varieties. Apply fungicide if infection is severe.'),
        ('Grape Black Rot', 'Remove mummified fruits. Apply preventive fungicide sprays.'),
        ('Apple Scab', 'Rake and destroy fallen leaves. Apply fungicide during wet periods.'),
        ('Pepper Bell Bacterial Spot', 'Use disease-free seeds. Apply copper sprays and avoid overhead irrigation.'),
        ('Healthy Leaf', 'Plant is healthy. Continue regular maintenance and monitoring.')
    ]
    
    disease_name, recommendation = random.choice(diseases)
    confidence = round(random.uniform(85, 98), 2)
    
    return disease_name, confidence, recommendation


def predict_disease(image_path):
    """
    Predict disease from image
    Returns: disease_name, confidence, recommendation
    """
    # Preprocess image
    processed_img = preprocess_image(image_path)
    if processed_img is None:
        return None, None, None
    
    # Load model
    model = load_model()
    
    # If model not available, use mock prediction
    if model is None:
        return mock_prediction()
    
    # Make prediction with real model
    try:
        predictions = model.predict(processed_img, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx]) * 100
        
        # PlantVillage dataset class names (38 classes)
        class_names = [
            'Apple Scab', 'Apple Black Rot', 'Apple Cedar Rust',
            'Apple Healthy', 'Blueberry Healthy', 'Cherry Powdery Mildew',
            'Cherry Healthy', 'Corn Cercospora Leaf Spot',
            'Corn Common Rust', 'Corn Healthy', 'Corn Northern Leaf Blight',
            'Grape Black Rot', 'Grape Esca (Black Measles)',
            'Grape Leaf Blight', 'Grape Healthy', 'Orange Haunglongbing',
            'Peach Bacterial Spot', 'Peach Healthy', 'Pepper Bell Bacterial Spot',
            'Pepper Bell Healthy', 'Potato Early Blight', 'Potato Late Blight',
            'Potato Healthy', 'Raspberry Healthy', 'Soybean Healthy',
            'Squash Powdery Mildew', 'Strawberry Leaf Scorch',
            'Strawberry Healthy', 'Tomato Bacterial Spot',
            'Tomato Early Blight', 'Tomato Late Blight', 'Tomato Leaf Mold',
            'Tomato Septoria Leaf Spot', 'Tomato Spider Mites',
            'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus',
            'Tomato Mosaic Virus', 'Tomato Healthy'
        ]
        
        disease_name = class_names[predicted_class_idx]
        
        # Comprehensive recommendations mapping
        recommendations = {
            'Apple Scab': 'Rake and destroy fallen leaves. Apply fungicide during wet periods. Use resistant varieties.',
            'Apple Black Rot': 'Prune infected branches. Remove mummified fruits. Apply fungicide before rain.',
            'Apple Cedar Rust': 'Remove nearby cedar trees if possible. Apply fungicide in early spring.',
            'Apple Healthy': 'Plant is healthy. Continue regular monitoring and good orchard practices.',
            'Blueberry Healthy': 'Plant is healthy. Maintain proper soil pH and continue regular care.',
            'Cherry Powdery Mildew': 'Improve air circulation. Apply sulfur-based fungicide. Avoid overhead watering.',
            'Cherry Healthy': 'Plant is healthy. Continue regular pruning and monitoring.',
            'Corn Cercospora Leaf Spot': 'Use resistant hybrids. Practice crop rotation. Apply fungicide if severe.',
            'Corn Common Rust': 'Use resistant varieties. Apply fungicide only if infection is severe.',
            'Corn Healthy': 'Plant is healthy. Continue regular monitoring and weed control.',
            'Corn Northern Leaf Blight': 'Plant resistant hybrids. Practice crop rotation. Remove infected debris.',
            'Grape Black Rot': 'Remove mummified fruits. Apply preventive fungicide sprays during growing season.',
            'Grape Esca (Black Measles)': 'Prune infected vines. No effective treatment available. Remove severely infected plants.',
            'Grape Leaf Blight': 'Improve air circulation. Apply copper-based fungicide. Remove infected leaves.',
            'Grape Healthy': 'Plant is healthy. Continue proper vineyard management practices.',
            'Orange Haunglongbing': 'No cure available. Remove infected trees. Control psyllid insects. Use certified disease-free plants.',
            'Peach Bacterial Spot': 'Use resistant varieties. Apply copper sprays. Avoid overhead irrigation.',
            'Peach Healthy': 'Plant is healthy. Continue regular pruning and pest monitoring.',
            'Pepper Bell Bacterial Spot': 'Use disease-free seeds. Apply copper sprays. Avoid overhead watering.',
            'Pepper Bell Healthy': 'Plant is healthy. Continue proper watering and fertilization.',
            'Potato Early Blight': 'Apply copper-based fungicide. Practice crop rotation. Remove plant debris.',
            'Potato Late Blight': 'Remove infected plants immediately. Apply fungicide. Ensure proper spacing.',
            'Potato Healthy': 'Plant is healthy. Continue regular monitoring and proper cultural practices.',
            'Raspberry Healthy': 'Plant is healthy. Maintain good air circulation and regular pruning.',
            'Soybean Healthy': 'Plant is healthy. Continue crop rotation and pest monitoring.',
            'Squash Powdery Mildew': 'Improve air circulation. Apply sulfur or neem oil. Remove infected leaves.',
            'Strawberry Leaf Scorch': 'Remove infected leaves. Improve air circulation. Apply appropriate fungicide.',
            'Strawberry Healthy': 'Plant is healthy. Continue proper watering and runner management.',
            'Tomato Bacterial Spot': 'Use disease-free transplants. Apply copper sprays. Avoid overhead watering.',
            'Tomato Early Blight': 'Apply fungicide preventively. Mulch to prevent soil splash. Practice crop rotation.',
            'Tomato Late Blight': 'Remove infected plants. Apply fungicide immediately. Ensure good air circulation.',
            'Tomato Leaf Mold': 'Improve greenhouse ventilation. Reduce humidity. Apply appropriate fungicide.',
            'Tomato Septoria Leaf Spot': 'Remove infected leaves. Apply fungicide. Mulch around plants.',
            'Tomato Spider Mites': 'Spray with water to dislodge mites. Use insecticidal soap or neem oil.',
            'Tomato Target Spot': 'Apply fungicide. Improve air circulation. Remove infected plant parts.',
            'Tomato Yellow Leaf Curl Virus': 'Control whitefly vectors. Remove infected plants. Use resistant varieties.',
            'Tomato Mosaic Virus': 'Remove infected plants. Sanitize tools. Use resistant varieties. Control aphids.',
            'Tomato Healthy': 'Plant is healthy. Continue regular maintenance and monitoring.'
        }
        
        recommendation = recommendations.get(disease_name, 'Consult with an agricultural expert for specific treatment recommendations.')
        
        return disease_name, round(confidence, 2), recommendation
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def get_gpt_insights(disease_name, confidence, image_path=None):
    """
    Get enhanced insights from OpenAI GPT-4o
    Extracts plant species name from disease classification
    Returns: enhanced_insights dict with plant_species or None if disabled/error
    """
    # Check if GPT is enabled
    print(f"DEBUG - GPT Enabled: {app.config.get('ENABLE_GPT_INSIGHTS')}")
    print(f"DEBUG - API Key Set: {bool(app.config.get('OPENAI_API_KEY'))}")
    
    if not app.config.get('ENABLE_GPT_INSIGHTS'):
        print("DEBUG - GPT disabled in config")
        return None
        
    if not app.config.get('OPENAI_API_KEY'):
        print("DEBUG - No OpenAI API key found")
        return None
    
    try:
        from openai import OpenAI
        
        print(f"DEBUG - Calling GPT-4o to extract plant name from: {disease_name}")
        client = OpenAI(api_key=app.config['OPENAI_API_KEY'])
        
        # Simple text-based extraction (no vision needed, cheaper and faster)
        messages = [
            {
                "role": "system",
                "content": "You are a botanical expert. Extract plant species from disease names."
            },
            {
                "role": "user",
                "content": f"""Extract the plant species from this disease classification: "{disease_name}"

Respond with ONLY the plant name in this format:
[Scientific name] ([Common name])

Examples:
- "Apple Scab" → Malus domestica (Apple)
- "Tomato Late Blight" → Solanum lycopersicum (Tomato)  
- "Grape Black Rot" → Vitis vinifera (Grape)
- "Corn Common Rust" → Zea mays (Corn)

Your response:"""
            }
        ]
        
        # Call GPT-4o (text-only, no vision)
        print(f"DEBUG - Sending request to GPT-4o...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=30,
            temperature=0.3
        )
        
        print(f"DEBUG - Got response from GPT-4o")
        print(f"DEBUG - Response object: {response}")
        insights_text = response.choices[0].message.content.strip()
        print(f"DEBUG - Raw insights text: '{insights_text}'")
        print(f"DEBUG - Insights text length: {len(insights_text)}")
        print(f"DEBUG - Insights text type: {type(insights_text)}")
        
        # The response is directly the plant species name
        plant_species = insights_text if insights_text else "Unknown"
        
        print(f"✓ Extracted plant: {plant_species}")
        print(f"DEBUG - Tokens used: {response.usage.total_tokens}")
        
        return {
            'insights': insights_text,
            'plant_species': plant_species,
            'model': 'gpt-4o',
            'tokens_used': response.usage.total_tokens
        }
        
    except Exception as e:
        print(f"❌ Error getting GPT insights: {e}")
        import traceback
        traceback.print_exc()
        return None


# ======================== ROUTES ========================

@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')


@app.route('/upload')
def upload():
    """Upload page"""
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    
    # Check if file is in request
    if 'file' not in request.files:
        flash('No file uploaded', 'danger')
        return redirect(url_for('upload'))
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('upload'))
    
    # Validate file type
    if not allowed_file(file.filename):
        flash('Invalid file type. Please upload a JPG, JPEG, or PNG image.', 'danger')
        return redirect(url_for('upload'))
    
    try:
        # Save file securely
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        print(f"✓ File saved: {filepath}")
        
        # Make prediction
        print("Starting prediction...")
        disease_name, confidence, recommendation = predict_disease(filepath)
        print(f"Prediction result: {disease_name}, {confidence}%")
        
        if disease_name is None:
            print("ERROR: Prediction returned None")
            flash('Error processing image. Please try again.', 'danger')
            return redirect(url_for('upload'))
        
        # Get GPT insights if enabled
        print("Fetching GPT insights...")
        gpt_insights = get_gpt_insights(disease_name, confidence, filepath)
        if gpt_insights:
            print(f"✓ GPT insights received: {gpt_insights['tokens_used']} tokens")
        else:
            print("⚠ GPT insights disabled or failed")
        
        # Save prediction to database (optional - app works even if this fails)
        try:
            print("Saving to database...")
            new_prediction = Prediction(
                image_name=unique_filename,
                predicted_class=disease_name,
                confidence=confidence
            )
            db.session.add(new_prediction)
            db.session.commit()
            print("✓ Saved to database")
        except Exception as db_error:
            print(f"⚠ Database save failed (non-critical): {str(db_error)}")
            # Rollback the session to prevent issues
            db.session.rollback()
        
        # Render result page
        print("Rendering result page...")
        return render_template('result.html',
                             image_name=unique_filename,
                             disease_name=disease_name,
                             confidence=confidence,
                             recommendation=recommendation,
                             gpt_insights=gpt_insights)
        
    except Exception as e:
        print(f"❌ ERROR in predict route: {str(e)}")
        import traceback
        traceback.print_exc()
        flash(f'An error occurred: {str(e)}', 'danger')
        return redirect(url_for('upload'))


@app.route('/history')
def history():
    """Display prediction history (supports both DB and local storage)"""
    try:
        print("DEBUG - Attempting to query predictions...")
        print(f"DEBUG - Database URI: {app.config['SQLALCHEMY_DATABASE_URI']}")
        predictions = Prediction.query.order_by(Prediction.created_at.desc()).all()
        print(f"DEBUG - Found {len(predictions)} predictions")
        return render_template('history.html', predictions=predictions, use_local_storage=False)
    except Exception as e:
        print(f"⚠ History route error (falling back to local storage): {str(e)}")
        # Fall back to local storage mode if database fails
        return render_template('history.html', predictions=[], use_local_storage=True)


@app.route('/api/predictions/save', methods=['POST'])
def save_prediction_api():
    """API endpoint to receive prediction data from frontend"""
    try:
        data = request.get_json()
        
        # Still try to save to database but don't fail the response
        try:
            new_prediction = Prediction(
                image_name=data.get('image_name', 'unknown'),
                predicted_class=data.get('predicted_class', 'unknown'),
                confidence=float(data.get('confidence', 0))
            )
            db.session.add(new_prediction)
            db.session.commit()
        except Exception as db_err:
            print(f"⚠ Database save failed: {str(db_err)}")
            db.session.rollback()
        
        return jsonify({'status': 'success', 'message': 'Prediction saved to history'})
    except Exception as e:
        print(f"❌ API save error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 400


# ======================== ERROR HANDLERS ========================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    flash('Page not found', 'warning')
    return redirect(url_for('index'))


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    db.session.rollback()
    flash('An internal error occurred', 'danger')
    return redirect(url_for('index'))


# ======================== DATABASE INITIALIZATION ========================

def init_db():
    """Initialize database"""
    try:
        with app.app_context():
            db.create_all()
            print("✓ Database initialized successfully!")
    except Exception as e:
        print(f"⚠ Database initialization note: {e}")


# ======================== APPLICATION ENTRY POINT ========================

if __name__ == '__main__':
    # Run application
    print(f"\nStarting {app.config['APP_NAME']} v{app.config['VERSION']}")
    print("Application running at http://127.0.0.1:5000\n")
    
    # Initialize database on first request
    @app.before_request
    def initialize_database():
        """Initialize database on first request"""
        if not hasattr(app, '_database_initialized'):
            try:
                db.create_all()
                app._database_initialized = True
                print("✓ Database tables created/verified")
            except Exception as e:
                app._database_initialized = True  # Set to true anyway to prevent spam
                print(f"⚠ Database initialization failed: {e}")
                print("  → App will run without database history feature")
    
    # Run with reloader disabled to prevent TensorFlow reload loops
    app.run(debug=True, use_reloader=False)
