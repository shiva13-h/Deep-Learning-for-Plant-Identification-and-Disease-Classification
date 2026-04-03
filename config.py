"""
Configuration file for Flask application
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class"""
    
    # Base directory
    basedir = os.path.abspath(os.path.dirname(__file__))
    
    # Secret key for session management
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Database configuration - use absolute path for Windows compatibility
    db_path = os.path.join(basedir, 'instance', 'database.db')
    # Convert Windows backslashes to forward slashes for SQLite URI
    db_path = db_path.replace('\\', '/')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or f'sqlite:///{db_path}'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'connect_args': {
            'check_same_thread': False,
            'timeout': 30
        },
        'pool_pre_ping': True,
        'pool_recycle': 300,
    }
    
    # Upload folder configuration
    UPLOAD_FOLDER = os.path.join(basedir, 'static', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # Model configuration
    MODEL_PATH = os.path.join(basedir, 'model', 'disease_model_tf2.keras')  # Primary model
    MODEL_PATH_H5 = os.path.join(basedir, 'model', 'my_cnn_model.h5')  # Fallback model
    IMAGE_SIZE = (150, 150)  # Model expects 150x150 images
    
    # Application settings
    APP_NAME = 'AgriVision AI'
    VERSION = '1.0.0'
    
    # OpenAI GPT Configuration (Optional)
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    ENABLE_GPT_INSIGHTS = os.environ.get('ENABLE_GPT_INSIGHTS', 'False').lower() == 'true'
