# Model Format Information

## You have 2 model files:

### 1. disease_model_tf2.keras ✓ RECOMMENDED
- **Format:** Native Keras (TensorFlow 2.x)
- **Status:** Currently being used
- **Advantages:**
  - Newer format (TF 2.13+)
  - Better compatibility with TensorFlow 2.x
  - Faster loading
  - More reliable
  - Future-proof

### 2. my_cnn_model.h5 (Fallback)
- **Format:** HDF5 (older format)
- **Status:** Backup only
- **Advantages:**
  - Compatible with older TensorFlow versions
  - Widely supported

## Current Configuration

Your app is configured to:
1. **Try .keras first** (disease_model_tf2.keras)
2. **Fall back to .h5** if .keras not found (my_cnn_model.h5)
3. **Use mock predictions** if neither exists

## Model Details

- **Input Size:** 150x150 pixels
- **Classes:** 38 plant diseases
- **Dataset:** PlantVillage
- **Plants Covered:**
  - Apple (4 classes)
  - Blueberry (1 class)
  - Cherry (2 classes)
  - Corn (4 classes)
  - Grape (4 classes)
  - Orange (1 class)
  - Peach (2 classes)
  - Pepper Bell (2 classes)
  - Potato (3 classes)
  - Raspberry (1 class)
  - Soybean (1 class)
  - Squash (1 class)
  - Strawberry (2 classes)
  - Tomato (10 classes)

## Which Format to Keep?

**Keep both files** for maximum compatibility:
- App will automatically use the best available format
- .keras is preferred for Python 3.11.0 + TensorFlow 2.15+
- .h5 serves as backup

## Verification

Run the app to see which model loaded:
```bash
python app.py
```

Check the console output:
- "✓ Model loaded successfully (.keras format)" → Using .keras
- "✓ Model loaded successfully (.h5 format)" → Using .h5
- "⚠ No model file found" → Using mock predictions
