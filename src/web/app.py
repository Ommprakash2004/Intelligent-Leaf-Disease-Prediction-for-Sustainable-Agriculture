import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf

# Import our custom modules
sys_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
sys.path.append(sys_path)
from src.data.preprocess import preprocess_image_for_prediction
from src.models.model_builder import LeafDiseaseModel

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'leaf_disease_prediction_secret_key'

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/uploads')
MODEL_PATH = os.path.join(sys_path, 'models/best_model.h5')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Disease classes (these should match your model's output classes)
DISEASE_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy'
]

# Disease information database
DISEASE_INFO = {
    'Apple___Apple_scab': {
        'description': 'Apple scab is a common disease of apple trees caused by the fungus Venturia inaequalis.',
        'symptoms': 'Dark, scabby lesions on leaves and fruit.',
        'treatment': 'Apply fungicides early in the growing season. Remove and destroy fallen leaves.'
    },
    'Apple___Black_rot': {
        'description': 'Black rot is a fungal disease caused by Botryosphaeria obtusa affecting apples.',
        'symptoms': 'Circular lesions on leaves and rotting fruit with concentric rings.',
        'treatment': 'Prune out diseased branches. Apply fungicides during the growing season.'
    },
    'Apple___Cedar_apple_rust': {
        'description': 'Cedar apple rust is caused by the fungus Gymnosporangium juniperi-virginianae.',
        'symptoms': 'Bright orange-yellow spots on leaves and fruit.',
        'treatment': 'Remove nearby cedar trees if possible. Apply fungicides in spring.'
    },
    'Apple___healthy': {
        'description': 'Healthy apple leaves show no signs of disease.',
        'symptoms': 'Vibrant green color, no spots or lesions.',
        'treatment': 'Continue regular maintenance and preventive care.'
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'description': 'Gray leaf spot is a fungal disease caused by Cercospora zeae-maydis.',
        'symptoms': 'Rectangular gray to tan lesions on corn leaves.',
        'treatment': 'Rotate crops. Plant resistant varieties. Apply fungicides if severe.'
    },
    'Corn_(maize)___Common_rust': {
        'description': 'Common rust is caused by the fungus Puccinia sorghi.',
        'symptoms': 'Small, circular to elongated, powdery, reddish-brown pustules on both leaf surfaces.',
        'treatment': 'Plant resistant hybrids. Apply fungicides at first sign of disease.'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'description': 'Northern leaf blight is caused by the fungus Exserohilum turcicum.',
        'symptoms': 'Large, cigar-shaped lesions that are grayish-green to tan in color.',
        'treatment': 'Plant resistant varieties. Rotate crops. Apply fungicides.'
    },
    'Corn_(maize)___healthy': {
        'description': 'Healthy corn leaves show no signs of disease.',
        'symptoms': 'Uniform green color, no spots or lesions.',
        'treatment': 'Continue regular maintenance and preventive care.'
    },
    'Potato___Early_blight': {
        'description': 'Early blight is caused by the fungus Alternaria solani.',
        'symptoms': 'Dark brown to black lesions with concentric rings, yellowing of surrounding tissue.',
        'treatment': 'Rotate crops. Remove infected plants. Apply fungicides preventatively.'
    },
    'Potato___Late_blight': {
        'description': 'Late blight is caused by the oomycete Phytophthora infestans.',
        'symptoms': 'Water-soaked lesions that quickly turn brown to black, white fuzzy growth on undersides of leaves.',
        'treatment': 'Remove infected plants immediately. Apply fungicides preventatively. Plant resistant varieties.'
    },
    'Potato___healthy': {
        'description': 'Healthy potato leaves show no signs of disease.',
        'symptoms': 'Uniform green color, no spots or lesions.',
        'treatment': 'Continue regular maintenance and preventive care.'
    }
}

# Load the model at startup
model = None

def load_model():
    global model
    try:
        # Check if model file exists
        if os.path.exists(MODEL_PATH):
            # Load the saved model
            model = LeafDiseaseModel()
            model.load_model(MODEL_PATH)
        else:
            print(f"Warning: Model file not found at {MODEL_PATH}. Using a new model.")
            # Initialize a new model (this won't be trained)
            model = LeafDiseaseModel(model_type='transfer_learning', num_classes=len(DISEASE_CLASSES))
            model.build_model()
            model.compile_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Preprocess the image
        preprocessed_image = preprocess_image_for_prediction(file_path)
        
        if preprocessed_image is None:
            flash('Error processing the image')
            return redirect(url_for('index'))
        
        # Make prediction
        if model is not None:
            try:
                predictions = model.predict(preprocessed_image)
                predicted_class_index = np.argmax(predictions[0])
                predicted_class = DISEASE_CLASSES[predicted_class_index]
                confidence = float(predictions[0][predicted_class_index]) * 100
                
                # Get disease information
                disease_info = DISEASE_INFO.get(predicted_class, {
                    'description': 'Information not available',
                    'symptoms': 'Information not available',
                    'treatment': 'Information not available'
                })
                
                return render_template('result.html', 
                                      filename=filename,
                                      disease=predicted_class.replace('___', ' - '),
                                      confidence=confidence,
                                      description=disease_info['description'],
                                      symptoms=disease_info['symptoms'],
                                      treatment=disease_info['treatment'])
            except Exception as e:
                flash(f'Error making prediction: {str(e)}')
                return redirect(url_for('index'))
        else:
            flash('Model not loaded. Please try again later.')
            return redirect(url_for('index'))
    
    flash('Invalid file type. Please upload a PNG, JPG, or JPEG image.')
    return redirect(url_for('index'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Preprocess the image
        preprocessed_image = preprocess_image_for_prediction(file_path)
        
        if preprocessed_image is None:
            return jsonify({'error': 'Error processing the image'}), 500
        
        # Make prediction
        if model is not None:
            try:
                predictions = model.predict(preprocessed_image)
                predicted_class_index = np.argmax(predictions[0])
                predicted_class = DISEASE_CLASSES[predicted_class_index]
                confidence = float(predictions[0][predicted_class_index]) * 100
                
                # Get disease information
                disease_info = DISEASE_INFO.get(predicted_class, {
                    'description': 'Information not available',
                    'symptoms': 'Information not available',
                    'treatment': 'Information not available'
                })
                
                return jsonify({
                    'disease': predicted_class.replace('___', ' - '),
                    'confidence': confidence,
                    'description': disease_info['description'],
                    'symptoms': disease_info['symptoms'],
                    'treatment': disease_info['treatment'],
                    'image_url': url_for('static', filename=f'uploads/{filename}')
                })
            except Exception as e:
                return jsonify({'error': f'Error making prediction: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Model not loaded. Please try again later.'}), 503
    
    return jsonify({'error': 'Invalid file type. Please upload a PNG, JPG, or JPEG image.'}), 400

# Load the model when the app starts
# Flask 2.0+ removed before_first_request
# Initialize the model at startup
with app.app_context():
    load_model()

if __name__ == '__main__':
    # Create the upload folder if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)