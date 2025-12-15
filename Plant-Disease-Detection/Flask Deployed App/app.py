import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import json
import joblib
import timm
from torchvision import transforms

# =====================================================
# LOAD DISEASE INFO
# =====================================================
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# =====================================================
# DEVICE SETUP
# =====================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =====================================================
# LOAD ENSEMBLE INFO
# =====================================================
with open('ensemble_info.json', 'r') as f:
    ensemble_info = json.load(f)

LOCAL_CLASSES = ensemble_info['local_model_classes']
COLAB_CLASSES = ensemble_info['colab_model_classes']
ALL_CLASSES = ensemble_info['all_class_names']
NUM_LOCAL = len(LOCAL_CLASSES)
NUM_COLAB = len(COLAB_CLASSES)

print(f"Local classes: {NUM_LOCAL}")
print(f"Colab classes: {NUM_COLAB}")
print(f"Output classes: {len(ALL_CLASSES)}")

# =====================================================
# CREATE TIMM EFFICIENTNET MODEL
# =====================================================
def create_efficientnet(num_classes):
    model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=num_classes)
    return model

# =====================================================
# LOAD MODELS
# =====================================================
print("Loading Local Model (34 classes)...")
local_model = create_efficientnet(NUM_LOCAL)
local_checkpoint = torch.load('local_model.pth', map_location=device, weights_only=False)
local_model.load_state_dict(local_checkpoint['model_state_dict'])
local_model = local_model.to(device)
local_model.eval()
print("✅ Local model loaded!")

print("Loading Colab Model (38 classes)...")
colab_model = create_efficientnet(NUM_COLAB)
colab_checkpoint = torch.load('pakistan_model_best.pth', map_location=device, weights_only=False)
colab_model.load_state_dict(colab_checkpoint['model_state_dict'])
colab_model = colab_model.to(device)
colab_model.eval()
print("✅ Colab model loaded!")

# =====================================================
# LOAD RANDOM FOREST ENSEMBLE
# =====================================================
print("Loading Random Forest Ensemble...")
rf_model = joblib.load('ensemble_rf_model.joblib')
print("✅ Ensemble model loaded!")

# =====================================================
# IMAGE TRANSFORMS
# =====================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =====================================================
# ENSEMBLE PREDICTION FUNCTION
# =====================================================
def get_ensemble_features(image_tensor):
    """Get combined features from both models"""
    with torch.no_grad():
        # Local model prediction
        local_output = local_model(image_tensor)
        local_probs = torch.softmax(local_output, dim=1).cpu().numpy()[0]
        
        # Colab model prediction
        colab_output = colab_model(image_tensor)
        colab_probs = torch.softmax(colab_output, dim=1).cpu().numpy()[0]
        
        # Additional features
        local_max = np.max(local_probs)
        colab_max = np.max(colab_probs)
        local_entropy = -np.sum(local_probs * np.log(local_probs + 1e-10))
        colab_entropy = -np.sum(colab_probs * np.log(colab_probs + 1e-10))
        
        # Combine features
        features = np.concatenate([
            local_probs,
            colab_probs,
            [local_max, colab_max, local_entropy, colab_entropy]
        ])
        
    return features, local_probs, colab_probs

def prediction(image_path):
    """Ensemble prediction using Random Forest"""
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get features from both models
    features, local_probs, colab_probs = get_ensemble_features(image_tensor)
    
    # Random Forest prediction
    rf_pred = rf_model.predict(features.reshape(1, -1))[0]
    
    # Get confidence
    rf_proba = rf_model.predict_proba(features.reshape(1, -1))[0]
    confidence = np.max(rf_proba) * 100
    
    print(f"🌿 Predicted: {ALL_CLASSES[rf_pred]} | Confidence: {confidence:.2f}%")
    
    return rf_pred

# =====================================================
# FLASK APP
# =====================================================
app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(f"📸 Image: {file_path}")
        
        pred = prediction(file_path)
        
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        
        return render_template('submit.html', 
                             title=title, 
                             desc=description, 
                             prevent=prevent, 
                             image_url=image_url,
                             user_image='/' + file_path.replace('\\', '/'),
                             pred=pred,
                             sname=supplement_name, 
                             simage=supplement_image_url, 
                             buy_link=supplement_buy_link)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', 
                          supplement_image=list(supplement_info['supplement image']),
                          supplement_name=list(supplement_info['supplement name']), 
                          disease=list(disease_info['disease_name']), 
                          buy=list(supplement_info['buy link']))

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌱 KISAAN MADADGAAR - Plant Disease Detection")
    print("="*60)
    print(f"📊 Ensemble: Local ({NUM_LOCAL}) + Colab ({NUM_COLAB}) → {len(ALL_CLASSES)} classes")
    print(f"🎯 Ensemble Accuracy: {ensemble_info['ensemble_accuracy']:.2f}%")
    print("="*60 + "\n")
    app.run(debug=True)
